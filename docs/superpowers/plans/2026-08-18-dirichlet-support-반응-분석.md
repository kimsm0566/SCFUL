# Dirichlet Support 반응 분석 구현 계획

> **에이전트 작업자용:** 이 계획은 `superpowers:executing-plans`로 task별로 실행한다. 각 step은 checkbox로 완료 상태를 기록한다.

**목표:** 균등 데이터량 Dirichlet `alpha=0.1` client deletion에서 클래스별 삭제 support 비율과 original·재학습·중앙집중 full SCU의 retrain functional gap을 같은 schema로 기록·집계한다.

**구조:** `src/evaluation/support_response.py`는 split allocation과 class metric을 결합하는 순수 분석 모듈이다. `federated_deletion.py`는 original/retrain-only mode와 run별 `support_by_class.csv`를 제공한다. `support_response.py` experiment는 여러 run artifact를 읽어 aggregate CSV와 scatter figure를 만든다. Target-only forgetting과 기존 repair path는 이 계획에서 실행하지 않는다.

**기술:** Python 3.10, NumPy, PyTorch, matplotlib, pytest, mpi4py, CIFAR-10, CUDA, 10-client FedAvg.

## 전역 제약

- 삭제 대상은 client 0 전체이며 모든 model의 기준은 `retrain_without_client_0`이다.
- `rho = target_count / total_class_count`, `remaining_fraction = 1-rho`를 사용한다.
- `target_count < 500`인 label은 `not_targeted`로 표시해 support-group 평균에서 제외한다.
- `target_specific`은 retained count가 정확히 0인 경우만 사용하며 `low_support`보다 우선한다.
- Analysis-only run은 original/retrain만 만들고 `forget-only`, joint forgetting, naive repair, conflict-aware repair를 실행하지 않는다.
- 중앙집중 full SCU는 비교선에서만 raw `D_t`, pooled `D_r`를 일시적으로 사용한다. 새 federated method로 표현하지 않는다.
- 모든 새 Markdown 문서는 한국어로 작성한다.
- repository top-level은 Git repository가 아니다. commit, branch, worktree 작업을 수행하지 않는다.

---

### Task 1: 클래스 support와 metric 결합 모듈

**Files:**

- Create: `src/evaluation/support_response.py`
- Modify: `src/evaluation/__init__.py`
- Create: `src/tests/test_support_response.py`

**Interfaces:**

- Produces: `SupportGroupingConfig(min_target_samples, shared_high_max_deletion_fraction, low_support_min_deletion_fraction)`
- Produces: `build_class_support_records(class_labels, train_allocation, target_client_id, config) -> list[dict]`
- Produces: `merge_support_with_class_metrics(support_records, class_metric_records) -> list[dict]`
- Produces: `summarize_support_groups(records, candidate_models) -> list[dict]`

- [x] **Step 1: support record의 실패 test를 작성한다.**

```python
def test_build_class_support_records_assigns_groups_from_deletion_fraction():
    records = build_class_support_records(
        class_labels=np.array([0, 1, 2, 3]),
        train_allocation=np.array([[900, 3500, 500, 100], [4100, 1500, 4500, 4900]]),
        target_client_id=0,
        config=SupportGroupingConfig(min_target_samples=500),
    )

    assert records[0]["support_group"] == "shared_high"
    assert records[1]["support_group"] == "low_support"
    assert records[2]["support_group"] == "medium_support"
    assert records[3]["support_group"] == "not_targeted"
```

- [x] **Step 2: test가 helper 미정의로 실패하는지 확인한다.**

Run: `pytest -q src/tests/test_support_response.py::test_build_class_support_records_assigns_groups_from_deletion_fraction`

Expected: FAIL with import error for `src.evaluation.support_response`.

- [x] **Step 3: 최소 support helper를 구현한다.**

```python
@dataclass(frozen=True)
class SupportGroupingConfig:
    min_target_samples: int = 500
    shared_high_max_deletion_fraction: float = 0.20
    low_support_min_deletion_fraction: float = 0.60

def build_class_support_records(class_labels, train_allocation, target_client_id, config):
    records = []
    for column, label in enumerate(class_labels):
        target_count = int(train_allocation[target_client_id, column])
        total_count = int(train_allocation[:, column].sum())
        retained_count = total_count - target_count
        deletion_fraction = target_count / total_count
        records.append({...})
    return records
```

Group priority는 `target_specific`, `not_targeted`, `shared_high`, `low_support`, `medium_support` 순으로 명시적으로 구현한다. `total_count <= 0`, target id 범위 오류, allocation/class label shape 불일치는 `ValueError`로 거부한다.

- [x] **Step 4: metric merge와 group summary의 실패 test를 작성한다.**

```python
def test_merge_support_with_class_metrics_preserves_model_metrics_and_summary():
    records = merge_support_with_class_metrics(support_records, class_metric_records)
    assert records[0]["deletion_fraction"] == 0.18
    assert records[0]["model"] == "original"
    summary = summarize_support_groups(records, candidate_models=["original"])
    assert summary[0]["support_group"] == "shared_high"
    assert summary[0]["models"] == "original"
```

- [x] **Step 5: merge와 summary를 구현하고 module test를 통과시킨다.**

`merge_support_with_class_metrics()`는 class metric의 `group="class"`, `group_id`와 support record의 `label`을 exact match한다. 출력 record에는 `target_train_samples`, `retained_train_samples`, `total_train_samples`, `deletion_fraction`, `remaining_fraction`, `support_group`와 원래 metric field를 모두 둔다. `summarize_support_groups()`는 `not_targeted`를 제외하고 `(model, support_group)`별 sample-weighted mean과 label 수를 계산한다.

Run: `pytest -q src/tests/test_support_response.py`

Expected: PASS.

### Task 2: run별 support artifact와 original/retrain-only mode

**Files:**

- Modify: `src/experiments/federated_deletion.py`
- Create: `src/configs/federated_cifar10_dirichlet_01_support.json`
- Create: `src/configs/federated_cifar10_dirichlet_01_support_smoke.json`
- Modify: `src/tests/test_federated_deletion_smoke.py`

**Interfaces:**

- Consumes: `unlearning.mode == "analysis_original_retrain"`
- Produces: only `initial.pt`, `original.pt`, `retrain_without_client_0.pt`, `metrics/per_class.csv`, `metrics/support_by_class.csv`, and `metrics/support_group_summary.csv`

- [x] **Step 1: analysis-only config selection test를 작성한다.**

```python
def test_dirichlet_support_config_selects_original_retrain_analysis_only():
    config = load_federated_experiment_config(DIRICHLET_SUPPORT_CONFIG_PATH)
    assert config["split"]["alpha"] == 0.1
    assert config["unlearning"]["mode"] == "analysis_original_retrain"
    assert config["federation"]["rounds"] == 200
```

- [x] **Step 2: test가 unsupported mode로 실패하는지 확인한다.**

Run: `pytest -q src/tests/test_federated_deletion_smoke.py::test_dirichlet_support_config_selects_original_retrain_analysis_only`

Expected: FAIL with `unsupported unlearning.mode`.

- [x] **Step 3: validation과 no-unlearning branch를 최소 구현한다.**

`load_federated_experiment_config()`가 `analysis_original_retrain`을 허용한다. `run_federated_deletion_experiment()`는 original/retrain 직후 이 mode에서 client rank를 반환하고, server는 두 state만 checkpoint/evaluation 대상으로 둔다. 이전 `mode is None` branch의 legacy repair 동작은 변경하지 않는다.

```python
if analysis_original_retrain:
    checkpoints = {
        "original": original_state,
        "retrain_without_client_0": retrain_state,
    }
    training_history = original_history + retrain_history
    unlearning_diagnostics = {"analysis_only": True}
```

Evaluation 뒤 `build_class_support_records()`와 `merge_support_with_class_metrics()`를 호출해 support CSV와 group summary CSV를 작성한다. 동일 schema를 central full SCU mode에도 적용한다.

- [x] **Step 4: artifact schema를 CPU MPI smoke에서 검증한다.**

```python
def test_analysis_only_smoke_outputs_two_checkpoints_and_support_metrics(tmp_path):
    run_dir = run_federated_deletion_experiment(...)
    assert checkpoint_names == {"initial.pt", "original.pt", "retrain_without_client_0.pt"}
    assert (run_dir / "metrics" / "support_by_class.csv").exists()
```

- [x] **Step 5: CPU MPI smoke와 tests를 통과시킨다.**

Run:

```bash
mpirun --allow-run-as-root --oversubscribe -np 11 python src/main.py \
  --experiment federated_deletion \
  --config src/configs/federated_cifar10_dirichlet_01_support_smoke.json \
  --seed 0 \
  --output-root outputs/dirichlet_support_smoke
```

Expected: checkpoint 세 개와 두 support CSV가 생성되고 `joint_forgetting`, `naive_repair`, `conflict_aware_repair` checkpoint는 없다.

Run: `pytest -q src/tests/test_federated_deletion_smoke.py src/tests/test_support_response.py`

Expected: PASS.

### Task 3: 여러 run의 support-response 집계 entry point

**Files:**

- Create: `src/experiments/support_response.py`
- Modify: `src/main.py`
- Modify: `src/utils/option.py`
- Modify: `src/tests/test_main_cli.py`
- Modify: `src/tests/test_support_response.py`

**Interfaces:**

- Produces: `collect_support_response_runs(runs_root) -> list[Path]`
- Produces: `aggregate_support_response(run_directories, output_root, seed) -> Path`
- Produces: `screen_dirichlet_support(config_path, global_seeds) -> list[dict]`
- Consumes CLI: `python src/main.py --experiment support_response --runs-root <run-root> --output-root <analysis-root> --seed 0`

- [x] **Step 1: existing run scan의 실패 test를 작성한다.**

```python
def test_collect_support_response_runs_uses_only_completed_support_artifacts(tmp_path):
    complete = tmp_path / "run_complete"
    incomplete = tmp_path / "run_incomplete"
    (complete / "metrics").mkdir(parents=True)
    (complete / "metrics" / "support_by_class.csv").write_text("model,label\n")
    incomplete.mkdir()
    assert collect_support_response_runs(tmp_path) == [complete]
```

- [x] **Step 2: scan test가 helper 미정의로 실패하는지 확인한다.**

Run: `pytest -q src/tests/test_support_response.py::test_collect_support_response_runs_uses_only_completed_support_artifacts`

Expected: FAIL with import error.

- [x] **Step 3: aggregate CSV와 figure를 구현한다.**

`screen_dirichlet_support()`는 config의 `split.alpha`, `split.seed`, `federation.n_clients`, `unlearning.target_client_id`와 CIFAR-10 class count를 사용해 model 학습 없이 각 global seed의 allocation과 support record를 반환한다. `split_partition_seed()`를 재사용해 training run과 동일한 split seed를 계산한다.

각 input run의 `config.json`, `run_metadata.json`, `metrics/support_by_class.csv`를 읽는다. Run directory name에서 seed를 추론하지 않고 `run_metadata.json`의 recorded global/split seed를 사용한다. 모든 row에 `run_directory`, `global_seed`, `split_seed`를 추가한다.

`metrics/support_response.csv`에는 전체 merged row를 쓰고,
`metrics/support_group_summary.csv`에는 `(model, support_group)`별 label count,
deletion fraction mean, MSE/FG mean·standard deviation을 쓴다. Figure는 `original`의
`deletion_fraction` 대 `fg_output` scatter를 반드시 그리고, input에
`SCU_central_full`이 있으면 model별 scatter도 추가한다. Empty run set, support file 없는
run, header 없는 CSV는 `ValueError`로 명확히 중단한다.

- [x] **Step 4: aggregate artifact와 CLI parser test를 작성하고 통과시킨다.**

```python
def test_aggregate_support_response_writes_csv_and_figure(tmp_path):
    run_dir = make_minimal_completed_support_run(tmp_path / "runs")
    output = aggregate_support_response([run_dir], tmp_path / "analysis", seed=0)
    assert (output / "metrics" / "support_response.csv").exists()
    assert (output / "figures" / "original_support_response.png").exists()

def test_main_accepts_support_response_experiment():
    args = args_parser(["--experiment", "support_response", "--runs-root", "outputs/runs"])
    assert args.experiment == "support_response"
```

Run: `pytest -q src/tests/test_support_response.py src/tests/test_main_cli.py`

Expected: PASS.

### Task 4: support screening과 original/retrain 다중 seed 실행

**Files:**

- Create: `docs/experiment/08-18-01-dirichlet-support-반응-분석.md`
- Produces: `outputs/dirichlet_support_original_retrain/<timestamp>_seed-0`부터 seed 4까지 총 다섯 run
- Produces: `outputs/dirichlet_support_response_original_retrain/<timestamp>_seed-0`

- [x] **Step 1: model 학습 없이 seed 0--4 split을 screening한다.**

Run:

```bash
python -c "from src.experiments.support_response import screen_dirichlet_support; screen_dirichlet_support('src/configs/federated_cifar10_dirichlet_01_support.json', [0, 1, 2, 3, 4])"
```

Expected: 각 seed의 client 0 label별 target/retained count, `rho`, group이 출력되고, model metric을 보지 않고 seed 3의 label 1 `low_support`, label 7 `shared_high` 조건을 재확인한다.

- [x] **Step 2: seed 0--4 original/retrain-only CUDA run을 수행한다.**

Run each `seed` in `0 1 2 3 4`:

```bash
mpirun --allow-run-as-root --oversubscribe -np 11 python src/main.py \
  --experiment federated_deletion \
  --config src/configs/federated_cifar10_dirichlet_01_support.json \
  --seed <seed> \
  --output-root outputs/dirichlet_support_original_retrain
```

Expected: 각 run은 original/retrain checkpoint와 support CSV만 저장한다.

- [x] **Step 3: 다섯 run을 집계한다.**

Run:

```bash
python src/main.py \
  --experiment support_response \
  --runs-root outputs/dirichlet_support_original_retrain \
  --output-root outputs/dirichlet_support_response_original_retrain \
  --seed 0
```

Expected: 50 label row의 aggregate CSV, group summary CSV, original support-response scatter가 생성된다.

- [x] **Step 4: 원본·재학습 관계를 문서화한다.**

`08-18-01`에 seed별 support 구성, `rho`와 original functional gap의 scatter, group별 평균과 standard deviation을 기록한다. 관계가 약하거나 역방향이면 그 결과를 그대로 기록하며, global average만으로 class 변화 부재를 주장하지 않는다.

### Task 5: 사전 선택한 seed 3 중앙집중 full SCU 비교

**Files:**

- Create: `src/configs/federated_cifar10_dirichlet_01_centralized_full_scu.json`
- Create: `docs/experiment/08-18-02-dirichlet-full-scu-support-분석.md`
- Produces: `outputs/dirichlet_support_centralized_full_scu/<timestamp>_seed-3`
- Produces: `outputs/dirichlet_support_response_full_scu/<timestamp>_seed-0`

- [x] **Step 1: seed 3 support selection을 config-independent하게 재검증한다.**

`split_manifest.json` 또는 `screen_dirichlet_support()` output에서 label 1의 `rho >= 0.60`, label 7의 `rho <= 0.20`, 두 label의 target sample 수 `>= 500`을 확인한다. 조건이 달라지면 model metric을 보지 않고 다음 eligible seed를 선택하고 이유를 run log에 기록한다.

- [x] **Step 2: full SCU config test와 config를 추가한다.**

```python
def test_dirichlet_full_scu_config_uses_seed_three_and_full_mode():
    config = load_federated_experiment_config(DIRICHLET_FULL_SCU_CONFIG_PATH)
    assert config["unlearning"]["mode"] == "centralized_full_scu"
    assert config["federation"]["rounds"] == 200
    assert config["unlearning"]["scu_epochs"] == 200
```

Run: `pytest -q src/tests/test_federated_deletion_smoke.py::test_dirichlet_full_scu_config_uses_seed_three_and_full_mode`

Expected: PASS.

- [x] **Step 3: seed 3 full SCU CUDA run을 수행한다.**

Run:

```bash
mpirun --allow-run-as-root --oversubscribe -np 11 python src/main.py \
  --experiment federated_deletion \
  --config src/configs/federated_cifar10_dirichlet_01_centralized_full_scu.json \
  --seed 3 \
  --output-root outputs/dirichlet_support_centralized_full_scu
```

Expected: `original.pt`, `retrain_without_client_0.pt`, `SCU_central_full.pt`, class support CSV, group summary CSV가 생성된다. Target-only forgetting이나 repair checkpoint는 생성되지 않는다.

- [x] **Step 4: full SCU를 같은 support-response schema로 집계·문서화한다.**

Run:

```bash
python src/main.py \
  --experiment support_response \
  --runs-root outputs/dirichlet_support_centralized_full_scu \
  --output-root outputs/dirichlet_support_response_full_scu \
  --seed 0
```

`08-18-02`에는 label 1 low-support와 label 7 shared-high에서 original·retrain·full SCU의 MSE 및 functional gap을 직접 비교한다. central full SCU의 retained raw-data 사용 사실과 distributed repair가 아직 구현되지 않았음을 명시한다.

### Task 6: 전체 검증과 artifact 점검

**Files:**

- Modify: `docs/superpowers/plans/2026-08-18-dirichlet-support-반응-분석.md`

- [x] **Step 1: 전체 Python test와 compile check를 실행한다.**

Run:

```bash
pytest -q src/tests
python -m py_compile src/evaluation/support_response.py src/experiments/support_response.py src/experiments/federated_deletion.py src/main.py
```

Expected: zero failed tests and exit code 0.

- [x] **Step 2: artifact schema를 검증한다.**

각 run의 checkpoint 이름과 `support_by_class.csv` column을 확인한다. analysis-only run에는 target-only forgetting/repair model checkpoint가 없고, full SCU run에는 raw image array artifact가 없는지 확인한다.

- [x] **Step 3: 문서 tree를 확인하고 계획 checkbox를 갱신한다.**

Run: `find . -maxdepth 3 -type f | sort`

Expected: 두 `08-18` experiment 문서, support analysis source/test/config, 결과 artifact가 존재한다.

## 자체 검토

- Support 정의·구간 기준·seed 3 사전 선택은 Task 1, 4, 5에 각각 구현과 검증을 둔다.
- Original/retrain-only 실행은 Task 2와 4에서 target-only forgetting 없이 독립적으로 수행한다.
- Full SCU는 Task 5에서 centralized comparator로만 실행하며, 현 시점에 없는 federated distributed repair는 결과 문서에서 미구현으로 명시한다.
- Support를 model metric으로 사후 선택하지 않도록 split screening과 full SCU seed selection을 model 학습 전 단계로 분리했다.
