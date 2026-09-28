# 균등 데이터량 Non-IID CIFAR-10 연합 삭제 실험 구현 계획

> **에이전트 작업자용:** 이 계획은 작업별로 실행하고, 각 작업에서 먼저 실패하는 테스트를 작성한다.

**목표:** CIFAR-10 전체 학습 데이터를 10개 클라이언트에 각각 5,000개씩 배정하되 클래스 분포만 Dirichlet `alpha=0.1`로 다르게 만들고, client_0 전체 삭제에 대한 재구성 codec 연합학습·재학습 기준·forgetting·repair 비교를 실행한다.

**구조:** `data.py`는 정확한 행·열 합을 갖는 client-class 할당 행렬과 manifest를 만든다. 재구성 전용 연합학습은 codec의 encoder와 decoder를 함께 FedAvg로 학습하고, 삭제 단계는 target client의 joint forgetting, 남은 9개 client의 로컬 repair, 서버의 naive 또는 conflict-aware aggregation으로 분리한다. 서버와 client 사이에는 model state 또는 update와 scalar metric만 전송하며 retained raw data는 서버로 보내지 않는다.

**기술:** Python, PyTorch, torchvision CIFAR-10, mpi4py, NumPy, pytest, CUDA/AWGN 10 dB.

## 전역 제약

- 데이터는 CIFAR-10 train 50,000개 전체를 사용한다.
- client 수는 10명이고 각 client의 train sample 수는 정확히 5,000개다.
- class 수는 10개이며 전체 class별 sample 수는 원본 CIFAR-10과 동일하게 정확히 5,000개다.
- label skew는 `Dirichlet(alpha=0.1)`에서 유도한다. 데이터 수 불균형은 만들지 않는다.
- target은 `client_0`의 전체 데이터이며, unlearned model의 주 기준은 client_0 제외 재학습 model과의 functional gap이다.
- encoder와 decoder는 forgetting 및 repair에서 항상 함께 갱신한다.
- retained raw data는 server process에 보내지 않는다. local repair는 retained client의 local data만 사용한다.
- 첫 채널 조건은 AWGN 10 dB이고, model은 `CIFAR10SemanticCodec`을 사용한다.
- Markdown 문서는 한국어로 작성한다.
- repository root는 Git repository가 아니므로 이 계획의 commit 단계는 수행하지 않는다.

---

## 파일 구조

- 수정: `src/data/data.py` — 균등 client 크기 Dirichlet split, test split, split manifest helper
- 생성: `src/training/federated_reconstruction.py` — local VAE reconstruction update 및 sample-count FedAvg
- 생성: `src/unlearning/federated_scu.py` — target joint forgetting, local repair, conflict-aware update filtering
- 생성: `src/evaluation/federated_behavior.py` — public-test functional gap, MSE, PSNR, class/client summary
- 생성: `src/experiments/federated_deletion.py` — MPI process가 세 학습 단계를 순서대로 조율하고 artifact를 저장
- 수정: `src/utils/option.py` — `federated_deletion` 실험과 명시적 hyperparameter option
- 수정: `src/main.py` — `--experiment federated_deletion`을 짧게 dispatch
- 생성: `src/configs/federated_cifar10_dirichlet_01.json` — 실행 가능한 기본 config
- 생성: `src/tests/test_equal_size_dirichlet_split.py`
- 생성: `src/tests/test_federated_reconstruction.py`
- 생성: `src/tests/test_federated_scu.py`
- 생성: `src/tests/test_federated_deletion_smoke.py`

## 실험 단계

```text
같은 initial checkpoint
  ├─ 10명 FedAvg reconstruction training -> M^G
  ├─ client_0 없이 9명 FedAvg reconstruction training -> M^{-0}
  └─ M^G
       -> client_0의 joint forgetting -> M^F
       -> 9명의 local repair + naive FedAvg -> M^U_naive
       -> 9명의 local repair + conflict-aware aggregation -> M^U_conflict
```

첫 정식 실행 설정은 `200 global rounds`, `local_epochs=1`, `batch_size=128`, `lr=1e-4`, `latent_dim=64`, `AWGN=10 dB`다. 각 client는 한 round에 자신의 5,000개 전체를 한 번 보므로 original training은 총 1,000만 sample exposure를 갖는다. target을 제외한 재학습은 900만 sample exposure를 갖는 자연스러운 client-deletion 기준이다.

### Task 1: 정확한 균등 크기 Dirichlet 분할

**Files:**

- Modify: `src/data/data.py`
- Create: `src/tests/test_equal_size_dirichlet_split.py`

**Interfaces:**

- Produces: `partition_equal_size_dirichlet(labels, n_clients, alpha, seed) -> tuple[list[np.ndarray], np.ndarray]`
- Produces: `build_federated_split_manifest(train_indices, train_labels, test_indices, test_labels, alpha, seed) -> dict`

- [ ] **Step 1: 실패하는 test를 작성한다.**

```python
def test_equal_size_dirichlet_uses_every_example_once():
    labels = np.repeat(np.arange(10), 50)
    client_indices, allocation = partition_equal_size_dirichlet(
        labels, n_clients=10, alpha=0.1, seed=7
    )

    assert [len(indices) for indices in client_indices] == [50] * 10
    assert len(np.concatenate(client_indices)) == len(labels)
    assert len(np.unique(np.concatenate(client_indices))) == len(labels)
    assert allocation.sum(axis=1).tolist() == [50] * 10
    assert allocation.sum(axis=0).tolist() == [50] * 10
```

- [ ] **Step 2: 실패를 확인한다.**

Run: `pytest -q src/tests/test_equal_size_dirichlet_split.py::test_equal_size_dirichlet_uses_every_example_once`

Expected: `ImportError` 또는 `partition_equal_size_dirichlet` 미정의 실패.

- [ ] **Step 3: 최소 구현을 작성한다.**

`data.py`에 다음 순서의 helper를 둔다.

```python
def partition_equal_size_dirichlet(labels, n_clients, alpha, seed):
    # 1) gamma(alpha) weights를 만든다.
    # 2) 반복 행/열 scaling으로 client total과 class total을 만족하는 연속 quota를 만든다.
    # 3) floor quota의 남은 행/열 수요를 fractional quota가 큰 cell부터
    #    complete bipartite residual matching으로 정수 배정한다.
    # 4) class별 shuffled index를 allocation 수만큼 잘라 client에 준다.
    return client_indices, allocation
```

행 합은 `len(labels) // n_clients`, 열 합은 각 class의 실제 개수여야 하며, 나눠떨어지지 않는 input은 `ValueError`로 거부한다. `alpha <= 0`도 거부한다. allocation은 `(n_clients, n_classes)` 정수 배열이다.

- [ ] **Step 4: test를 통과시킨다.**

Run: `pytest -q src/tests/test_equal_size_dirichlet_split.py`

Expected: PASS. 같은 seed의 동일 allocation과 다른 seed의 다른 allocation도 추가 test한다.

### Task 2: 재구성 codec의 local update와 weighted FedAvg

**Files:**

- Create: `src/training/federated_reconstruction.py`
- Create: `src/tests/test_federated_reconstruction.py`

**Interfaces:**

- Consumes: `CIFAR10SemanticCodec`, `(images, labels)` client data, AWGN SNR
- Produces: `train_local_reconstruction(model, client_data, config, seed) -> tuple[dict[str, Tensor], dict[str, float]]`
- Produces: `average_model_updates(updates, sample_counts) -> dict[str, Tensor]`

- [ ] **Step 1: 실패하는 test를 작성한다.**

```python
def test_weighted_average_model_updates_respects_sample_counts():
    updates = [{"weight": torch.tensor([1.0])}, {"weight": torch.tensor([5.0])}]
    result = average_model_updates(updates, sample_counts=[1, 3])
    assert torch.equal(result["weight"], torch.tensor([4.0]))


def test_local_reconstruction_update_changes_codec_weights():
    model = CIFAR10SemanticCodec(CIFAR10CodecConfig(latent_dim=4, base_channels=4))
    images = torch.rand(8, 3, 32, 32)
    before = {name: value.clone() for name, value in model.state_dict().items()}
    state, metrics = train_local_reconstruction(model, (images, torch.zeros(8)), config, seed=1)
    assert any(not torch.equal(before[name], state[name]) for name in before)
    assert metrics["reconstruction_loss"] >= 0.0
```

- [ ] **Step 2: 실패를 확인한다.**

Run: `pytest -q src/tests/test_federated_reconstruction.py`

Expected: `ModuleNotFoundError: src.training.federated_reconstruction`.

- [ ] **Step 3: 최소 구현을 작성한다.**

`train_local_reconstruction`은 local model copy에 대해 `reconstruct_training_sample()`의 `MSE + beta * KL(q||N(0,I))`를 `local_epochs` 동안 최적화하고 CPU state dict 및 scalar metrics만 반환한다. labels는 reconstruction loss에 쓰지 않는다. `average_model_updates`는 각 tensor를 `sample_count / total_count`로 평균하고 original global state가 아닌 **full model state**를 반환한다.

- [ ] **Step 4: test를 통과시킨다.**

Run: `pytest -q src/tests/test_federated_reconstruction.py`

Expected: PASS.

### Task 3: target forgetting, retained local repair, conflict-aware filtering

**Files:**

- Create: `src/unlearning/federated_scu.py`
- Create: `src/tests/test_federated_scu.py`

**Interfaces:**

- Consumes: original state `M^G`, client_0 local data, retained client local data
- Produces: `forget_target_client(model, target_data, config, seed) -> tuple[dict[str, Tensor], dict[str, float]]`
- Produces: `repair_retained_client(model, retained_data, original_model, config, seed) -> tuple[dict[str, Tensor], dict[str, float]]`
- Produces: `filter_conflicting_repair_update(forgetting_update, repair_update) -> dict[str, Tensor]`

- [ ] **Step 1: 실패하는 test를 작성한다.**

```python
def test_conflict_filter_removes_component_that_undoes_forgetting():
    forgetting = {"weight": torch.tensor([3.0, 0.0])}
    repair = {"weight": torch.tensor([-2.0, 4.0])}
    filtered = filter_conflicting_repair_update(forgetting, repair)
    assert torch.allclose(filtered["weight"], torch.tensor([0.0, 4.0]))
```

- [ ] **Step 2: 실패를 확인한다.**

Run: `pytest -q src/tests/test_federated_scu.py::test_conflict_filter_removes_component_that_undoes_forgetting`

Expected: `ModuleNotFoundError: src.unlearning.federated_scu`.

- [ ] **Step 3: 최소 구현을 작성한다.**

`forget_target_client`은 target data에서 encoder와 decoder를 joint update한다. 첫 ablation은 `lambda_kl * KL(q_F || q_G) - lambda_rec * MSE(x_hat_F, x)`를 최소화한다. `repair_retained_client`은 local retained data에서 reconstruction loss와 선택적 original-model distillation만 사용하며, target data를 받거나 참조하지 않는다.

Conflict filter는 flatten된 전체 parameter에서 다음을 계산한다.

```python
dot = dot(repair_update, forgetting_update)
if dot < 0:
    repair_update = repair_update - dot / squared_norm(forgetting_update) * forgetting_update
```

`squared_norm == 0`이면 repair update를 그대로 둔다.

- [ ] **Step 4: test를 통과시킨다.**

Run: `pytest -q src/tests/test_federated_scu.py`

Expected: PASS. target forgetting이 model state를 바꾸고, repair의 함수가 target data를 argument로 받지 않는 것도 test한다.

### Task 4: MPI 기반 end-to-end 삭제 실험과 artifact

**Files:**

- Create: `src/experiments/federated_deletion.py`
- Create: `src/evaluation/federated_behavior.py`
- Modify: `src/main.py`
- Modify: `src/utils/option.py`
- Create: `src/configs/federated_cifar10_dirichlet_01.json`
- Create: `src/tests/test_federated_deletion_smoke.py`

**Interfaces:**

- Produces: `run_federated_deletion_experiment(comm, config, output_root) -> Path`
- Consumes: MPI world size `11` (`server + 10 clients`), split manifest, codec config
- Produces: `metrics/summary.json`, `metrics/per_class.csv`, `metrics/per_client.csv`, checkpoints, config, manifest, run log

- [ ] **Step 1: 실패하는 smoke test를 작성한다.**

```python
def test_smoke_config_has_equal_size_dirichlet_and_client_zero_target():
    config = load_json_config(FEDERATED_CONFIG_PATH)
    assert config["federation"]["n_clients"] == 10
    assert config["federation"]["samples_per_client"] == 5000
    assert config["split"]["alpha"] == 0.1
    assert config["unlearning"]["target_client_id"] == 0
```

- [ ] **Step 2: 실패를 확인한다.**

Run: `pytest -q src/tests/test_federated_deletion_smoke.py`

Expected: config path 또는 loader가 없어 실패.

- [ ] **Step 3: 최소 구현을 작성한다.**

`main.py`에는 다음처럼 짧은 dispatch만 추가한다.

```python
if args.experiment == "federated_deletion":
    run_federated_deletion(args)
    return
```

실험 process는 다음 규칙을 지킨다.

1. rank 0은 global state만 보유하고, rank 1--10은 각자 한 client data만 보유한다.
2. original과 client_0 제외 재학습은 같은 `initial.pt`에서 시작한다.
3. retraining 동안 rank 1(client_0)은 model update를 보내지 않고 idle protocol에 참여한다.
4. forgetting 후 client_0은 scalar metric을 보낸 뒤 raw data reference를 버리고 repair phase에 참여하지 않는다.
5. retained client update는 naive와 filtered mode에서 별도 clone으로 만들고, server는 state/update만 집계한다.
6. public CIFAR-10 test input에 대해 MSE, PSNR, output functional gap, posterior-mean gap, symmetric posterior KL을 계산한다.

Artifact는 `outputs/federated_deletion/<UTC timestamp>_seed-<seed>/`에 저장한다.

```text
config.json
split_manifest.json
environment.json
checkpoints/initial.pt
checkpoints/original.pt
checkpoints/retrain_without_client_0.pt
checkpoints/joint_forgetting.pt
checkpoints/naive_repair.pt
checkpoints/conflict_aware_repair.pt
metrics/summary.json
metrics/per_class.csv
metrics/per_client.csv
run.log
```

- [ ] **Step 4: test를 통과시킨다.**

Run: `pytest -q src/tests/test_federated_deletion_smoke.py`

Expected: PASS.

### Task 5: CPU smoke와 CUDA pilot 검증

**Files:**

- Modify: `src/configs/federated_cifar10_dirichlet_01.json`
- Modify: `docs/experiment/` 아래의 새 한국어 실행 안내 문서

- [ ] **Step 1: CPU smoke config를 추가한다.**

Synthetic CIFAR-10 10 classes × 20 examples, 2 clients × 100 examples, 2 rounds, local epoch 1, latent dimension 4로 줄인 smoke config를 만든다. production config의 10 clients / 5,000 samples 규칙과 섞지 않는다.

- [ ] **Step 2: MPI smoke를 실행한다.**

Run: `mpirun --allow-run-as-root -np 3 python src/main.py --experiment federated_deletion --config src/configs/federated_cifar10_smoke.json`

Expected: 모든 rank가 종료하고 artifact에 모든 checkpoint와 metric file이 있다.

- [ ] **Step 3: 정식 CUDA pilot을 실행한다.**

Run: `mpirun --allow-run-as-root -np 11 python src/main.py --experiment federated_deletion --config src/configs/federated_cifar10_dirichlet_01.json --seed 0`

Expected: split manifest의 client sizes는 모두 5,000이고, `client_0` 삭제 재학습과 두 repair model의 functional gap이 기록된다.

- [ ] **Step 4: 전체 test와 compile check를 실행한다.**

Run: `pytest -q && python -m py_compile src/main.py src/data/data.py src/training/federated_reconstruction.py src/unlearning/federated_scu.py src/experiments/federated_deletion.py`

Expected: PASS.

## 자체 검토

- 균등 data count와 Dirichlet label skew는 Task 1에서 행·열 합 assertion으로 검증한다.
- codec reconstruction training과 joint encoder-decoder update는 Task 2와 3에서 검증한다.
- retained raw data 비중앙화는 Task 3과 4의 function boundary와 MPI protocol에 반영했다.
- 재학습 기준, naive repair, conflict-aware repair, class/client별 metric은 Task 4에서 모두 기록한다.
- 정식 실행 전에 CPU smoke와 MPI protocol smoke를 수행하므로 GPU 장시간 실행 전에 schema와 deadlock을 검증한다.
