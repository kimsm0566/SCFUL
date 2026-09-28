# `src` 실행 구조 리팩터링 구현 계획

> **에이전트 작업자용:** 이 계획은 task별로 `superpowers:executing-plans`를 사용해 순서대로 실행한다. 모든 단계는 체크박스로 추적한다.

**목표:** 외부 baseline clone을 `third_party/`로 분리하고, `python src/main.py` 한 명령으로 Oracle Behavior 실험 전체를 재현 가능하게 실행한다.

**아키텍처:** CA-SSFL의 단순한 `data/models/utils + runner` 배치를 참고하되, 이 프로젝트의 training과 unlearning 경계를 유지한다. `main.py`는 CLI와 workflow 호출만 담당하고, data/model/training/unlearning/evaluation/utils 모듈이 각각 한 책임만 갖는다.

**기술 스택:** Python 3.10, PyTorch 2.4.1+cu121, torchvision 0.19.1, NumPy, Matplotlib, `unittest`, Docker Compose GPU container.

## 전역 제약

- 새 Python 코드는 모두 `src/` 아래에 둔다.
- 새 Markdown 문서는 한국어로 작성한다.
- codec은 `encoder -> power normalization -> AWGN channel -> decoder`의 reconstruction model로 유지한다.
- unlearning 성공은 target MSE가 아니라 retrain-without-target oracle과의 functional gap으로 평가한다.
- SCU-style comparator는 centralized oracle이며, 다음 단계의 distributed repair는 retained raw data를 서버에 중앙화하지 않는다.
- 기존 `outputs/oracle_behavior/`와 `outputs/scu/` run은 수정하거나 덮어쓰지 않는다.
- `third_party/SCU`와 `third_party/FedOSD`의 clone 내부 파일, `.git`, 현재 working-tree 변경은 이동 외에 수정하지 않는다.
- top-level workspace는 Git repository가 아니므로 이 계획의 각 task는 commit 대신 검증 명령과 출력 기록으로 checkpoint를 남긴다.
- 테스트는 먼저 실패하는 것을 확인한 뒤 최소 변경으로 통과시킨다. Python 변경 후에는 관련 test와 compile check를 실행한다.

---

## 목표 파일 지도

| 경로 | 책임 |
| --- | --- |
| `third_party/SCU/`, `third_party/FedOSD/` | 외부 baseline clone 보관. 새 연구 code는 import하지 않는다. |
| `src/main.py` | `--config`, `--seed`, `--smoke`, `--output-root` parsing과 `run_oracle_behavior` 호출. |
| `src/configs/oracle_mnist.json` | 기본 MNIST Oracle Behavior 설정. dataset root는 `third_party/SCU/data/mnist`. |
| `src/data/mnist.py` | real/synthetic MNIST loading과 labels/metadata bundle. |
| `src/data/client_split.py` | controlled Non-IID split 및 manifest. |
| `src/data/probes.py` | audit probe reference, sampling, probe dataset. |
| `src/models/semantic_codec.py` | posterior, codec config, VAE codec, power-normalized AWGN path. |
| `src/training/reconstruction.py` | Original/Retrain 공통 reconstruction training. |
| `src/unlearning/scu_style.py` | original teacher 기반 forget-only/self-sharing SCU adaptation. |
| `src/evaluation/oracle_behavior.py` | 세 모델 학습 흐름 조립, probe functional-gap evaluation, artifact summary. |
| `src/evaluation/posterior_audit.py` | per-client diagonal Gaussian sufficiency diagnostics. |
| `src/utils/config.py` | config load, smoke override, device 선택. |
| `src/utils/artifacts.py` | non-overwriting run directory와 JSON/CSV/checkpoint/prediction 저장. |
| `src/utils/metrics.py` | MSE, symmetric KL, bootstrap summary. |
| `src/utils/plotting.py` | reconstruction grid, latent comparison, client Gaussian figure. |
| `src/utils/reproducibility.py` | seed derivation, deterministic CUDA, environment metadata. |

## 작업 1: 외부 clone을 `third_party`로 분리하고 legacy 실행 경로를 보존한다

**파일:**

- Create: `third_party/README.md`
- Create: `src/tests/test_workspace_layout.py`
- Move: `src/SCU/` → `third_party/SCU/`
- Move: `src/FedOSD/` → `third_party/FedOSD/`
- Modify: `compose.yaml:20-23`

**인터페이스:**

- 소비: 현재 nested clone의 `.git` 디렉터리와 `compose.yaml`의 `PYTHONPATH`.
- 생산: `third_party/SCU/.git`, `third_party/FedOSD/.git`, 그리고 `/workspace/third_party/FedOSD`를 가리키는 Docker `PYTHONPATH`.

- [ ] **Step 1: clone 이동 전 안전 조건과 실패하는 layout test를 작성한다.**

```python
# src/tests/test_workspace_layout.py
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


class WorkspaceLayoutTests(unittest.TestCase):
    def test_external_baseline_clones_are_outside_research_src_package(self):
        self.assertTrue((ROOT / "third_party/SCU/.git").is_dir())
        self.assertTrue((ROOT / "third_party/FedOSD/.git").is_dir())
        self.assertFalse((ROOT / "src/SCU").exists())
        self.assertFalse((ROOT / "src/FedOSD").exists())


if __name__ == "__main__":
    unittest.main()
```

먼저 `test -d src/SCU/.git`, `test -d src/FedOSD/.git`, `test ! -e third_party/SCU`, `test ! -e third_party/FedOSD`를 실행해 source와 target을 확정한다.

- [ ] **Step 2: 실패를 확인한다.**

Run: `python -m unittest src.tests.test_workspace_layout -v`  
Expected: `third_party/SCU/.git` assertion failure.

- [ ] **Step 3: clone을 디렉터리 단위로 이동하고 Docker 경로를 갱신한다.**

`third_party/`를 만든 뒤 `mv src/SCU third_party/SCU`와 `mv src/FedOSD third_party/FedOSD`를 한 번씩 실행한다. 이동 뒤 `git -C third_party/SCU status --short`와 `git -C third_party/FedOSD status --short`로 nested clone history와 SCU의 기존 수정 상태가 유지됐는지 기록한다.

`compose.yaml`의 다음 값을 변경한다.

```yaml
PYTHONPATH: /workspace/third_party/FedOSD
```

`third_party/README.md`에는 두 clone이 external baseline이며, 새 code가 이 경로를 import하지 않고, legacy 실행은 각 clone 내부 README/entry point를 사용한다는 한국어 안내를 작성한다.

- [ ] **Step 4: layout test와 compose 경로를 검증한다.**

Run:

```bash
python -m unittest src.tests.test_workspace_layout -v
rg -n 'PYTHONPATH: /workspace/third_party/FedOSD' compose.yaml
git -C third_party/SCU status --short
git -C third_party/FedOSD status --short
```

Expected: layout test PASS, compose path 한 건, SCU의 기존 modified/untracked 상태가 이동 전과 동일.

## 작업 2: data 책임을 MNIST, split, probe 모듈로 분리한다

**파일:**

- Create: `src/data/__init__.py`
- Create: `src/data/mnist.py`
- Move and split: `src/utils/client_splits.py` → `src/data/client_split.py`, `src/data/probes.py`
- Create: `src/tests/test_mnist_data.py`
- Modify: `src/tests/test_client_splits.py`

**인터페이스:**

- `DatasetBundle(train: Dataset[Any], test: Dataset[Any], train_labels: Tensor, test_labels: Tensor, metadata: dict[str, Any])`
- `load_mnist_datasets(dataset_config: Mapping[str, Any], *, workspace_root: Path, seed: int) -> DatasetBundle`
- `SyntheticMNIST(per_digit: int, seed: int)`
- `build_controlled_noniid_split(labels, split_seed, shared_digits, shared_per_digit, target_specific_digit, retained_clients) -> ControlledClientSplit`
- `select_audit_indices(split, train_labels, test_labels, probe_seed, target_per_group, retained_total) -> dict[str, list[ProbeRef]]`
- `ProbeDataset(source: Dataset[Any], refs: list[ProbeRef])`

- [ ] **Step 1: 새 data import가 실패하는 test를 작성한다.**

```python
# src/tests/test_mnist_data.py
import unittest
from pathlib import Path

from src.data.mnist import SyntheticMNIST, load_mnist_datasets


class MNISTDataTests(unittest.TestCase):
    def test_synthetic_dataset_is_deterministic_and_has_mnist_shape(self):
        first = SyntheticMNIST(per_digit=3, seed=7)
        second = SyntheticMNIST(per_digit=3, seed=7)
        self.assertEqual(len(first), 30)
        self.assertEqual(tuple(first[0][0].shape), (1, 28, 28))
        self.assertEqual(first[0][1].item(), second[0][1].item())
        self.assertTrue((first.images == second.images).all())

    def test_synthetic_dataset_bundle_exposes_labels_and_metadata(self):
        bundle = load_mnist_datasets(
            {"name": "synthetic_mnist", "synthetic_train_per_digit": 4,
             "synthetic_test_per_digit": 3},
            workspace_root=Path.cwd(),
            seed=5,
        )
        self.assertEqual(len(bundle.train_labels), 40)
        self.assertEqual(bundle.metadata["name"], "synthetic_mnist")


if __name__ == "__main__":
    unittest.main()
```

`test_client_splits.py`와 smoke test의 split/probe import도 `src.data.client_split` 및 `src.data.probes`로 바꾼다.

- [ ] **Step 2: 실패를 확인한다.**

Run: `python -m unittest src.tests.test_mnist_data -v`  
Expected: `ModuleNotFoundError: No module named 'src.data'`.

- [ ] **Step 3: data 모듈을 구현하고 기존 split 의미를 보존한다.**

`src/data/mnist.py`에 synthetic image 생성과 torchvision MNIST loading을 옮긴다. real MNIST path는 relative path일 때 `workspace_root / data_root`를 사용하고, dataset name이 `mnist` 또는 `synthetic_mnist`가 아니면 `ValueError`를 발생시킨다.

`ControlledClientSplit`와 `build_controlled_noniid_split`은 `src/data/client_split.py`로 옮긴다. `ProbeRef`, `_take_random`, `select_audit_indices`, `ProbeDataset`은 `src/data/probes.py`에 둔다. `src/data/probes.py`는 `ControlledClientSplit`만 import하며 model, optimizer, artifact writer를 import하지 않는다.

- [ ] **Step 4: data unit tests를 통과시킨다.**

Run:

```bash
python -m unittest src.tests.test_mnist_data src.tests.test_client_splits -v
python -m compileall -q src/data
```

Expected: 모든 test PASS, compile output 없음.

## 작업 3: codec의 이름과 import를 semantic communication 역할에 맞게 정리한다

**파일:**

- Move: `src/models/mnist_codec.py` → `src/models/semantic_codec.py`
- Modify: `src/models/__init__.py`
- Modify: `src/training/reconstruction.py`
- Modify: `src/unlearning/scu_style.py`
- Modify: `src/utils/client_posterior.py`
- Modify: `src/utils/oracle_metrics.py`
- Modify: `src/experiments/oracle_behavior.py`
- Move: `src/tests/test_mnist_codec.py` → `src/tests/test_semantic_codec.py`
- Modify: `src/tests/test_oracle_metrics.py`, `src/tests/test_scu_style.py`

**인터페이스:**

- `from src.models.semantic_codec import CodecConfig, Posterior, CodecOutput, MNISTSemanticCodec`
- `MNISTSemanticCodec.reconstruct_training_sample(inputs, snr_db, posterior_generator, channel_generator) -> CodecOutput`
- `MNISTSemanticCodec.decode_posterior_mean(inputs) -> CodecOutput`
- `MNISTSemanticCodec.decode_with_awgn(inputs, noise) -> CodecOutput`

- [ ] **Step 1: 새 codec module import를 요구하도록 test를 바꾼다.**

```python
# src/tests/test_semantic_codec.py
from src.models.semantic_codec import CodecConfig, MNISTSemanticCodec


def test_power_normalization_and_fixed_awgn_noise_are_explicit(self):
    model = MNISTSemanticCodec(
        CodecConfig(latent_dim=4, encoder_hidden_dim=8, decoder_hidden_dim=8)
    )
    normalized = model.normalized_latent(torch.tensor([[3.0, 4.0, 0.0, 0.0]]))
    self.assertAlmostEqual(normalized.norm(dim=1).item(), 2.0, places=6)
```

`test_oracle_metrics.py`, `test_scu_style.py`도 model import를 `semantic_codec`으로 변경한다.

- [ ] **Step 2: 실패를 확인한다.**

Run: `python -m unittest src.tests.test_semantic_codec -v`  
Expected: `ModuleNotFoundError: No module named 'src.models.semantic_codec'`.

- [ ] **Step 3: model file을 이동하고 모든 production import를 갱신한다.**

`mnist_codec.py`의 class와 method body는 바꾸지 않고 `semantic_codec.py`로 이동한다. `rg -n 'models\.mnist_codec' src --glob '!tests/**'`의 모든 결과를 `src.models.semantic_codec`으로 바꾼다. `src/models/__init__.py`는 공개 class를 re-export하지 않고 package marker로만 둔다.

- [ ] **Step 4: codec, trainer, SCU가 새 import로 동작하는지 검증한다.**

Run:

```bash
python -m unittest src.tests.test_semantic_codec src.tests.test_oracle_metrics src.tests.test_scu_style -v
rg -n 'models\.mnist_codec' src || true
python -m compileall -q src/models src/training src/unlearning
```

Expected: 모든 test PASS, obsolete import 결과 없음, compile output 없음.

## 작업 4: config·artifact·metric helper를 `utils`의 안정된 이름으로 만든다

**파일:**

- Create: `src/utils/config.py`
- Move: `src/utils/oracle_artifacts.py` → `src/utils/artifacts.py`
- Move: `src/utils/oracle_metrics.py` → `src/utils/metrics.py`
- Move: `src/experiments/configs/oracle_mnist.json` → `src/configs/oracle_mnist.json`
- Create: `src/configs/__init__.py`
- Modify: `src/experiments/oracle_behavior.py`
- Modify: `src/tests/test_oracle_metrics.py`
- Create: `src/tests/test_config.py`

**인터페이스:**

```text
DEFAULT_ORACLE_CONFIG_PATH: Path
WORKSPACE_ROOT: Path
load_json_config(path: Path) -> dict[str, Any]
build_effective_config(config: Mapping[str, Any], *, smoke_test: bool) -> dict[str, Any]
select_device(config: Mapping[str, Any]) -> torch.device
```

`RunArtifacts`, `per_sample_mse`, `symmetric_diagonal_kl`, `summarize_records`는 이름과 계산식을 유지하고 import path만 `src.utils.artifacts`, `src.utils.metrics`로 바뀐다.

- [ ] **Step 1: config와 새 utility import의 failing tests를 작성한다.**

```python
# src/tests/test_config.py
import unittest

from src.utils.config import (
    DEFAULT_ORACLE_CONFIG_PATH,
    build_effective_config,
    load_json_config,
)


class ConfigTests(unittest.TestCase):
    def test_default_config_uses_moved_scu_dataset_path_without_mutating_source(self):
        source = load_json_config(DEFAULT_ORACLE_CONFIG_PATH)
        smoke = build_effective_config(source, smoke_test=True)
        self.assertEqual(source["dataset"]["data_root"], "third_party/SCU/data/mnist")
        self.assertEqual(smoke["dataset"]["name"], "synthetic_mnist")
        self.assertEqual(source["training"]["epochs"], 20)


if __name__ == "__main__":
    unittest.main()
```

`test_oracle_metrics.py`의 imports를 `src.utils.artifacts`와 `src.utils.metrics`로 변경한다.

- [ ] **Step 2: 실패를 확인한다.**

Run: `python -m unittest src.tests.test_config -v`  
Expected: `ModuleNotFoundError: No module named 'src.utils.config'`.

- [ ] **Step 3: config and utility modules를 이동·구현한다.**

`build_effective_config`는 deep copy 후 `smoke_test=True`일 때 기존 `_smoke_config`의 정확한 값을 적용한다. smoke config는 `device="cpu"`, latent dim 4, hidden width 16, `shared_per_digit=4`, `target_per_group=4`, `retained_total=18`, training/SCU epoch 1, batch 16, AWGN repetitions 2를 유지한다.

기본 JSON의 `dataset.data_root`는 `third_party/SCU/data/mnist`로 변경한다. config read error는 경로를 포함한 `FileNotFoundError`, invalid JSON은 `ValueError`로 변환한다. `select_device`는 기존 `auto`, explicit CPU, unavailable CUDA 오류 semantics를 유지한다.

`RunArtifacts`와 metrics helper를 새 파일명으로 이동하고, 현재 experiment module의 import를 새 path로 바꾼다. 이동 뒤 old file은 남기지 않는다.

- [ ] **Step 4: config와 storage/metric 계약을 검증한다.**

Run:

```bash
python -m unittest src.tests.test_config src.tests.test_oracle_metrics src.tests.test_reproducibility -v
rg -n 'oracle_artifacts|oracle_metrics|experiments/configs' src || true
python -m compileall -q src/utils
```

Expected: 모든 test PASS, obsolete module path 결과 없음, compile output 없음.

## 작업 5: posterior audit과 plotting을 evaluation/util 경계로 이동한다

**파일:**

- Create: `src/evaluation/__init__.py`
- Move: `src/utils/client_posterior.py` → `src/evaluation/posterior_audit.py`
- Create: `src/utils/plotting.py`
- Modify: `src/experiments/oracle_behavior.py`
- Modify: `src/tests/test_client_posterior.py`
- Create: `src/tests/test_plotting.py`

**인터페이스:**

```python
from src.evaluation.posterior_audit import (
    ClientPosteriorAuditConfig,
    ClientGaussianSummary,
    audit_client_gaussian_sufficiency,
    client_posterior_summary_payload,
)

save_oracle_figures(
    predictions: dict[str, Any],
    client_summaries: list[ClientGaussianSummary],
    output_dir: Path,
    probes_per_group: int,
) -> None
```

- [ ] **Step 1: new module imports와 figure outputs를 요구하는 test를 작성한다.**

```python
# src/tests/test_plotting.py
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.utils.plotting import save_oracle_figures


class PlottingTests(unittest.TestCase):
    def test_save_oracle_figures_writes_all_three_png_files(self):
        predictions, summaries = make_minimal_oracle_prediction_fixture()
        with TemporaryDirectory() as directory:
            save_oracle_figures(predictions, summaries, Path(directory), probes_per_group=1)
            self.assertTrue((Path(directory) / "reconstruction_grid.png").is_file())
            self.assertTrue((Path(directory) / "latent_comparison.png").is_file())
            self.assertTrue((Path(directory) / "client_gaussian_diagnostic.png").is_file())
```

`make_minimal_oracle_prediction_fixture`는 test file 안에 `target_train_member` partition, `shared_high`/`target_specific` reference 각 하나, 1×28×28 input, 그리고 `original`, `retrain`, `scu_forget_only`, `scu_self_sharing`의 reconstruction/latent tensor를 만든다. `test_client_posterior.py` import는 `src.evaluation.posterior_audit`로 변경한다.

- [ ] **Step 2: 실패를 확인한다.**

Run:

```bash
python -m unittest src.tests.test_client_posterior src.tests.test_plotting -v
```

Expected: `src.evaluation.posterior_audit` 또는 `src.utils.plotting` import failure.

- [ ] **Step 3: posterior audit을 이동하고 figure rendering을 추출한다.**

`client_posterior.py`의 moment matching, MMD, distribution-shape, label-mixture diagnostic code를 `evaluation/posterior_audit.py`로 이동한다. only model dependency is `src.models.semantic_codec`; artifact I/O를 추가하지 않는다.

기존 `_plot_reconstruction_grid`, `_plot_latent_comparison`, `_plot_client_gaussian_diagnostic`의 bodies를 `utils/plotting.py`의 private helpers로 옮기고, public `save_oracle_figures`가 세 PNG path를 `output_dir` 아래에 고정해 전달하게 한다. `MPLBACKEND=Agg`를 전제로 figure를 저장하고 `plt.close(figure)`를 각 helper 끝에 호출한다.

- [ ] **Step 4: audit math과 non-interactive figure 생성을 검증한다.**

Run:

```bash
MPLBACKEND=Agg python -m unittest src.tests.test_client_posterior src.tests.test_plotting -v
rg -n 'client_posterior|matplotlib\.pyplot' src --glob '!tests/**' || true
python -m compileall -q src/evaluation src/utils
```

Expected: test PASS. `client_posterior` old module import와 `matplotlib.pyplot` import는 `src/utils/plotting.py` 외에는 없음.

## 작업 6: Oracle workflow를 얇은 evaluation coordinator로 재구성한다

**파일:**

- Move and rewrite: `src/experiments/oracle_behavior.py` → `src/evaluation/oracle_behavior.py`
- Delete: `src/experiments/__init__.py`, `src/experiments/`
- Modify: `src/tests/test_oracle_behavior_smoke.py`
- Modify: `src/tests/test_scu_style.py`, `src/tests/test_oracle_metrics.py` as imports require

**인터페이스:**

```python
run_oracle_behavior(
    config: Mapping[str, Any],
    *,
    seed: int,
    output_root: Path,
    smoke_test: bool = False,
) -> Path

evaluate_probe_partition(
    models: Mapping[str, MNISTSemanticCodec],
    dataset: ProbeDataset,
    device: torch.device,
    evaluation_config: Mapping[str, Any],
    noise_seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]
```

- [ ] **Step 1: smoke test가 새 public workflow name을 요구하도록 변경한다.**

```python
from src.evaluation.oracle_behavior import run_oracle_behavior
from src.utils.config import DEFAULT_ORACLE_CONFIG_PATH, load_json_config

run_dir = run_oracle_behavior(
    load_json_config(DEFAULT_ORACLE_CONFIG_PATH),
    seed=3,
    output_root=Path(directory),
    smoke_test=True,
)
```

기존 required artifact list와 `per_sample.csv` column assertion을 그대로 유지한다. 추가로 `run.log`가 마지막 줄 `completed`를 포함하고, summary의 `training`에 `original`, `retrain`, `scu_self_sharing` key가 존재하는 assertion을 넣는다.

- [ ] **Step 2: 실패를 확인한다.**

Run: `MPLBACKEND=Agg python -m unittest src.tests.test_oracle_behavior_smoke -v`  
Expected: `ModuleNotFoundError: No module named 'src.evaluation.oracle_behavior'`.

- [ ] **Step 3: workflow를 data/training/unlearning/evaluation/utils 호출 조립으로 옮긴다.**

`run_oracle_behavior`는 config cloning/smoke override, dataset load, split/probe selection, initial state creation, Original/Retrain/replica training, SCU modes, posterior audit, evaluation, summary/artifact write 순서만 조립한다.

`SyntheticMNIST`, `ProbeDataset`, config parsing, `_select_device`, `_load_datasets`, plotting bodies, argparse `main`은 이 file에 남기지 않는다. `evaluate_probe_partition`은 identical fixed AWGN draws를 모든 model에 적용하고 posterior-mean/AWGN record schema를 기존과 동일하게 생성한다. `retrain` functional gaps는 0으로 강제하는 existing behavior를 유지한다.

`src/experiments` directory는 new workflow/test import가 통과한 뒤 제거한다. output path, checkpoint names, CSV columns, JSON keys, figure file names은 이전 completed run과 동일하게 유지한다.

- [ ] **Step 4: workflow smoke artifact contract를 검증한다.**

Run:

```bash
MPLBACKEND=Agg python -m unittest src.tests.test_oracle_behavior_smoke -v
python -m compileall -q src/evaluation
test ! -e src/experiments
```

Expected: smoke test PASS, compile output 없음, old experiment directory 없음.

## 작업 7: `src/main.py`를 유일한 실행 진입점으로 추가한다

**파일:**

- Create: `src/main.py`
- Create: `src/tests/test_main_cli.py`

**인터페이스:**

```python
parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace
main(argv: Sequence[str] | None = None) -> Path
```

Accepted CLI:

```text
python src/main.py
python src/main.py --config src/configs/oracle_mnist.json --seed 3
python src/main.py --smoke
python src/main.py --output-root outputs/oracle_behavior
python -m src.main --smoke
```

`--smoke-test`은 prior command compatibility를 위해 `--smoke`의 alias로 함께 받는다.

- [ ] **Step 1: CLI parser와 runner delegation의 failing tests를 작성한다.**

```python
# src/tests/test_main_cli.py
import unittest
from pathlib import Path
from unittest.mock import patch

from src.main import main, parse_args


class MainCLITests(unittest.TestCase):
    def test_parse_args_accepts_documented_overrides(self):
        args = parse_args(["--seed", "7", "--smoke", "--output-root", "tmp/run"])
        self.assertEqual(args.seed, 7)
        self.assertTrue(args.smoke)
        self.assertEqual(args.output_root, Path("tmp/run"))

    @patch("src.main.run_oracle_behavior", return_value=Path("outputs/example"))
    def test_main_loads_config_and_delegates_without_embedding_training(self, run_workflow):
        run_dir = main(["--smoke", "--seed", "2"])
        self.assertEqual(run_dir, Path("outputs/example"))
        self.assertEqual(run_workflow.call_args.kwargs["seed"], 2)
        self.assertTrue(run_workflow.call_args.kwargs["smoke_test"])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 실패를 확인한다.**

Run: `python -m unittest src.tests.test_main_cli -v`  
Expected: `ModuleNotFoundError: No module named 'src.main'`.

- [ ] **Step 3: direct-script와 module execution을 지원하는 thin main을 구현한다.**

`src/main.py`의 direct invocation bootstrap은 다음처럼 repository root만 `sys.path`에 추가한다.

```python
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
```

그 뒤 `load_json_config`, `DEFAULT_ORACLE_CONFIG_PATH`, `run_oracle_behavior`를 import한다. parser는 default seed 0, default output root `outputs/oracle_behavior`, default config path `src/configs/oracle_mnist.json`을 제공한다. `main`은 config를 load하고 workflow에 keyword argument로 넘긴 뒤 returned run directory를 print하고 반환한다.

- [ ] **Step 4: unit delegation과 두 invocation form을 검증한다.**

Run:

```bash
python -m unittest src.tests.test_main_cli -v
python src/main.py --help
python -m src.main --help
```

Expected: unit test PASS, 두 help output 모두 `--config`, `--seed`, `--smoke`, `--output-root`를 표시.

## 작업 8: 사용자 문서와 source tree 설명을 실제 구조로 갱신한다

**파일:**

- Modify: `README.md`
- Modify: `src/README.md`
- Modify: `third_party/README.md` if task 1 text needs final paths

**인터페이스:**

- top-level README는 research scope, source map, canonical commands를 안내한다.
- `src/README.md`는 source directory 역할과 `src/main.py` arguments를 안내한다.

- [ ] **Step 1: current structure를 요구하는 documentation assertions를 작성한다.**

```python
# `src/tests/test_workspace_layout.py`에 추가
def test_readmes_document_main_entrypoint_and_third_party_boundary(self):
    root_readme = (ROOT / "README.md").read_text(encoding="utf-8")
    source_readme = (ROOT / "src/README.md").read_text(encoding="utf-8")
    self.assertIn("python src/main.py", root_readme)
    self.assertIn("third_party/", root_readme)
    self.assertIn("src/main.py", source_readme)
    self.assertIn("src/data/", source_readme)
```

- [ ] **Step 2: 실패를 확인한다.**

Run: `python -m unittest src.tests.test_workspace_layout -v`  
Expected: canonical command or `third_party/` text assertion failure.

- [ ] **Step 3: README를 한국어로 갱신한다.**

Top-level README에는 `third_party/`가 external baseline clone임을 명시하고, `src/` tree와 `python src/main.py`, config override, smoke command를 적는다. `src/README.md`에는 각 package의 책임과 “main에는 optimizer, loss, figure 로직을 넣지 않는다”는 유지 규칙을 적는다. README의 연구 목적은 joint encoder-decoder forgetting, retrain equivalence, retained-client distributed repair라는 현재 연구 불변조건과 일치시킨다.

- [ ] **Step 4: 문서와 tree가 실제 상태를 설명하는지 검증한다.**

Run:

```bash
python -m unittest src.tests.test_workspace_layout -v
find . -maxdepth 3 -type f | sort
rg -n 'src/experiments|src/SCU|src/FedOSD' README.md src/README.md || true
```

Expected: test PASS, old source locations을 안내하는 문구 없음.

## 작업 9: 전체 회귀 검증과 CUDA pilot으로 리팩터링 전후 실험 계약을 확인한다

**파일:**

- Create: 새 non-overwriting run under `outputs/oracle_behavior/`
- Modify: 없음

**인터페이스:**

- `python src/main.py --smoke`는 complete artifact schema를 생성한다.
- CUDA seed-0 run은 original, retrain, retrain_replica, SCU forget-only, SCU self-sharing checkpoint를 만든다.

- [ ] **Step 1: 모든 unit test와 package compile을 실행한다.**

Run:

```bash
docker compose exec -T scful-dev bash -lc \
  'cd /workspace && MPLBACKEND=Agg python -m unittest discover -s src/tests -v && python -m compileall -q src'
```

Expected: 모든 test PASS, compile error 없음.

- [ ] **Step 2: canonical direct command의 CPU smoke run을 실행한다.**

Run:

```bash
docker compose exec -T scful-dev bash -lc \
  'cd /workspace && MPLBACKEND=Agg python src/main.py --smoke --output-root outputs/oracle_behavior'
```

Expected: stdout에 새 run directory 하나, 해당 `run.log` 마지막 줄 `completed`.

- [ ] **Step 3: CUDA seed-0 pilot을 canonical command로 실행한다.**

Run:

```bash
RUN_LOG=outputs/oracle_behavior/refactor_cuda_seed0.log
docker compose exec -T scful-dev bash -lc \
  'cd /workspace && MPLBACKEND=Agg python src/main.py --config src/configs/oracle_mnist.json --seed 0 --output-root outputs/oracle_behavior' \
  | tee "$RUN_LOG"
export ORACLE_RUN_DIR="$(tail -n 1 "$RUN_LOG")"
test -d "$ORACLE_RUN_DIR"
```

Expected: CUDA device가 environment metadata에 기록되고 completed run directory가 생성된다.

- [ ] **Step 4: 새 CUDA artifact schema와 numeric invariants를 검사한다.**

새 CUDA command가 출력한 run directory를 환경 변수에 저장한 뒤 다음 검증을 실행한다.

```python
import csv
import json
import math
import os
from pathlib import Path

run_dir = Path(os.environ["ORACLE_RUN_DIR"])
required = [
    "checkpoints/initial.pt", "checkpoints/original.pt", "checkpoints/retrain.pt",
    "checkpoints/retrain_replica.pt", "checkpoints/scu_forget_only.pt",
    "checkpoints/scu_self_sharing.pt", "metrics/per_sample.csv",
    "metrics/summary.json", "predictions/probes.pt",
    "figures/reconstruction_grid.png", "figures/latent_comparison.png",
    "figures/client_gaussian_diagnostic.png", "run.log",
]
assert all((run_dir / item).is_file() for item in required)
assert (run_dir / "run.log").read_text(encoding="utf-8").splitlines()[-1] == "completed"
with (run_dir / "metrics/per_sample.csv").open(encoding="utf-8", newline="") as handle:
    rows = list(csv.DictReader(handle))
assert rows
for row in rows:
    for field in ("mse_input", "fg_output", "fg_mu", "fg_latent"):
        assert math.isfinite(float(row[field]))
    if row["model"] == "retrain":
        assert float(row["fg_output"]) == 0.0
        assert float(row["fg_mu"]) == 0.0
        assert float(row["fg_latent"]) == 0.0
summary = json.loads((run_dir / "metrics/summary.json").read_text(encoding="utf-8"))
assert {"original", "retrain", "scu_self_sharing"} <= set(summary["training"])
```

Expected: all assertions pass. 결과 수치는 reparameterization/optimizer stochasticity의 영향으로 기존 pilot과 exact equality를 요구하지 않으며, config·schema·metric invariants만 동일해야 한다.

## 계획 자기 점검

- 설계 문서의 clone 분리, source tree, direct main execution, config path, existing artifact preservation, test strategy, CUDA run 요구사항을 Task 1–9에 각각 배정했다.
- 모든 새 public interface는 목표 파일 지도 또는 해당 task의 interface block에서 정의했다.
- `third_party` 이동 후 data root와 compose `PYTHONPATH`를 함께 갱신하도록 명시했다.
- 각 task에 failing test, expected failure, implementation, verification command를 포함했다.
- top-level Git 부재를 명시해 존재하지 않는 commit workflow를 요구하지 않는다.
