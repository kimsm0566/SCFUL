# `trainer.py` SCU 구현 계획

> **에이전트 작업자용:** 이 계획을 실행할 때는 `superpowers:executing-plans`를 사용한다. 각 단계는 체크박스 순서대로 수행한다.

**목표:** 기존 FL 흐름을 변경하지 않고 `src/utils/trainer.py`에서 단일 클라이언트의 SCU 학습을 선택할 수 있게 한다.

**구조:** `Trainer.train()`은 알고리즘 이름에 따라 기존 `train_fl()` 또는 새 `train_scu()`를 호출한다. `train_scu()`는 한 client의 `node.data`를 target/retained로 seed 고정 분할하고, 먼저 전체 data로 codec을 학습한 후 같은 `trainer.py` 안의 SCU loss loop로 target contribution을 지운다. 새 production 파일이나 wrapper는 만들지 않는다.

**기술:** Python, PyTorch, MPI, `MNISTSemanticCodec`, `adapt_scu_style`, pytest

## 공통 제약

- Production 코드 변경은 `src/utils/trainer.py` 한 파일로 제한한다.
- 기존 `FL` 함수와 로직을 수정하지 않는다.
- SCU는 단일 client에서만 동작하며, server에 retained raw data를 보내지 않는다.
- SCU 모델은 `node.model`을 사용하며 `src/models/semantic_codec.py`의 codec을 전제로 한다.
- `node.data`는 `(images, labels)` tuple이며 MNIST `(N, 1, 28, 28)` 또는 `(N, 28, 28)` input이어야 한다. Pixel value가 255-scale이면 trainer에서 0--1 range로 변환한다.
- target split은 `args.scu_forget_ratio`(기본 `0.1`)와 `args.seed`로 고정한다. `D^t`와 `D^r`가 모두 비지 않도록 검증한다.
- 타입이 복잡한 함수 선언과 별도 wrapper 파일을 만들지 않는다.

---

### 작업 1: SCU import와 알고리즘 선택 분기 검증

**파일:**

- 생성: `src/tests/test_trainer_scu.py`
- 수정: `src/utils/trainer.py`

**인터페이스:**

- 입력: `Trainer.train(node, test_data, comm, logger, args)`에서 `args.algorithm == "SCU"`
- 출력: SCU 결과 dictionary

- [x] **1단계: 실패하는 import 테스트 작성**

```python
def test_trainer_import_does_not_require_unused_fl_packages():
    result = subprocess.run(
        ["python", "-c", "import utils.trainer"],
        env={"PYTHONPATH": "<repository>/src"},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
```

- [x] **2단계: 기존 eager import 때문에 실패하는지 확인**

```bash
pytest -q src/tests/test_trainer_scu.py::test_trainer_dispatches_scu_algorithm
```

기대 결과: 기존 `wandb` import 때문에 실패한다.

- [x] **3단계: FL 전용 import를 해당 FL 함수 내부로 이동**

SCU를 실행할 때 `wandb`, `data.data`, `utils.eval`, `models.channel`이 import되지 않게 한다. FL 함수가 호출될 때만 필요한 dependency를 import한다.

- [x] **4단계: import 테스트 통과 확인**

```bash
pytest -q src/tests/test_trainer_scu.py::test_trainer_dispatches_scu_algorithm
```

확인 결과: PASS.

### 작업 2: 단일 client SCU 학습 경로 검증

**파일:**

- 생성: `src/tests/test_trainer_scu.py`
- 수정: `src/utils/trainer.py`

**인터페이스:**

- 입력: `node.model`, `node.data`, `args`의 SCU hyperparameter
- 출력: `{"algorithm", "model", "original_loss_trace", "scu_loss_trace", "steps", "target_examples", "retained_examples"}`

- [x] **1단계: 실패하는 end-to-end CPU 테스트 작성**

```python
def test_train_scu_updates_a_single_client_codec():
    node = SimpleNamespace(
        model=MNISTSemanticCodec(CodecConfig(latent_dim=2, encoder_hidden_dim=4, decoder_hidden_dim=4)),
        data=(torch.cat([torch.zeros(2, 1, 28, 28), torch.ones(2, 1, 28, 28)]), torch.tensor([0, 0, 1, 1])),
    )
    args = SimpleNamespace(
        device=torch.device("cpu"), n_epochs=1, batch_size=2, lr=1e-4,
        snr_db=10.0, seed=0, scu_forget_ratio=0.5,
        lambda_kl=1.0, lambda_rec=0.1, lambda_ss=1.0,
        beta=1e-4, epsilon=1e-8,
    )

    result = Trainer(args).train(node, None, None, None, args)

    assert result["algorithm"] == "SCU"
    assert result["steps"] == 1
    assert result["target_examples"] == 2
    assert result["retained_examples"] == 2
```

- [x] **2단계: 테스트가 SCU data/config 경로 부재로 실패하는지 확인**

```bash
pytest -q src/tests/test_trainer_scu.py::test_train_scu_updates_a_single_client_codec
```

기대 결과: SCU 결과 dictionary 또는 dataset/config 변환이 아직 없어서 실패한다.

- [x] **3단계: 읽기 쉬운 SCU 경로 구현**

`trainer.py` 내부에 다음의 작은 함수만 추가한다.

```python
def _split_scu_data(data, forget_ratio, seed):
    return target_data, retained_data


def _train_original_codec(model, full_data, args):
    return original_loss_trace


def train_scu(comm, node, args, logger):
    full_data = _make_scu_dataset(node.data)
    target_data, retained_data = _split_scu_data(...)
    original_loss_trace = _train_original_codec(node.model, full_data, args)
    scu_loss_trace = _run_scu_unlearning(node.model, target_data, retained_data, args)
    return {...}
```

각 helper는 `trainer.py`에서 직접 읽을 수 있는 짧은 PyTorch loop만 가진다. Forget loss는 `lambda_kl * KL(q_U || q_G) - lambda_rec * MSE(x_hat_t, x_t)`이고 retained loss는 `beta * KL(q_U || N(0,I)) + MSE(x_hat_r, x_r)`다. `args.scu_mode`가 없으면 `self_sharing`을 사용하고, `forget_only`이면 forget loss만 최적화한다. `args`에 SCU 전용 coefficient가 없으면 명시된 기본값을 사용한다. single client data 이외의 MPI 통신은 하지 않는다.

- [x] **4단계: 실제 Trainer SCU 테스트 통과 확인**

```bash
pytest -q src/tests/test_trainer_scu.py
python -m py_compile src/utils/trainer.py
```

확인 결과: import test와 실제 Trainer SCU 테스트가 통과한다.

### 작업 3: 회귀 확인

**파일:**

- 수정 없음

- [x] **1단계: 기존 SCU loss 테스트 실행**

```bash
pytest -q src/tests/test_scu_style.py
```

확인 결과: 기존 SCU-style loss 테스트가 통과한다.

- [x] **2단계: 변경 범위 확인**

```bash
git diff -- src/utils/trainer.py src/tests/test_trainer_scu.py
```

최상위 Git 저장소가 없는 workspace이므로, 파일별 `diff` 대신 수정 파일 목록과 `trainer.py` 내용을 직접 검토한다.

## 자체 검토

- FL 로직 수정 금지: 작업 1·2는 `Trainer.train()`의 SCU 분기와 새 SCU 함수만 추가한다.
- single client 원칙: 작업 2는 `node.data`에서 만든 target/retained dataset만 읽고 MPI 통신을 하지 않는다.
- semantic codec 원칙: 작업 2는 `MNISTSemanticCodec`가 제공하는 posterior, training sample reconstruction과 AWGN 경로를 사용한다.
- 코드 이해성: helper는 세 개 이하이며 모든 함수 선언은 한 줄로 작성한다.
- 불확실한 dataset 자동 분할은 넣지 않는다. main/client가 `target_data`와 `retained_data`를 제공해야 한다.

### 작업 4: CIFAR-10 전용 semantic codec과 SCU 연결

**파일:**

- 수정: `src/models/semantic_codec.py`
- 수정: `src/data/data.py`
- 수정: `src/utils/client.py`
- 수정: `src/utils/trainer.py`
- 수정: `src/utils/option.py`
- 생성: `src/tests/test_cifar10_semantic_codec.py`

**인터페이스:**

- 입력: CIFAR-10 image `(B, 3, 32, 32)`
- 출력: reconstruction `(B, 3, 32, 32)`과 MNIST codec과 같은 posterior/channel interface

- [ ] **1단계: 실패하는 CIFAR-10 codec shape 테스트 작성**

```python
model = CIFAR10SemanticCodec(CIFAR10CodecConfig(latent_dim=8))
output = model.decode_posterior_mean(torch.zeros(2, 3, 32, 32))
assert output.reconstruction.shape == (2, 3, 32, 32)
```

- [ ] **2단계: convolutional CIFAR-10 codec 구현**

`3x32x32 -> 32 -> 64 -> 128` stride-2 convolution encoder와 반대 순서의 transposed-convolution decoder를 구현한다. Latent posterior, power normalization, AWGN path는 MNIST codec과 같은 public method를 제공한다.

- [ ] **3단계: SCU dataset/model 선택 연결**

`data.py`는 `dataset_name`에 따라 MNIST 또는 CIFAR-10 shape를 검증한다. `client.py`와 `trainer.py`는 `args.dataset`에 따라 알맞은 codec을 선택·검증한다.

- [ ] **4단계: 기본 삭제 비율과 검증**

`option.py`에 `--scu_forget_ratio`를 추가하고 기본값을 `0.06`으로 둔다. CIFAR codec, SCU data split, Trainer smoke test를 실행한다.
