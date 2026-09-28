# Main/Forward 중심 코드 구조 정리

## 목적

프로젝트 코드의 실행 흐름을 쉽게 따라갈 수 있도록 다음 규칙으로 정리했다.

- 모든 실험은 `src/main.py`에서 시작한다.
- 프로젝트가 직접 관리하는 `src/` 코드에는 함수나 클래스 위에 붙는 데코레이터
  문법을 사용하지 않는다.
- codec의 학습과 언러닝은 모델 내부 메서드를 직접 호출하지 않고 `forward`를
  통해 수행한다.
- 세부 알고리즘은 기존처럼 `src/training/`, `src/unlearning/`,
  `src/experiments/`에 나누어 둔다.

외부 기준 구현인 `third_party/`는 원본 보존을 위해 수정하지 않았다.

## 실행 분기

현재 주 실행은 `src/main.py --algorithm FSCU`다. `main.py`는 Trainer의 FSCU
분기로만 넘기고 Trainer가 다음 과정을 연속 수행한다.

- Original과 target 제외 Retrain 준비
- target-client joint unlearning
- server public-data contrastive compensation
- 전체 headline metric 자동 평가

현재 `FSCU`는 public-data 버전만 뜻한다. 이후 변경 방법은 별도 version으로
새 분기를 만든다. 과거 checkpoint 재평가와 baseline 재현 명령은 보조 경로로만
유지하며 기본 FSCU 실행에 필요하지 않다.

```text
python src/main.py --algorithm FSCU --config <config> --output-root <output>
```

## Codec forward 사용법

세 CIFAR-10 codec과 MNIST codec은 동일한 mode 이름을 사용한다.

| mode | 의미 | 주요 반환값 |
| --- | --- | --- |
| `training` | posterior sampling과 channel을 포함한 학습 경로 | reconstruction, posterior, transmitted latent |
| `posterior` | encoder posterior 계산 | mean, log variance |
| `mean_reconstruction` | posterior mean을 사용한 복원 | codec output |
| `paired_awgn` | 전달받은 표준 잡음으로 AWGN 복원 | codec output |
| `fixed_awgn` | 전달받은 실제 잡음으로 복원 | codec output |
| `semantic` | 복원 영상의 분류 head 계산 | class logits |

학습과 언러닝 구현에서는 다음 형태로 호출한다.

```python
output = model(images, mode="training", snr_db=20.0)
posterior = model(images, mode="posterior")
logits = model(output.reconstruction, mode="semantic")
```

## 데코레이터 제거 범위

`src/`의 dataclass 설정과 결과 객체는 일반 클래스와 명시적인 생성자로 바꿨다.
no-grad 함수는 함수 내부의 `with torch.no_grad()` 블록을 사용한다. class method와
static method도 일반 함수 또는 일반 instance method로 바꿨다.

## 검증

- 모든 `src/` Python 파일 compile 검사 통과
- 전체 unit/smoke test 125개 통과
- 네 codec의 `forward` posterior/training 경로 shape 확인
- CIFAR-10 codec 세 종류의 semantic forward shape 확인
