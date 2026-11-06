# 800-round Original/Retrain reference checkpoint bank

## 목적

새 unlearning 프레임워크를 반복 실험할 때마다 Original과 Retrain을 다시 학습하지 않는다. 완전히 지정된 각 시나리오마다 800라운드 reference를 한 번만 고정하고, 이후 방법은 동일 fingerprint의 reference를 로드해 비교한다.

## 고정 시나리오

다음 6개 조합을 seed 0으로 고정했다.

1. IID + 방식 1 가중치 오염
2. IID + 방식 2 데이터 오염
3. shared-class-2 + 방식 1 가중치 오염
4. shared-class-2 + 방식 2 데이터 오염
5. Dirichlet 0.1 + 방식 1 가중치 오염
6. Dirichlet 0.1 + 방식 2 데이터 오염

모든 reference는 800 global rounds, 10 clients, 1 local epoch이며, headline audit은 paired AWGN posterior mean 20 dB로 통일했다. shared-class-2의 target은 client 0이고 주 클래스는 airplane/automobile이다. Dirichlet 0.1의 target은 client 8이고 주 클래스는 deer/automobile/ship이다.

## 저장 위치

통합 색인과 핵심 수치는 `outputs/2026-10-04-15-cifar10-reference-bank-800rounds/`에 저장한다.

- `reference_bank.json`: source/audit 경로, fingerprint, checkpoint SHA-256
- `metrics/reference_core_metrics.csv`: 6 시나리오 x Original/Retrain 12행
- `metrics/reference_marker_metrics.csv`: 방식별 marker 기준값
- `metrics/target_client_summary.csv`: target client 데이터 분포 요약

각 entry의 source run에는 `initial.pt`, `original.pt`, `retrain_without_client_<id>.pt`, `reference_manifest.json`, `split_manifest.json`, 학습 이력과 학습 곡선이 있다. 실제 체크포인트 SHA-256을 manifest와 다시 대조했으며 18개 파일이 모두 일치했다.

## 핵심지표 정책

주 의사결정에는 clean/erased RMSE-255, PSNR, 입력 대비 SSIM, Retain u-FID,
semantic accuracy, encoder의 Retrain 대비 posterior-mean MSE와 symmetric posterior
KL, decoder의 FID/PSNR/SSIM-to-Retrain, 방식별 marker 지표와 runtime을 사용한다.

Client-ID probe는 제외한다. 정규화 Client-IDI는 이번 기준들에서 Original-minus-Retrain 분모가 안정적이지 않아 사용하지 않는다. reconstruction MIA는 repository headline artifact에는 계속 남기되 프레임워크 선택에는 보조 자료로만 쓴다.

방식 1은 Marker-PUL을 핵심 marker 지표로 쓰고, masked-input 방식 2는 latent/output
probe sensitivity와 marker projection을 쓴다. 방식 2 Marker-PUL은 호환 진단으로만
남긴다. Marker 지표만으로 unlearning을 판정하지 않고 반드시 Retrain에 대한
output/encoder gap 및 retain utility와 함께 해석한다. 상세 정의는
[`10-08-02-핵심-평가지표-최신화.md`](10-08-02-핵심-평가지표-최신화.md)를 따른다.

## 재사용 규칙

새 방법 실행 시 정확히 일치하는 entry의 `source`를 `--reference-run`으로 사용한다. 다음 중 하나라도 다르면 reference를 재사용하지 않는다.

- codec과 initial-state hash
- client split manifest와 target client
- marker/backdoor realization
- public-data reservation
- round/local epoch/batch/optimizer/learning-rate
- train channel/SNR와 기타 학습 영향 옵션

10 dB로 잘못 실행된 `outputs/2026-10-04-11-reference-audit-shared2-marker-weight-800rounds` 및 중단된 root 10, 12, 13, 14는 이 기준에서 제외한다.
