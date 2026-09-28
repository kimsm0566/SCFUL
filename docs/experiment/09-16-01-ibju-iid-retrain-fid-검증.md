# IBJU_FedAvg IID: 전체 데이터 retrain-reference FID 검증

## 범위

현재 IBJU_FedAvg의 완료된 IID seed 0 checkpoint를 다시 평가한다. 학습과 unlearning은
추가로 수행하지 않고 기존 checkpoint와 기존 결과는 변경하지 않는다. 현재 버전의
non-IID 본 실험 checkpoint는 없으므로, 이전 FSCUv2 non-IID 결과를 현재 방법의 결과로
섞지 않는다.

- Source: `outputs/2026-09-10-21-ibju-fedavg-iid/20260910T140049Z_seed-0`
- Source config: 위 source의 `config.json`
- 등록 config: `src/configs/federated_cifar10_iid_vdeepjscc_ibju_fedavg.json`
- Global seed: 0, split: IID, target client: 0
- Original 학습: CIFAR-10 49k, 10 clients. Retrain은 target client를 제외한 44,100장,
  9 clients. 둘 다 800 rounds, latent channels 16, batch 200, beta 0.0001, AWGN 20 dB.
- JU: prior-KL collapse + reconstruction ascent, encoder/decoder joint update
- Repair: remaining 9 clients, local 1 epoch, 100 rounds, no teacher/KD/conflict projection
- FID audit output root: `outputs/2026-09-16-01-retrain-fid-ibju-iid`

## 기준 모델 검증

평가 전 `load_fscu_reference_models`로 전체 training scenario fingerprint를 비교하고
initial/original/retrain checkpoint SHA-256 및 source run의 symlink가 같은 checkpoint를
가리키는지 확인했다. 검증 결과는 모두 일치했다.

- Reference bank: `outputs/2026-09-10-08-fscuv2-iid-v3-49k-seed0/20260909T165825Z_seed-0`
- Scenario fingerprint: `db90edeac7726a4f7ba07b2a813b61d2e4fc94595481f85f329a51824ed44451`
- Initial SHA-256: `5cb7daf0b6a63ad615af3be2a3b2bb277cd1a46d0bc47038c7dd271a9df07bbc`
- Original SHA-256: `560fa6f7e7f16f382f2cb059be81d020fb66fff2cde7d8198cceaf019fd6384c`
- Retrain SHA-256: `6d046e1bce277b29557e19dfcf122c06abf9b336719709f853bfbb46b7ae744d`

Reference 폴더의 과거 method 이름은 현재 평가 method를 뜻하지 않는다. 여기서는
검증된 Original/Retrain reference state만 재사용한다.

## 평가 protocol

- Clean: 공식 CIFAR-10 test 10,000장 전체
- Erased: target client 0의 local training 4,900장 전체, 기록된 backdoor 위치 재현
- 대상: Original, Retrain, Initial, JU 직후, repair 1/5/10/20/50/100 rounds
- 최종 checkpoint는 기존에 정한 100 rounds. 이번 test FID를 보고 checkpoint를 재선택하지 않는다.
- Posterior mean + paired AWGN 20 dB, evaluation batch 200
- Frozen pytorch-fid 0.3.0 Inception pool3 2048-D, feature batch 32
- Reference는 같은 입력과 같은 AWGN에서 얻은 **Retrain의 복원 출력**이다.
- FID의 낮은 값은 출력 특징 분포가 가깝다는 뜻이며, 그 자체로 삭제 보장을 뜻하지 않는다.
- 기존 PSNR/RMSE/functional gap 및 semantic/privacy 지표도 같은 audit에서 기록한다.
- 이 평가는 posterior-sampled MC10 training diagnostic과 구분한다.

## 사전 해석 기준

최종 100-round 모델과 Original을 각각 같은 Retrain 기준에 비교한다. Clean과 erased
FID가 모두 Original보다 작으면 이 seed에서 분포 수준 접근이 개선된 것으로 해석한다.
둘 다 커지면 FID에서도 개선을 확인하지 못한 것이다. 한쪽만 개선되면 split 간 trade-off로
보고한다. JU 직후보다 좋아진 것만으로는 삭제 성공이라고 판단하지 않는다.

FID가 개선되어도 paired-output MSE와 반대 방향이면 두 지표가 측정하는 특성이 다른
것으로 보고, 삭제 보장이나 sample-wise retrain equivalence로 확대 해석하지 않는다.
단일 seed 결과이므로 학습 seed 간 변동을 넘어선 유의한 개선도 주장하지 않는다.

## 실행 명령

```bash
docker compose exec -T scful-dev env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -u src/main.py --experiment reconstruction_channel_audit \
  --source-run outputs/2026-09-10-21-ibju-fedavg-iid/20260910T140049Z_seed-0 \
  --config src/configs/federated_cifar10_iid_vdeepjscc_ibju_fedavg.json \
  --output-root outputs/2026-09-16-01-retrain-fid-ibju-iid \
  --seed 0 --snr_db 20 --batch_size 200 --audit-device cuda:0
```

실행 시 source run의 저장 config를 실제 평가 설정으로 읽는다. 재실행은 새 날짜/순번의
output root를 사용해야 한다.

## 결과

완료 run: `outputs/2026-09-16-01-retrain-fid-ibju-iid/20260916T093831Z_seed-0`

| Model / stage | Clean FID-to-Retrain (N=10,000) | Erased FID-to-Retrain (N=4,900) |
| --- | ---: | ---: |
| Original | 2.671686 | 5.369075 |
| Retrain self-comparison | 0.000000 | 0.000000 |
| Initial | 549.549122 | 551.527549 |
| JU-only | 560.601582 | 560.820866 |
| Repair round 1 | 552.930132 | 553.858728 |
| Repair round 5 | 256.827871 | 258.915260 |
| Repair round 10 | 89.697512 | 93.839258 |
| Repair round 20 | 8.939973 | 12.552836 |
| Repair round 50 | 3.033751 | 5.943922 |
| Repair round 100 (등록 final) | 2.734761 | 5.498305 |

최종 모델의 FID는 Original보다 clean에서 2.3609%, erased에서 2.4069% 크다.
FID 관점에서도 final 모델이 Original보다 Retrain에 가까워졌다는 개선을 확인하지
못했다. 다만 약 2.4% 차이의 통계적 유의성은 single-seed 평가로 판단하지 않는다.

JU 이후 분포 손상은 repair를 거치며 크게 줄어들고, 기록된 checkpoint 중 final
100 rounds가 가장 낮다. 그러나 이는 **JU로 손상된 utility의 회복**을 보여줄 뿐,
Original 대비 더 나은 retrain equivalence나 target contribution 삭제를 증명하지 않는다.
Test 결과를 보고 round budget이나 final checkpoint 선택은 변경하지 않았다.

기존 paired-output MSE 결과와도 같은 방향이다.

| Model | Clean output MSE gap to Retrain | Erased output MSE gap to Retrain |
| --- | ---: | ---: |
| Original | 0.000520168 | 0.000525412 |
| Final Unlearned | 0.000556724 | 0.000563343 |

현재 결과는 IID seed 0에 한정한다. Non-IID, 다른 training seeds, 다른 method의 우열로
확장하지 않는다. BA 역시 이전과 동일하게 Original/Retrain/Final 모두 약 0.68%여서
이 시나리오의 삭제 식별 근거로 사용하지 않는다.

## 산출물 검증

- `metrics/retrain_fid.csv`: 10 models x 2 splits = 20행, 모두 finite/nonnegative
- `metrics/headline_metrics.csv`: 10 models, 등록 headline 지표 모두 생성
- `metrics/retrain_fid_protocol.json`: feature weights hash, 전처리, reference, seed, SNR,
  전체 표본 수 및 runtime scope 기록
- `metrics/scu_extended_metrics.csv`: 기존 output gap을 포함한 보조 지표
- `figures/retrain_fid_comparison.png`: 전체 checkpoint의 clean/erased FID
- `figures/unlearning_headline_comparison.png`: 기존 headline 비교
- `figures/target_information_rate.png`: prior-KL 보조 진단

모든 그래프는 같은 `src/main.py` 실행에서 생성되었고 해당 run의 CSV/JSON으로 뒷받침된다.
전체-stage FID 그래프는 JU/초기 손상 값이 커 최종 모델 간 작은 차이가 눈에 잘 보이지
않으므로 위 표와 raw CSV로 Original/Final 차이를 함께 확인한다.

두 split의 Retrain self-FID는 모두 0이며 이는 정의상 자기 비교 결과다. 이전 source
평가와 PSNR, paired-output MSE gap, MIA AUC 및 BA를 대조한 최대 절대 차이는 0이었다.
Feature model load 및 clean/erased reconstruction/FID 평가 합계는 292.361초이며
기존 training/unlearning algorithm Runtime에 더하지 않았다.
