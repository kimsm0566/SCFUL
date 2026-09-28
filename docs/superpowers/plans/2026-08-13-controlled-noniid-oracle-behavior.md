# 통제된 Non-IID Oracle Behavior 실험 구현 계획

> **에이전트 작업자용:** 이 계획은 task별로 `superpowers:executing-plans`를 사용해 순서대로 구현한다. 각 단계는 체크박스로 추적한다.

**목표:** Whole target client를 삭제한 뒤 SCU-style centralized VAE가 retraining oracle의 재구성 및 posterior behavior를 얼마나 근사하는지 재현 가능하게 측정한다.

**구조:** 새 MNIST VAE codec은 posterior, power-normalized AWGN channel, deterministic reconstruction API를 분리한다. Split/학습/SCU adaptation/평가/산출물을 독립 module로 나누고, experiment runner가 단일 config와 seed로 모델 5개, audit set 3개, 표와 그림을 생성한다.

**기술 스택:** Python 3.10, PyTorch 2.4.1+cu121, torchvision 0.19.1, NumPy 1.26, Matplotlib 3.9, 표준 `unittest`.

## 전역 제약

- 새 executable code와 test는 `src/` 아래에만 둔다. `src/SCU/`의 legacy algorithm은 수정하지 않는다.
- Markdown 문서는 한국어로 작성한다.
- Codec은 posterior sampling 뒤 `sqrt(d) * z / (||z|| + epsilon_power)` power normalization과 AWGN을 반드시 사용한다.
- `original`, `retrain`, `retrain_replica`는 동일 `initial.pt` state에서 출발한다.
- `scu_forget_only`는 `L_forget`만, `scu_self_sharing`은 명세의 dynamic-weight objective만 사용한다. Teacher forward는 gradient가 없어야 한다.
- 평가의 AWGN noise는 probe/repetition별로 모든 model에 공통이다. Functional gap은 draw별로 계산한 뒤 평균한다.
- Target training-member, target held-out, retained held-out probe를 분리하고 CSV에 dataset split과 source index를 모두 기록한다.
- 완료된 original codec에서 client별 aggregated posterior를 moment-matched 단일 diagonal Gaussian으로 fit하고, held-out empirical posterior 대비 충분성을 측정한다.
- 단일 Gaussian 충분성은 MMD와 empirical sampling noise floor, off-diagonal correlation, excess kurtosis, label-mixture NLL gain으로 평가한다. Label mixture는 진단용 upper reference이며 method summary가 아니다.
- 모든 run은 non-overwriting output directory, config/provenance, checkpoint, CSV/JSON, prediction tensor, 두 figure를 남긴다.
- Top-level workspace는 Git repository가 아니므로 task별 commit을 만들 수 없다.

---

### Task 1: 재현성·통제된 client split 기반 구축

**파일:**
- 생성: `src/__init__.py`
- 생성: `src/utils/__init__.py`
- 생성: `src/utils/reproducibility.py`
- 생성: `src/utils/client_splits.py`
- 생성: `src/tests/__init__.py`
- 생성: `src/tests/test_client_splits.py`

**인터페이스:**
- `build_controlled_noniid_split(labels: Tensor, split_seed: int, shared_digits: tuple[int, ...], shared_per_digit: int, target_specific_digit: int, retained_clients: int) -> ControlledClientSplit`
- `select_audit_indices(split: ControlledClientSplit, train_labels: Tensor, test_labels: Tensor, probe_seed: int, target_per_group: int, retained_total: int) -> dict[str, list[ProbeRef]]`
- `ControlledClientSplit.manifest() -> dict[str, object]`
- `derive_seed(global_seed: int, namespace: str) -> int`, `set_global_seed(seed: int, deterministic: bool) -> None`, `collect_environment(device: torch.device) -> dict[str, object]`

- [ ] **Step 1: 실패할 split contract test를 작성한다.**

```python
def test_controlled_split_is_complete_balanced_and_removes_digit_one(self):
    labels = torch.tensor([digit for digit in range(10) for _ in range(24)])
    split = build_controlled_noniid_split(
        labels, split_seed=7, shared_digits=(3, 4, 5, 6),
        shared_per_digit=6, target_specific_digit=1, retained_clients=4,
    )
    self.assertEqual(set(split.target_groups["target_specific"]), set(range(24, 48)))
    self.assertEqual(len(set().union(*split.client_indices.values())), len(labels))
    self.assertEqual(sum(len(v) for v in split.client_indices.values()), len(labels))
    retained = [i for client in range(1, 5) for i in split.client_indices[f"client_{client}"]]
    self.assertFalse(torch.any(labels[retained] == 1))
    retained_sizes = [len(split.client_indices[f"client_{client}"]) for client in range(1, 5)]
    self.assertLessEqual(max(retained_sizes) - min(retained_sizes), 1)
```

- [ ] **Step 2: test가 module 부재로 실패하는지 확인한다.**

실행: `docker compose exec -T scful-dev python -m unittest src.tests.test_client_splits -v`

예상: `ModuleNotFoundError: No module named 'src.utils.client_splits'`.

- [ ] **Step 3: seed, manifest digest, rotating round-robin split, 세 audit partition 선택을 최소 구현한다.**

`ControlledClientSplit`은 `client_0` target, `client_1`--`client_4` retained index와 target group index를 보관한다. Class chunk가 끝날 때 assignment cursor를 다음 client로 이어서 전체 retained size 차이가 1 이하가 되게 한다. `ProbeRef`는 `dataset_split`, `source_index`, `probe_partition`, `support_group`을 보관한다.

- [ ] **Step 4: split test를 통과시키고 manifest assertion을 검사한다.**

실행: `docker compose exec -T scful-dev python -m unittest src.tests.test_client_splits -v`

예상: 1개 이상 test 통과, digit-1 retained count는 0.

### Task 2: Power-normalized AWGN VAE codec 구현

**파일:**
- 생성: `src/models/__init__.py`
- 생성: `src/models/mnist_codec.py`
- 생성: `src/tests/test_mnist_codec.py`

**인터페이스:**
- `CodecConfig(latent_dim, encoder_hidden_dim, decoder_hidden_dim, activation, logvar_min, logvar_max, epsilon_power)`
- `MNISTSemanticCodec(config).posterior(x) -> Posterior`
- `MNISTSemanticCodec.normalized_latent(z) -> Tensor`
- `MNISTSemanticCodec.decode_posterior_mean(x) -> CodecOutput`
- `MNISTSemanticCodec.decode_with_awgn(x, noise: Tensor) -> CodecOutput`
- `Posterior(mu: Tensor, logvar: Tensor)` 및 `CodecOutput(reconstruction: Tensor, posterior: Posterior)`

- [ ] **Step 1: 실패할 channel contract test를 작성한다.**

```python
def test_power_normalization_and_fixed_awgn_noise_are_explicit(self):
    model = MNISTSemanticCodec(CodecConfig(latent_dim=4, encoder_hidden_dim=8, decoder_hidden_dim=8))
    z = torch.tensor([[3.0, 4.0, 0.0, 0.0]])
    normalized = model.normalized_latent(z)
    self.assertAlmostEqual(normalized.norm(dim=1).item(), 2.0, places=6)
    x = torch.zeros(2, 1, 28, 28)
    noise = torch.zeros(2, 4)
    deterministic = model.decode_posterior_mean(x).reconstruction
    noisy = model.decode_with_awgn(x, noise).reconstruction
    self.assertTrue(torch.allclose(deterministic, noisy))
```

- [ ] **Step 2: test가 codec module 부재로 실패하는지 확인한다.**

실행: `docker compose exec -T scful-dev python -m unittest src.tests.test_mnist_codec -v`

예상: `ModuleNotFoundError: No module named 'src.models.mnist_codec'`.

- [ ] **Step 3: MLP encoder/decoder, clamped diagonal posterior, power normalization, deterministic/fixed-noise decoder를 구현한다.**

Posterior sampling API는 reparameterization generator를 받는다. Training path는 sampled posterior latent에 AWGN noise를 더하고, evaluation path는 caller가 생성한 fixed noise만 사용한다. Reconstruction tensor는 항상 `(B, 1, 28, 28)`이다.

- [ ] **Step 4: codec test와 shape check를 통과시킨다.**

실행: `docker compose exec -T scful-dev python -m unittest src.tests.test_mnist_codec -v`

예상: power norm과 zero-noise equality가 통과한다.

### Task 3: Reconstruction trainer와 정확한 SCU-style adaptation 구현

**파일:**
- 생성: `src/training/__init__.py`
- 생성: `src/training/reconstruction.py`
- 생성: `src/unlearning/__init__.py`
- 생성: `src/unlearning/scu_style.py`
- 생성: `src/tests/test_scu_style.py`

**인터페이스:**
- `TrainingConfig(epochs, batch_size, learning_rate, beta, snr_db, ...)`
- `train_reconstruction(model, dataset, config, device, data_order_seed, posterior_seed, channel_seed) -> TrainingResult`
- `SCUConfig(epochs, batch_size, learning_rate, lambda_kl, lambda_rec, lambda_ss, beta, epsilon, snr_db)`
- `adapt_scu_style(original, target_dataset, retained_dataset, config, device, mode, seeds) -> AdaptationResult`
- `diagonal_kl(p: Posterior, q: Posterior) -> Tensor`

- [ ] **Step 1: 실패할 SCU objective test를 작성한다.**

```python
def test_forget_only_uses_only_the_declared_forgetting_loss(self):
    target = TensorDataset(torch.zeros(4, 1, 28, 28), torch.zeros(4, dtype=torch.long))
    retained = TensorDataset(torch.ones(4, 1, 28, 28), torch.ones(4, dtype=torch.long))
    result = adapt_scu_style(
        original=self.model, target_dataset=target, retained_dataset=retained,
        config=SCUConfig(epochs=1, batch_size=4, lambda_kl=1.0, lambda_rec=0.1, lambda_ss=9.0),
        device=torch.device("cpu"), mode="forget_only", seeds=self.seeds,
    )
    self.assertEqual(result.steps, 1)
    self.assertEqual(result.loss_trace[0]["mode"], "forget_only")
    self.assertNotIn("retain_loss", result.loss_trace[0]["optimized_terms"])
```

- [ ] **Step 2: test가 adaptation module 부재로 실패하는지 확인한다.**

실행: `docker compose exec -T scful-dev python -m unittest src.tests.test_scu_style -v`

예상: `ModuleNotFoundError: No module named 'src.unlearning.scu_style'`.

- [ ] **Step 3: trainer와 SCU objective를 구현한다.**

`train_reconstruction`은 model마다 새 Adam optimizer를 만들고, training sample posterior와 power-normalized AWGN으로 `MSE + beta * KL(q||p)`를 최소화한다. `adapt_scu_style`은 original copy를 freeze하고 no-grad teacher posterior를 사용한다. `mode="forget_only"`에서는 `L_forget`만, `mode="self_sharing"`에서는 detach한 absolute scalar weight로 `L_self-sharing`만 최소화한다. Retained batch는 target update 수와 동일하게 cycling한다.

- [ ] **Step 4: SCU test와 작은 CPU training smoke를 통과시킨다.**

실행: `docker compose exec -T scful-dev python -m unittest src.tests.test_scu_style -v`

예상: forget-only가 정확한 mode/term trace를 기록하고 한 update를 수행한다.

### Task 4: Functional-gap metric, provenance, artifact writer 구현

**파일:**
- 생성: `src/utils/oracle_metrics.py`
- 생성: `src/utils/oracle_artifacts.py`
- 생성: `src/tests/test_oracle_metrics.py`

**인터페이스:**
- `symmetric_diagonal_kl(p: Posterior, q: Posterior) -> Tensor`
- `summarize_records(records: list[dict[str, object]], bootstrap_seed: int, resamples: int) -> dict[str, object]`
- `RunArtifacts.create(output_root: Path, seed: int) -> RunArtifacts`
- `RunArtifacts.write_json(relative_path: str, payload: object) -> Path`
- `RunArtifacts.write_records(records: list[dict[str, object]]) -> Path`
- `RunArtifacts.save_checkpoint(name: str, state: dict[str, Tensor]) -> Path`

- [ ] **Step 1: hand-calculated functional-gap test를 작성한다.**

```python
def test_symmetric_kl_is_zero_for_equal_posteriors_and_positive_otherwise(self):
    p = Posterior(mu=torch.tensor([[0.0, 0.0]]), logvar=torch.tensor([[0.0, 0.0]]))
    q = Posterior(mu=torch.tensor([[1.0, 0.0]]), logvar=torch.tensor([[0.0, 0.0]]))
    self.assertEqual(symmetric_diagonal_kl(p, p).item(), 0.0)
    self.assertAlmostEqual(symmetric_diagonal_kl(p, q).item(), 0.5, places=6)
```

- [ ] **Step 2: test가 metric module 부재로 실패하는지 확인한다.**

실행: `docker compose exec -T scful-dev python -m unittest src.tests.test_oracle_metrics -v`

예상: `ModuleNotFoundError: No module named 'src.utils.oracle_metrics'`.

- [ ] **Step 3: diagonal KL, draw-wise aggregate, bootstrap summary, atomic artifact writer를 구현한다.**

Summary key는 `probe_partition/support_group/model/evaluation_condition`이다. Writer는 run directory와 required subdirectory를 한번 만들고, JSON/CSV/checkpoint/prediction/figure path를 명시적으로 반환한다.

- [ ] **Step 4: metric/artifact test를 통과시킨다.**

실행: `docker compose exec -T scful-dev python -m unittest src.tests.test_oracle_metrics -v`

예상: hand-calculated symmetric KL, summary grouping, non-overwriting artifact directory가 통과한다.

### Task 5: Client semantic distribution 충분성 진단 구현

**파일:**
- 생성: `src/utils/client_posterior.py`
- 생성: `src/tests/test_client_posterior.py`

**인터페이스:**
- `ClientGaussianSummary(client_id, mean, variance, metrics, fit_count, evaluation_count)`
- `fit_moment_matched_diagonal_gaussian(mu: Tensor, logvar: Tensor, minimum_variance: float) -> Posterior`
- `audit_client_gaussian_sufficiency(model, dataset, client_indices, labels, config, device, seed) -> list[ClientGaussianSummary]`
- `rbf_mmd_squared(x: Tensor, y: Tensor, bandwidth: float | None = None) -> Tensor`

- [ ] **Step 1: aggregated-posterior moment test를 먼저 작성한다.**

```python
def test_moment_matched_variance_contains_posterior_and_between_sample_variance(self):
    mu = torch.tensor([[0.0], [2.0]])
    logvar = torch.log(torch.tensor([[1.0], [1.0]]))
    summary = fit_moment_matched_diagonal_gaussian(mu, logvar, minimum_variance=1e-6)
    self.assertAlmostEqual(summary.mu.item(), 1.0, places=6)
    self.assertAlmostEqual(summary.logvar.exp().item(), 2.0, places=6)
```

- [ ] **Step 2: test가 module 부재로 실패하는지 확인한다.**

실행: `docker compose exec -T scful-dev python -m unittest src.tests.test_client_posterior -v`

예상: `ModuleNotFoundError: No module named 'src.utils.client_posterior'`.

- [ ] **Step 3: held-out fit/evaluation split, moment matching, MMD noise floor, correlation/kurtosis, label-mixture NLL gain을 구현한다.**

Client별 80% fitting index에서 Gaussian moment를 fit한다. 나머지 20%에서 두 독립 posterior sample `Z_emp_a`, `Z_emp_b`와 fit Gaussian sample `Z_single`을 fixed seed로 생성한다. MMD는 combined sample의 median pairwise distance bandwidth를 사용한다. Label mixture는 fitting label별 diagonal Gaussian과 empirical prior로만 fit하고 evaluation NLL을 단일 Gaussian NLL과 비교한다. 모든 metric은 CSV 행과 JSON summary에 저장한다.

- [ ] **Step 4: moment test와 synthetic multi-modal client audit를 통과시킨다.**

실행: `docker compose exec -T scful-dev python -m unittest src.tests.test_client_posterior -v`

예상: total variance formula가 맞고, 분리된 두 mode의 single-Gaussian MMD가 empirical draw noise floor보다 커진다.

### Task 6: Config-driven experiment runner와 figure 생성

**파일:**
- 생성: `src/experiments/__init__.py`
- 생성: `src/experiments/configs/oracle_mnist.json`
- 생성: `src/experiments/oracle_behavior.py`
- 생성: `src/tests/test_oracle_behavior_smoke.py`

**인터페이스:**
- CLI: `python -m src.experiments.oracle_behavior --config PATH --seed INT --output-root PATH [--smoke-test]`
- `run_experiment(config: dict[str, object], seed: int, output_root: Path, smoke_test: bool) -> Path`
- `evaluate_probe_partition(models, dataset, probes, ...) -> tuple[list[dict[str, object]], dict[str, object]]`

- [ ] **Step 1: 실패할 end-to-end smoke test를 작성한다.**

```python
def test_synthetic_smoke_run_writes_all_required_artifacts(self):
    with TemporaryDirectory() as directory:
        run_dir = run_experiment(default_config(), seed=3, output_root=Path(directory), smoke_test=True)
        required = [
            "config.json", "environment.json", "split_manifest.json", "metrics/per_sample.csv",
            "metrics/summary.json", "predictions/probes.pt", "figures/reconstruction_grid.png",
            "figures/latent_comparison.png", "checkpoints/retrain_replica.pt",
        ]
        for relative_path in required:
            self.assertTrue((run_dir / relative_path).is_file(), relative_path)
```

- [ ] **Step 2: test가 runner module 부재로 실패하는지 확인한다.**

실행: `docker compose exec -T scful-dev python -m unittest src.tests.test_oracle_behavior_smoke -v`

예상: `ModuleNotFoundError: No module named 'src.experiments.oracle_behavior'`.

- [ ] **Step 3: dataset loader, initial-state fairness, 다섯 model 학습, audit evaluation, figure 생성을 구현한다.**

Smoke mode는 download 없는 MNIST-shape synthetic data, CPU, 축소 width/dimension, 1 epoch, 4 probe/group, 2 AWGN repetition으로 default config를 override한다. Real mode는 MNIST를 config `data_root`에서 읽고 CUDA를 자동 선택한다. Evaluation은 `retrain` row의 self-gap을 0으로 기록하고 `retrain_replica` gap을 noise floor row로 남긴다.

- [ ] **Step 4: smoke test를 통과시키고 output schema를 확인한다.**

실행: `docker compose exec -T scful-dev python -m unittest src.tests.test_oracle_behavior_smoke -v`

예상: CPU synthetic run이 모든 required artifact를 만든다.

### Task 7: 전체 test suite, CUDA MNIST pilot, 결과 검증

**파일:**
- 수정: `docs/superpowers/specs/2026-08-13-controlled-noniid-oracle-behavior-design.md` (실제 config와 pilot command를 반영할 때만)

**인터페이스:**
- Real command: `python -m src.experiments.oracle_behavior --config src/experiments/configs/oracle_mnist.json --seed 0 --output-root outputs/oracle_behavior`

- [ ] **Step 1: source test suite와 compile check를 실행한다.**

실행: `docker compose exec -T scful-dev python -m unittest discover -s src/tests -v && docker compose exec -T scful-dev python -m compileall -q src/models src/training src/unlearning src/utils src/experiments`

예상: 모든 test 통과, compile error 없음.

- [ ] **Step 2: CUDA MNIST pilot을 background log로 실행한다.**

실행:

```bash
docker compose exec -T scful-dev bash -lc \
  'cd /workspace && nohup python -m src.experiments.oracle_behavior \
  --config src/experiments/configs/oracle_mnist.json --seed 0 \
  --output-root outputs/oracle_behavior > outputs/oracle_behavior/pilot_seed0.log 2>&1 &'
```

예상: GPU에서 `original`, `retrain`, `retrain_replica`, 두 SCU model을 모두 학습하고 timestamp run directory를 출력한다.

- [ ] **Step 3: 종료 후 artifact와 result invariant를 검증한다.**

실행: `find outputs/oracle_behavior -maxdepth 2 -type f | sort` 및 summary/CSV의 `dataset_split`, support group, model, evaluation condition, self-gap/retained digit-1 assertion을 검사한다.

예상: required artifact가 완비되고 `MSE_input`만으로 성공을 주장하지 않는 summary metadata와 client Gaussian diagnostic이 존재한다.

- [ ] **Step 4: pilot command, seed, output directory, noise-floor 한계를 보고한다.**

최종 보고에는 `scu_self_sharing`이 retained raw data를 직접 쓰는 centralized comparator임을 명시하고, client Gaussian diagnostic의 MMD/noise-floor/mixture 결과, 정식 결론에는 추가 global seed가 필요함을 적는다.
