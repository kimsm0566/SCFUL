# AGENTS.md

## Project Purpose

This repository is a research workspace for client-level federated unlearning in reconstruction-oriented semantic communication systems.

The core problem is:

```text
federatively trained reconstruction semantic codec
-> client-level deletion request
-> remove the target client's contribution from encoder and decoder
-> preserve retained-client reconstruction utility without centralizing raw data
```

The main expected contribution is **distributed repair** after joint codec forgetting.

## Repository Map

- `README.md`: short human-facing project overview
- `docs/literation/`: related papers, paper summaries, and comparison notes
- `docs/method/`: research idea, system model, and method design
- `docs/experiment/`: baselines, metrics, datasets, channels, and ablation plans
- `src/`: future implementation for models, federated training, unlearning, and experiments

## Execution Environment

All project code execution must run in the persistent Docker service `scful-dev`.
This includes Python scripts, tests, compilation/import checks, MPI jobs, and
experiments. Do not run project Python, `pytest`, or `mpirun` directly on the host.

```bash
# Start once when the service is stopped.
docker compose up -d scful-dev

# Non-interactive command (default for tests and experiments).
docker compose exec -T scful-dev <command>

# Interactive debugging shell.
docker compose exec scful-dev bash
```

The repository is mounted at `/workspace` inside the container. New run
instructions must show the `docker compose exec -T scful-dev ...` wrapper.

## Output Directory Names

Every new experiment output root under `outputs/` must use this format:

```text
YYYY-MM-DD-Index-brief-title
```

- `YYYY-MM-DD` is the creation date in Asia/Seoul time.
- `Index` is a two-digit sequence starting at `01` for that date. Increment it
  when creating another output root, including repeated runs of the same setup.
- `brief-title` is a concise lowercase hyphenated description.

For example:

```text
outputs/2026-08-12-01-gpu-development-container/
outputs/2026-08-12-02-gpu-development-container/
```

`RunArtifacts` may create its timestamp-and-seed directory inside this root;
the required convention applies to the user-facing `outputs/` root. Do not
reuse an existing output root for a new experiment invocation.

## User-facing Response Format

- Do not use LaTeX or MathJax syntax in responses to the user. This includes
  `$...$`, `$$...$$`, `\(...\)`, `\[...\]`, `\boxed`, `\begin`, and other
  backslash-based equation commands.
- Apply this rule even when the user writes the question or an equation in
  LaTeX. Translate it into ordinary Korean prose instead of mirroring the
  notation.
- When a compact relationship is necessary, use readable plain text such as
  `unlearned model ≈ retrained model` or
  `input -> encoder -> channel -> decoder -> reconstruction`.
- Prefer normal sentences. Use a short Markdown list or table only when it
  materially improves readability.
- This rule governs chat responses to the user. Repository research documents
  may keep mathematical notation when it is technically necessary, unless the
  user explicitly asks for plain-text documents as well.

Before changing research content, read the relevant files in this order:

1. `docs/method/idea_summary.md`
2. `docs/method/system_model.md`
3. `docs/method/unlearning_pipeline.md`
4. `docs/experiment/08-14-01-전체-실험-계획.md`
5. `docs/literation/comparison_matrix.md`

## Research Invariants

- Treat this as a **reconstruction semantic communication** project, not a standard classification-only federated unlearning project.
- Model the codec as an end-to-end encoder-channel-decoder system: `x -> f_theta -> H -> g_phi -> x_hat`.
- The unlearning target is a whole client, not only a sample, class, or risky semantic feature.
- The correct objective is retrain equivalence: `unlearned codec ~= codec retrained without target client`.
- Do not define success as simply increasing target-client reconstruction MSE.
- Preserve the distinction between:
  - initial joint encoder-decoder forgetting
  - distributed retain repair by remaining clients
  - conflict-aware aggregation at the server
- Keep distributed repair as the central novelty unless the user explicitly changes direction.

## Method Rules

- Joint forgetting must affect both encoder and decoder, unless an ablation explicitly studies encoder-only or decoder-only forgetting.
- Distributed repair must use only remaining clients' local retained data.
- Do not introduce a method that requires centralizing raw remaining-client data.
- If using distillation from the original model, apply it only on remaining-client data. Never distill target-client data from the original model.
- When designing aggregation after repair, account for the possibility that repair gradients can undo forgetting.
- Prefer conflict-aware filtering, masking, or projection over naive FedAvg when describing the full method.
- Keep physical channel effects visible in the method and experiments: AWGN, Rayleigh fading, SNR variation, and train/test SNR mismatch are part of the research scope.

## Literature Rules

- Do not claim broadly that semantic communication unlearning or federated unlearning has never been studied.
- Use the narrower novelty claim:

```text
Client-level federated unlearning for jointly trained reconstruction-oriented semantic codecs,
with distributed repair using only remaining clients' local data,
has not been directly addressed by the currently compared works.
```

- When adding literature notes, place them under `docs/literation/summaries/`.
- Use one summary file per paper, named with a short paper key, for example `scu.md`, `ssfu.md`, or `seful.md`.
- Do not edit PDFs in `docs/literation/papers/`.
- When uncertain about a paper's contribution, check the PDF before updating comparison claims.

## Experiment Rules

### Required Headline Metrics

- The current metric contract and implementation/verification status are recorded in
  `docs/experiment/10-22-01-dfil-core-metrics.md`. Keep README and the method/experiment
  summaries consistent with it; historical metric reports are not current requirements.
- Distinguish code implemented, tests executed, and real checkpoint evaluation completed.
  Never turn blocked Docker verification into a claim of passing tests or completed results.

- Every new reconstruction federated-unlearning experiment must report the same headline
  metric set for the original, retrained-without-target, and unlearned checkpoints:
  - `MPR`: absolute marker-positive decoded-output percentage using the frozen detector;
    report positive counts and held-out sample size, even when PUL is undefined
  - `PUL`: percentage reduction in marker-positive decoded outputs relative to Original,
    using one frozen detector calibrated without codec outputs
  - `FID`: frozen pytorch-fid Inception pool3 (2048-D) distance from remaining-client
    reconstructions to real clean remaining-client inputs
  - `PSNR`: reconstruction-to-input PSNR with peak value 1
  - `SSIM`: reconstruction-to-input RGB SSIM
  - `Runtime`: measured retraining/online-unlearning wall time, excluding offline evaluation
- All user-facing comparisons, CSVs, and figures use Original, Retrain-without-target,
  then Unlearned/FSCUv4 order. Always include all three models for clean PSNR/SSIM,
  erased PSNR/SSIM, and retain FID. Reference-only preparation may show only the two
  available reference models; do not fabricate an unlearned checkpoint.
- dFIL is supplementary, not an encoder-deletion pass/fail requirement. The current audit
  still automatically saves clean/retain/erased dFIL, trace Monte Carlo error, sample indices,
  observation protocol, and its separate figure for compatibility.
- MPR applies to S1 and S2. It is marker occurrence, not SCU downstream-classifier BA.
  For S1 also report continuous marker projection amplitude and centered NCC for all
  three models. Preserve negative scores and separate response-pool N from detection N.
  Local scores inspect only the fixed marker region. S1 additionally runs the registered
  whole-image template-bank audit: global/outside amplitude and NCC, global MPR, and
  pixel-union coverage. Save raw-only calibration/control FPR/TPR, indices, maps and figures.
  Never tune thresholds on codec outputs or label outside-region RMSE/color residual as
  marker-specific leakage. State the bank's limits for diffuse tint and unseen patterns.
- IDI is superseded by dFIL. dFIL is input reconstructability, not a membership
  test or proof of client deletion. Do not label its reciprocal as an arbitrary-attacker
  MSE guarantee without the additional prior Fisher assumptions.
- Do not use CKA as a core metric. Client-ID probes, custom latent/output sensitivity,
  and FID/PSNR/SSIM-to-Retrain may remain only as legacy diagnostics. The explicitly
  requested continuous S1 marker response is reported alongside MPR, not as deletion proof.
- Report reconstruction results separately on the clean test set and the target-client
  erased set. Keep normalized MSE and RMSE-255 as internal or supplementary diagnostics.
- A run is not a complete reported experiment until
  `src/experiments/reconstruction_channel_audit.py` has produced
  `metrics/selected_core_metrics.csv`, `metrics/selected_core_metrics_protocol.json`,
  and `figures/selected_core_metrics.png`. It must also retain per-input dFIL and its protocol,
  PUL detector protocol, and full-distribution FID protocol artifacts. Never average
  minibatch FIDs.
  Smoke tests may exercise metric plumbing, but their values are not paper results.

### Required Visual Artifacts

- New independent semantic observers must be initialized from scratch and trained
  by classifier-only FedAvg on remaining clients' clean local data. Exclude the
  target client from both training and local validation. Select by sample-weighted
  local validation CE, never by the official test set. Reuse one frozen observer
  for Original, Retrain, and Unlearned; observer CE must not update the codec.
- Output agreement audits additionally save clean/erased PSNR-to-Retrain,
  SSIM-to-Retrain, class agreement, probability JS (nats), and logit MSE, with
  per-input rows, logits, protocol metadata, and comparison figures. Do not
  confuse image-quality score matching with output-to-output similarity.

- Every experiment invocation, including pilots, smoke tests, ablations, training runs,
  and unlearning audits, must automatically create task-appropriate figures under
  `figures/`. Smoke-test figures validate the artifact path only and are not paper
  results. A run with missing required figures is incomplete and must not be reported as
  a completed experiment.
- Figures must be generated by the same `src/main.py` invocation that produces the
  metrics. Do not make manual plotting a separate step required to finish a run.
- Every plotted point must be recoverable from CSV or JSON stored under `metrics/` in
  the same run. Keep the raw unsmoothed series; smoothing may appear only as an optional,
  labeled overlay with its window recorded.
- Use global round or optimization step on the x-axis. Label units, split, channel/SNR,
  and train versus held-out/test provenance explicitly. Mark the final round and any
  selected checkpoint, and record the checkpoint-selection rule without using the final
  official test set for opportunistic round selection.
- Reconstruction training must plot:
  - total objective loss and its reconstruction and beta-weighted KL components
  - training reconstruction quality and periodic held-out RMSE-255/PSNR
  - semantic accuracy only when a frozen external observer is explicitly part of the
    evaluation; observer CE must never update the codec
- Independent classifier training must plot train and held-out CE and accuracy. Do not
  add PSNR, MSE, reconstruction loss, or codec metrics to classifier-only runs.
- Federated-unlearning runs must plot Original, Retrain-without-target, and Unlearned
  side by side using separate panels/scales for:
  - clean and erased PSNR and SSIM
  - retain FID to real retain inputs
  - marker MPR and PUL when a marker protocol is defined
  - continuous marker amplitude/NCC, especially for S1
  - supplementary encoder dFIL on clean/retain/erased inputs in its own figure
  - retraining and online-unlearning Runtime
  - stage-wise forgetting/repair loss curves when the method performs iterative
    optimization
- When comparing methods, seeds, IID/non-IID splits, or SNRs, preserve model ordering,
  axis definitions, and scales across figures wherever doing so does not hide the data.
  Data-partition experiments must also visualize per-client class counts or proportions.
- Multi-seed aggregate figures must show the individual seed points together with the
  mean and a clearly identified uncertainty measure; do not show only an aggregate bar.
- Round-budget claims such as whether 800 rounds are sufficient require periodic
  held-out curves and a stated convergence rule. Final-checkpoint metrics alone, or
  training metrics mislabeled as held-out metrics, are insufficient evidence.

### Reference Checkpoint Reuse

- Train the `initial`, `original`, and `retrain_without_target` reference states once per
  fully specified training scenario, then reuse them across unlearning methods and
  unlearning-hyperparameter runs.
- Never decide reuse from model name and global seed alone. Reuse is allowed only when a
  scenario fingerprint matches all of the following:
  - complete codec configuration and initial-state seed/hash
  - dataset name/version and exact client split manifest
  - public-data reservation indices and training-pool indices
  - split type and parameters, client count, samples per client, and target client ID
  - backdoor protocol, poisoned indices, poison seed, and semantic-loss settings
  - FL rounds, local epochs, batch size, optimizer, learning rate, and aggregation rule
  - training channel, SNR, deterministic setting, and all other training-affecting options
- Unlearning-method and unlearning-hyperparameter values are not part of the reference
  fingerprint. They belong to the child unlearning run.
- Reference artifacts are immutable. Store a canonical serialized fingerprint,
  `split_manifest.json`, training metrics/runtime, environment metadata, and SHA-256 hashes
  for `initial.pt`, `original.pt`, and `retrain_without_client_<id>.pt`.
- A reused run must write `reference_source.json` containing the reference artifact path,
  fingerprint, and checkpoint hashes. It must verify the fingerprint and hashes before
  loading; on any mismatch, train a new reference instead of reusing the old one.
- Load the cached original and retrained states for evaluation as well as unlearning. Do
  not copy them into every child run unless portability is explicitly required.
- Runtime comparisons use the stored retraining wall time from the reference artifact and
  the newly measured online unlearning wall time. Record checkpoint-loading time
  separately and exclude it from both algorithm runtimes.
- A different seed, client split, target client, backdoor realization, public-data
  reservation, codec/training configuration, or channel-training condition requires a
  separate reference artifact. In particular, 2-class and Dirichlet 0.1 runs never share
  a reference checkpoint.

### Experiment Document Names

- `docs/experiment/`에 새로 만드는 실험 문서는 반드시
  `MM-DD-Index-간단한제목.md` 형식을 쓴다.
- `MM-DD`는 한국 시간 기준 작성일이며, `Index`는 같은 날짜 안에서 `01`부터
  증가하는 두 자리 순번이다.
- 기존 문서의 이름을 바꾸면 Markdown link와 `AGENTS.md`의 참조도 함께 갱신한다.

- Always compare against the full retraining baseline without the target client.
- Include both utility and unlearning metrics.
- Utility metrics should prioritize PSNR, SSIM, and retain FID.
- Encoder latent privacy uses dFIL; decoded marker removal uses PUL.
- Backdoor or client marker experiments are useful diagnostics, but they are not sufficient alone.
- Include ablations for:
  - encoder-only unlearning
  - decoder-only unlearning
  - joint forgetting only
  - joint forgetting plus naive repair
  - joint forgetting plus distributed repair
  - full method with conflict-aware aggregation
- Evaluate IID and non-IID client splits.
- Evaluate multiple channel conditions and SNR values.

## Implementation Rules

- Use `src/main.py` as the only experiment execution entry point. Experiment,
  training, evaluation, and unlearning modules expose functions but must not add
  their own command-line `main` blocks.
- CIFAR-10 is the only supported dataset. The only project-owned semantic codec
  is `VariationalDeepJSCC`; do not add parallel MNIST, compact-CNN, or SCU
  Table-2 codec implementations.
- Every federated training round must print and append task-appropriate metrics to
  `run.log`. Reconstruction runs record training MSE/PSNR, objective loss,
  reconstruction loss, and KL loss; semantic accuracy/loss is present only when a
  frozen external observer is explicitly evaluated. Classifier-only runs record CE and
  accuracy and must not emit reconstruction metrics. Label periodic held-out evaluation
  values separately as `test_*`; do not silently present training metrics as test
  metrics.
- Do not use decorator syntax in project-owned `src/` Python code. Use explicit
  constructors, ordinary instance methods, standalone factory functions, and
  `with torch.no_grad()` blocks instead.
- Every semantic codec model must expose `forward`, and training and unlearning
  code must call the model through `model(...)` rather than directly selecting
  codec path methods.
- `FSCU` always names the target-client joint-unlearning plus server public-data
  contrastive-compensation method. `FSCUv4` names joint forgetting followed by
  retained-client repair and client-wise conflict-aware projection aggregation.
- One `FSCU` invocation must run unlearning and then automatically write the full
  registered core suite: MPR, PUL, FID, PSNR, SSIM, and Runtime, plus the currently
  retained supplementary dFIL and continuous marker response artifacts.
  Separate audit entry points are optional tools for old checkpoints, not steps in
  the normal FSCU workflow.
- Put experimental code under `src/`.
- Keep model definitions in `src/models/`.
- Do not recreate `src/training/` or `src/unlearning/`. The FSCU server/client
  orchestration belongs directly in `src/utils/trainer.py`.
- Keep reusable local federated updates and FedAvg in `src/utils/federated.py`,
  FSCU optimization stages in `src/utils/fscu.py`, and pure loss functions in
  `src/utils/loss.py`.
- Keep every dataset loader, split, subset, and probe constructor in
  `src/data/data.py`; do not spread data functions across additional data files.
- Use the real torchvision CIFAR-10 dataset in both experiments and smoke tests.
  Smoke tests may use deterministic subsets of real CIFAR-10, but must not replace
  it with generated or synthetic images.
- Delete obsolete branches instead of retaining pass-through wrappers or dormant
  prototype implementations. Do not introduce a generic pipeline function that
  hides the FSCU stages from `trainer.py`.
- Keep experiment entry points in `src/experiments/`.
- Keep shared dataset, channel, logging, and metric helpers in `src/utils/`.
- Keep reusable non-interactive figure generation in `src/utils/plotting.py`. Plotting
  tests must verify the expected filenames and that each figure is backed by non-empty
  metric rows.
- Prefer configuration-driven experiments over hard-coded constants.
- Record random seeds, client splits, target-client IDs, channel settings, and metric outputs for each run.
- Pass a newly named `outputs/YYYY-MM-DD-Index-brief-title` directory through
  `--output-root` for every experiment invocation.
- Keep early implementations small and reproducible before adding complex variants.

## Verification

- For documentation-only changes, verify the file tree with `find . -maxdepth 3 -type f | sort`.
- For Python code changes, run the smallest relevant smoke test first.
- If no test suite exists yet, at minimum run import or compile checks for touched Python modules.
- Execute every verification command through `docker compose exec -T scful-dev`.
- Do not report an experiment result unless the command, config, seed, and output location are clear.

## Common Pitfalls

- Do not collapse this project into a classification federated unlearning setup.
- Do not evaluate unlearning success only by poor reconstruction on the target client.
- Do not reintroduce target-client knowledge through teacher distillation.
- Do not describe the method as merely "SCU + FedAvg"; the intended contribution is distributed repair under raw-data constraints.
- Do not let `docs/method/background.md` become the only source of truth. Keep focused summaries in the smaller method and experiment files.
