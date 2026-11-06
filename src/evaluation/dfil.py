"""Input Fisher leakage of the codec's pre-normalization Gaussian posterior.

Maeng et al., NeurIPS 2023, eq. (3): dFIL = trace(I_input) / input_dim.
For an input-dependent diagonal Gaussian this trace includes BOTH the mean
Jacobian weighted by inverse variance and half the log-variance Jacobian norm.
Randomized VJPs estimate these Frobenius norms without materializing Jacobians.
This is not parameter Fisher, mutual information, or a deletion certificate.
"""

import math
import time

import numpy as np
import torch

from src.utils.reproducibility import derive_seed


def posterior_dfil(model, inputs, probes=32, seed=0):
    """Return per-input trace estimates and their Monte Carlo standard errors.

The caller supplies an eval-mode, sample-independent codec. Gradients are taken
only with respect to inputs; parameter .grad buffers are never populated.
Rademacher probes are generated on CPU for reproducibility across devices.
"""
    if not isinstance(probes, int) or isinstance(probes, bool) or probes < 2:
        raise ValueError("dFIL probes must be an integer >= 2")
    if model.training:
        raise ValueError("dFIL requires an eval-mode codec")
    if inputs.ndim < 2 or inputs.shape[0] == 0:
        raise ValueError("dFIL needs a nonempty batch of inputs")
    if not inputs.is_floating_point() or not torch.isfinite(inputs).all():
        raise ValueError("dFIL inputs must be finite floating point values")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    with torch.enable_grad():
        x = inputs.detach().clone().requires_grad_(True)
        posterior = model(x, mode="posterior")
        mu, logvar = posterior.mu, posterior.logvar
        if mu.shape != logvar.shape or mu.ndim != 2 or mu.shape[0] != len(x):
            raise ValueError("dFIL requires matching batch-by-latent Gaussian parameters")
        inverse_std = torch.exp(-0.5 * logvar.detach())
        if not all(torch.isfinite(t).all() for t in (mu, logvar, inverse_std)):
            raise ValueError("nonfinite Gaussian posterior in dFIL")
        estimates = []
        for _ in range(probes):
            terms = []
            for values, scale in ((mu, inverse_std), (logvar, math.sqrt(0.5))):
                signs = torch.randint(0, 2, values.shape, generator=generator)
                signs = signs.to(device=values.device, dtype=values.dtype) * 2 - 1
                if values.requires_grad:
                    gradient = torch.autograd.grad(
                        values, x, grad_outputs=signs * scale,
                        retain_graph=True, allow_unused=True,
                    )[0]
                else:
                    gradient = None
                terms.append(
                    torch.zeros(len(x), device=x.device, dtype=torch.float64)
                    if gradient is None
                    else gradient.detach().double().flatten(1).square().mean(1)
                )
            estimates.append(terms[0] + terms[1])
        estimates = torch.stack(estimates)
    if not torch.isfinite(estimates).all():
        raise ValueError("nonfinite dFIL estimate")
    return {
        "dfil": estimates.mean(0).cpu(),
        "mc_standard_error": (estimates.std(0, unbiased=True) / math.sqrt(probes)).cpu(),
        "input_dimensions": inputs[0].numel(),
        "latent_dimensions": mu.shape[1],
    }


def evaluate_encoder_dfil(model_factory, model_states, split_data, device="cpu",
                          probes=32, max_examples=128, seed=0, smoke_test=False):
    """Evaluate paired, uniformly sampled real inputs for every checkpoint.

split_data maps split names to (images, dataset_indices). Sampling and trace
probes are paired across states. One input per forward makes results independent
of evaluation batching and avoids cross-sample Jacobian terms.
"""
    if not model_states or not split_data:
        raise ValueError("dFIL requires model states and data splits")
    if not isinstance(max_examples, int) or isinstance(max_examples, bool) or max_examples < 0:
        raise ValueError("dfil_max_examples must be a nonnegative integer (0 means all)")
    if not isinstance(probes, int) or isinstance(probes, bool) or probes < 2:
        raise ValueError("dfil_probes must be an integer >= 2")
    started = time.perf_counter()
    per_sample, summary, selections = [], [], {}
    for split, (images, indices) in split_data.items():
        if len(images) == 0 or len(images) != len(indices):
            raise ValueError("dFIL split images and dataset indices must be nonempty and aligned")
        rng = np.random.default_rng(derive_seed(seed, f"dfil-selection:{split}"))
        count = len(images) if max_examples == 0 else min(max_examples, len(images))
        selections[split] = sorted(rng.choice(len(images), count, replace=False).tolist())
    for name, state in model_states.items():
        model = model_factory().to(device)
        model.load_state_dict(state)
        model.eval()
        for split, (images, indices) in split_data.items():
            rows = []
            for position in selections[split]:
                x = torch.as_tensor(images[position:position + 1], device=device)
                if x.dtype == torch.uint8:
                    x = x.float() / 255.0
                else:
                    x = x.float()
                if x.min() < 0 or x.max() > 1:
                    raise ValueError("dFIL requires RGB inputs in [0,1]; scale changes the metric")
                probe_seed = derive_seed(seed, f"dfil-probes:{split}:{int(indices[position])}")
                result = posterior_dfil(model, x, probes=probes, seed=probe_seed)
                row = {
                    "model": name, "split": split, "input_position": position,
                    "dataset_index": int(indices[position]),
                    "dfil": float(result["dfil"][0]),
                    "mc_standard_error": float(result["mc_standard_error"][0]),
                    "probes": probes, "probe_seed": probe_seed,
                    "input_dimensions": result["input_dimensions"],
                    "latent_dimensions": result["latent_dimensions"],
                }
                rows.append(row)
                per_sample.append(row)
            values = np.array([row["dfil"] for row in rows])
            summary.append({
                "model": name, "split": split, "dfil": float(values.mean()),
                "dfil_sample_std": float(values.std(ddof=1)) if len(values) > 1 else None,
                "mean_mc_standard_error": float(np.sqrt(sum(
                    row["mc_standard_error"] ** 2 for row in rows)) / len(rows)),
                "samples": len(rows), "available_samples": len(images),
                "status": "smoke_only" if smoke_test else "estimated",
            })
        del model
    return {"summary": summary, "per_sample": per_sample, "protocol": {
        "metric": "dFIL = trace(input Fisher information) / input dimensions",
        "source": "https://proceedings.neurips.cc/paper_files/paper/2023/file/a344f7f474958cc0775be7e46bc94309-Paper-Conference.pdf",
        "observation": "diagonal Gaussian posterior sample BEFORE power normalization and physical channel",
        "gaussian_trace": "sum_j ||grad_x mu_j||^2 / variance_j + 0.5 * sum_j ||grad_x logvar_j||^2",
        "estimator": "paired Rademacher VJP trace estimation; both Gaussian mean and scale derivatives",
        "input_range": "RGB [0,1]", "input_units": "inverse squared normalized pixel units",
        "channel": "pre_channel", "snr_db": None,
        "probes": probes, "max_examples_per_split": max_examples, "seed": int(seed),
        "sampling": "uniform without replacement from supplied splits; identical indices across models",
        "smoke_test": bool(smoke_test), "wall_seconds": time.perf_counter() - started,
        "runtime_scope": "offline metric evaluation; excluded from algorithm runtime",
        "interpretation": "input reconstructability diagnostic; not training membership or proof of deletion",
        "bound_note": "1/dFIL alone only bounds unbiased reconstruction under regularity assumptions; no arbitrary-attacker bound reported without prior Fisher",
    }}
