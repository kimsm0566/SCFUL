"""SCU-compatible semantic backdoor evaluation for federated codec states."""

from __future__ import annotations

import torch


def _normalized_images(data):
    images, labels = data
    images = torch.as_tensor(images, dtype=torch.float32)
    labels = torch.as_tensor(labels, dtype=torch.long)
    if images.max() > 1.0:
        images = images / 255.0
    if len(images) != len(labels):
        raise ValueError("backdoor evaluation images and labels must have equal length")
    return images, labels


def _semantic_predictions(
    models,
    semantic_classifier,
    data,
    batch_size,
    snr_db,
    device,
    generator,
):
    images, labels = _normalized_images(data)
    predictions = {name: [] for name in models}
    label_chunks = []
    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            batch = images[start : start + batch_size].to(device)
            batch_labels = labels[start : start + batch_size]
            posterior_epsilon = torch.randn(
                (len(batch), next(iter(models.values())).config.latent_dim),
                device=device,
                generator=generator,
            )
            channel_epsilon = torch.randn(
                posterior_epsilon.shape,
                device=device,
                generator=generator,
            )
            for name, model in models.items():
                posterior = model.posterior(batch)
                latent = posterior.mu + posterior.logvar.mul(0.5).exp() * posterior_epsilon
                received = model.transmit_with_standard_awgn(
                    latent, channel_epsilon, snr_db
                )
                reconstruction = model.decode_latent(received)
                prediction = semantic_classifier(reconstruction).argmax(dim=1)
                predictions[name].append(prediction.cpu())
            label_chunks.append(batch_labels)
    return {
        name: torch.cat(chunks) for name, chunks in predictions.items()
    }, torch.cat(label_chunks)


def evaluate_scu_backdoor_states(
    model_factory,
    model_states,
    semantic_classifier,
    backdoor_probe,
    clean_test_data,
    batch_size,
    snr_db,
    device,
    seed,
):
    """Report each model's SCU ASR and clean semantic accuracy.

    All models see the same posterior standard-normal draws and the same AWGN
    draws for each sample.  ASR is the accuracy against the trigger label on
    patched base-class inputs.
    """

    if batch_size <= 0:
        raise ValueError("backdoor evaluation batch_size must be positive")
    device = torch.device(device)
    semantic_classifier = semantic_classifier.to(device).freeze()
    models = {}
    for name, state in model_states.items():
        model = model_factory().to(device)
        model.load_state_dict(state)
        model.eval()
        models[name] = model

    generator_device = "cuda" if device.type == "cuda" else "cpu"
    backdoor_generator = torch.Generator(device=generator_device).manual_seed(seed)
    clean_generator = torch.Generator(device=generator_device).manual_seed(seed + 1)
    backdoor_predictions, trigger_labels = _semantic_predictions(
        models,
        semantic_classifier,
        backdoor_probe,
        batch_size,
        snr_db,
        device,
        backdoor_generator,
    )
    clean_predictions, clean_labels = _semantic_predictions(
        models,
        semantic_classifier,
        clean_test_data,
        batch_size,
        snr_db,
        device,
        clean_generator,
    )
    return [
        {
            "model": name,
            "backdoor_asr": float(
                (backdoor_predictions[name] == trigger_labels).float().mean()
            ),
            "backdoor_examples": int(len(trigger_labels)),
            "clean_semantic_accuracy": float(
                (clean_predictions[name] == clean_labels).float().mean()
            ),
            "clean_examples": int(len(clean_labels)),
            "channel": "paired_awgn",
            "snr_db": float(snr_db),
        }
        for name in models
    ]


def evaluate_clean_semantic_states(
    model_factory,
    model_states,
    semantic_classifier,
    clean_test_data,
    batch_size,
    snr_db,
    device,
    seed,
):
    """Evaluate decoded-image accuracy with one frozen clean observer."""

    if batch_size <= 0:
        raise ValueError("semantic evaluation batch_size must be positive")
    device = torch.device(device)
    semantic_classifier = semantic_classifier.to(device).freeze()
    models = {}
    for name, state in model_states.items():
        model = model_factory().to(device)
        model.load_state_dict(state)
        model.eval()
        models[name] = model
    generator_device = "cuda" if device.type == "cuda" else "cpu"
    generator = torch.Generator(device=generator_device).manual_seed(seed)
    predictions, labels = _semantic_predictions(
        models,
        semantic_classifier,
        clean_test_data,
        batch_size,
        snr_db,
        device,
        generator,
    )
    return [
        {
            "model": name,
            "clean_semantic_accuracy": float(
                (predictions[name] == labels).float().mean()
            ),
            "clean_examples": int(len(labels)),
            "channel": "paired_awgn",
            "snr_db": float(snr_db),
            "classifier_role": "shared_frozen_clean_observer",
        }
        for name in models
    ]
