import pytest

from src.evaluation.oracle_behavior import _build_codec as build_oracle_codec
from src.experiments.membership_inference_audit import (
    _make_codec as make_membership_audit_codec,
)
from src.experiments.reconstruction_channel_audit import (
    _make_codec as make_reconstruction_audit_codec,
)
from src.models.semantic_codec import (
    CURRENT_DEEPJSCC_CODEC_FORMAT_VERSION,
    DEEPJSCC_ARCHITECTURE,
    LEGACY_DEEPJSCC_ARCHITECTURE,
    LEGACY_DEEPJSCC_CODEC_FORMAT_VERSION,
    VariationalDeepJSCC,
)


def _build_with_each_loader(architecture):
    codec_values = {"architecture": architecture}
    return (
        make_reconstruction_audit_codec({"codec": codec_values}),
        make_membership_audit_codec({"codec": codec_values}),
        build_oracle_codec("CIFAR10", codec_values),
    )


def test_checkpoint_loaders_accept_current_and_legacy_deepjscc():
    cases = (
        (DEEPJSCC_ARCHITECTURE, CURRENT_DEEPJSCC_CODEC_FORMAT_VERSION),
        (LEGACY_DEEPJSCC_ARCHITECTURE, LEGACY_DEEPJSCC_CODEC_FORMAT_VERSION),
    )

    for architecture, expected_version in cases:
        for model in _build_with_each_loader(architecture):
            assert isinstance(model, VariationalDeepJSCC)
            assert model.config.codec_format_version == expected_version


def test_checkpoint_loaders_reject_removed_codec_architectures():
    removed_architecture = "compact_cnn_cifar10_v1"
    builders = (
        lambda: make_reconstruction_audit_codec(
            {"codec": {"architecture": removed_architecture}}
        ),
        lambda: make_membership_audit_codec(
            {"codec": {"architecture": removed_architecture}}
        ),
        lambda: build_oracle_codec(
            "CIFAR10", {"architecture": removed_architecture}
        ),
    )

    for build in builders:
        with pytest.raises(ValueError, match="unsupported"):
            build()
