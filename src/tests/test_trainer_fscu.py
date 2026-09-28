import csv
from pathlib import Path
from types import SimpleNamespace

import src.utils.trainer as trainer_module


def test_trainer_routes_fscu_to_the_fscu_pipeline(monkeypatch):
    expected = {"algorithm": "FSCU"}

    def fake_train_fscu(comm, data_state, args, logger):
        return expected

    monkeypatch.setattr(trainer_module, "train_fscu", fake_train_fscu)
    args = SimpleNamespace(algorithm="FSCU")

    result = trainer_module.Trainer(args).train(
        data_state={},
        comm=object(),
        logger=None,
        args=args,
    )

    assert result is expected


def test_fscu_output_root_requires_registered_name_format():
    valid = Path("outputs/2026-09-07-01-fscu")

    assert trainer_module._validate_fscu_output_root(valid) == valid

    try:
        trainer_module._validate_fscu_output_root(Path("outputs/fscu"))
    except ValueError as error:
        assert "YYYY-MM-DD-Index-title" in str(error)
    else:
        raise AssertionError("invalid FSCU output root was accepted")


def test_fscu_headline_reader_requires_all_registered_metrics(tmp_path):
    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir()
    metrics_path = metrics_dir / "headline_metrics.csv"
    fields = ("model",) + trainer_module.FSCU_REQUIRED_METRICS
    row = {field: "1.0" for field in fields}
    row["model"] = "FSCU"
    with metrics_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(row)

    loaded_path, loaded_row = trainer_module._read_fscu_headline_metrics(tmp_path)

    assert loaded_path == metrics_path
    assert loaded_row["model"] == "FSCU"
