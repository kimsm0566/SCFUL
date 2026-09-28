"""Oracle behavior run의 non-overwriting artifact 저장소."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if hasattr(value, "__dict__"):
        return dict(vars(value))
    raise TypeError(f"cannot serialize {type(value).__name__} to JSON")


class RunArtifacts:
    """단일 experiment run의 파일 경로와 atomic writer를 제공한다."""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir

    def path(self, relative_path: str) -> Path:
        path = self.run_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def append_log(self, message: str) -> Path:
        log_path = self.path("run.log")
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"{message}\n")
        return log_path

    def write_json(self, relative_path: str, payload: Any) -> Path:
        destination = self.path(relative_path)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default)
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(destination)
        return destination

    def write_records(self, records: list[dict[str, Any]]) -> Path:
        return self.write_csv("metrics/per_sample.csv", records)

    def write_csv(self, relative_path: str, records: list[dict[str, Any]]) -> Path:
        destination = self.path(relative_path)
        if not records:
            raise ValueError("cannot write an empty CSV record collection")
        fieldnames = list(records[0])
        for record in records[1:]:
            for fieldname in record:
                if fieldname not in fieldnames:
                    fieldnames.append(fieldname)
        with destination.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)
        return destination

    def save_checkpoint(self, name: str, state: dict[str, torch.Tensor]) -> Path:
        destination = self.path(f"checkpoints/{name}.pt")
        torch.save(state, destination)
        return destination

    def save_predictions(self, payload: Any) -> Path:
        destination = self.path("predictions/probes.pt")
        torch.save(payload, destination)
        return destination


def create_run_artifacts(output_root: Path, seed: int) -> RunArtifacts:
    """충돌하지 않는 새 run directory와 표준 하위 폴더를 만든다."""
    output_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base_name = f"{timestamp}_seed-{seed}"
    suffix = 0
    while True:
        candidate = output_root / (
            base_name if suffix == 0 else f"{base_name}_{suffix:02d}"
        )
        try:
            candidate.mkdir()
            break
        except FileExistsError:
            suffix += 1
    for name in ("checkpoints", "metrics", "predictions", "figures", "client_posterior"):
        (candidate / name).mkdir()
    (candidate / "run.log").touch()
    return RunArtifacts(candidate)
