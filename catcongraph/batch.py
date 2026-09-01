from __future__ import annotations

import csv
import json
import re
import traceback
from pathlib import Path
from typing import Any, Optional

from catcongraph.preflight import run_preflight
from catcongraph.workflow import run_workflow

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover
    yaml = None


def _read_manifest(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
    else:
        if yaml is None:
            raise RuntimeError("PyYAML is required for YAML batch manifests; use CSV instead.")
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        rows = data.get("projects", data if isinstance(data, list) else [])
    out = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        enabled = str(row.get("enabled", "true")).strip().lower() not in {"0", "false", "no", "off"}
        if enabled and row.get("config"):
            out.append(dict(row))
    return out


def _resolve_config(value: object, manifest_path: Path) -> Path:
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (manifest_path.parent / path).resolve()


def _safe_token(value: object) -> str:
    token = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_.")
    return token or "project"


def run_batch(
    manifest_path: str | Path,
    *,
    stage: str = "all",
    max_structures: Optional[int] = None,
    clean: bool = False,
    preflight_only: bool = False,
    continue_on_error: bool = True,
) -> dict[str, Any]:
    manifest_path = Path(manifest_path).resolve()
    rows = _read_manifest(manifest_path)
    output_dir = manifest_path.parent / f"{manifest_path.stem}_batch_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    for index, row in enumerate(rows, start=1):
        config_path = _resolve_config(row["config"], manifest_path)
        target_hint = str(row.get("target_id") or config_path.parent.name)
        result = {
            "manifest_order": index,
            "target_id": target_hint,
            "config": str(config_path),
            "preflight_status": "not_run",
            "workflow_status": "not_run",
            "error": "",
        }
        for key, value in row.items():
            if key not in {"enabled", "target_id", "config"}:
                result[f"meta_{key}"] = value
        try:
            preflight = run_preflight(config_path, max_structures=max_structures)
            result["target_id"] = preflight["target_id"]
            result["preflight_status"] = preflight["status"]
            result["preflight_report"] = preflight["report_path"]
            if preflight["errors"]:
                raise RuntimeError("; ".join(preflight["errors"]))
            if not preflight_only:
                run_workflow(
                    config=str(config_path),
                    stage=stage,
                    max_structures=max_structures,
                    clean=clean,
                )
                result["workflow_status"] = "completed"
        except Exception as exc:
            result["workflow_status"] = "failed"
            result["error"] = str(exc)
            failure_path = output_dir / f"{index:04d}_{_safe_token(target_hint)}_failure.txt"
            failure_path.write_text(traceback.format_exc(), encoding="utf-8")
            result["failure_traceback"] = str(failure_path)
            summary_rows.append(result)
            if not continue_on_error:
                break
            continue
        summary_rows.append(result)

    csv_path = output_dir / "batch_summary.csv"
    fieldnames = sorted({key for row in summary_rows for key in row})
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    report = {
        "manifest": str(manifest_path),
        "output_dir": str(output_dir),
        "n_projects": len(summary_rows),
        "n_completed": sum(x["workflow_status"] in {"completed", "not_run"} and not x["error"] for x in summary_rows),
        "n_failed": sum(bool(x["error"]) for x in summary_rows),
        "summary_csv": str(csv_path),
        "projects": summary_rows,
    }
    json_path = output_dir / "batch_report.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    report["report_json"] = str(json_path)
    return report
