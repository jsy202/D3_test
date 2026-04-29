from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean, median
from typing import Any


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    latencies = [float(row["latency_ms"]) for row in rows]
    if not latencies:
        return {"frames": 0}
    sorted_latencies = sorted(latencies)
    p95_index = min(len(sorted_latencies) - 1, int(round((len(sorted_latencies) - 1) * 0.95)))
    avg = mean(latencies)
    return {
        "frames": len(latencies),
        "latency_ms_avg": avg,
        "latency_ms_median": median(latencies),
        "latency_ms_min": min(latencies),
        "latency_ms_max": max(latencies),
        "latency_ms_p95": sorted_latencies[p95_index],
        "fps_from_latency_avg": 1000.0 / avg if avg > 0 else None,
    }
