from __future__ import annotations

import math
from datetime import UTC, datetime
from statistics import fmean, pstdev
from typing import Any

from pymongo.asynchronous.database import AsyncDatabase

from app.core.errors import AppError
from app.core.security import Principal
from app.modules.inspections.schemas import DeviceInspectionManifest, InspectionPlanningManifest
from app.modules.projects.data import query_data
from app.modules.projects.schemas import DataQuery

Document = dict[str, Any]


def _numbers(value: Any, result: dict[str, list[float]], prefix: str = "") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            _numbers(item, result, name)
        return
    if isinstance(value, list):
        for item in value:
            _numbers(item, result, prefix)
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return
    result.setdefault(prefix or "value", []).append(float(value))


def _timestamps(value: Any, result: list[datetime]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(item, str) and any(
                word in key.casefold() for word in ("time", "timestamp", "date")
            ):
                try:
                    parsed = datetime.fromisoformat(item.replace("Z", "+00:00"))
                    result.append(parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed)
                except ValueError:
                    pass
            else:
                _timestamps(item, result)
    elif isinstance(value, list):
        for item in value:
            _timestamps(item, result)


def summarize_payload(payload: Any, window_end: datetime) -> Document:
    series: dict[str, list[float]] = {}
    timestamps: list[datetime] = []
    _numbers(payload, series)
    _timestamps(payload, timestamps)
    metrics: dict[str, Document] = {}
    constant: list[str] = []
    for name, values in sorted(series.items()):
        deviation = pstdev(values) if len(values) > 1 else 0.0
        metrics[name] = {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "mean": fmean(values),
            "stddev": deviation,
        }
        if len(values) >= 3 and deviation <= max(abs(fmean(values)) * 1e-6, 1e-9):
            constant.append(name)
    latest = max(timestamps) if timestamps else None
    stale_seconds = max(0, int((window_end - latest).total_seconds())) if latest else None
    return {
        "sample_count": max((len(values) for values in series.values()), default=0),
        "numeric_series_count": len(series),
        "latest_timestamp": latest.isoformat() if latest else None,
        "stale_seconds": stale_seconds,
        "constant_series": constant,
        "metrics": metrics,
    }


def _structural_candidates(
    snapshot: Document, manifest: DeviceInspectionManifest
) -> list[Document]:
    candidates: list[Document] = []
    raw = next(
        (
            node
            for node in snapshot.get("nodes", [])
            if isinstance(node, dict) and str(node.get("id")) == manifest.node_id
        ),
        {},
    )
    raw_data = raw.get("data")
    data: Document = raw_data if isinstance(raw_data, dict) else {}
    if not manifest.related_node_ids:
        candidates.append(
            {
                "code": "isolated_equipment",
                "category": "topology",
                "severity": "warning",
                "detail": "The device has no RDF feed relationship to another target device.",
            }
        )
    raw_sensors = data.get("sensors")
    sensors: list[Any] = raw_sensors if isinstance(raw_sensors, list) else []
    incomplete = [
        item
        for item in sensors
        if not isinstance(item, dict)
        or not str(item.get("name") or "").strip()
        or not str(item.get("category") or "").strip()
    ]
    if incomplete:
        candidates.append(
            {
                "code": "sensor_mapping_incomplete",
                "category": "data_completeness",
                "severity": "critical",
                "detail": f"{len(incomplete)} sensor definitions lack a name or category.",
            }
        )
    if manifest.skipped_properties:
        candidates.append(
            {
                "code": "planned_properties_unavailable",
                "category": "data_completeness",
                "severity": "warning",
                "detail": (
                    f"{len(manifest.skipped_properties)} declared properties are unavailable."
                ),
            }
        )
    return candidates


async def screen_device(
    database: AsyncDatabase[Document],
    principal: Principal,
    project_id: str,
    snapshot: Document,
    planning: InspectionPlanningManifest,
    manifest: DeviceInspectionManifest,
) -> Document:
    candidates = _structural_candidates(snapshot, manifest)
    evidence: Document = {
        "premises": manifest.premises,
        "available_properties": manifest.available_properties,
        "selected_properties": manifest.selected_properties,
        "skipped_properties": manifest.skipped_properties,
    }
    if not manifest.selected_properties:
        candidates.append(
            {
                "code": "operational_data_unavailable",
                "category": "data_completeness",
                "severity": "warning",
                "detail": "No planned operational property is available for this device.",
            }
        )
        return {
            "node_id": manifest.node_id,
            "node_label": manifest.node_label,
            "screening_status": "inconclusive",
            "evidence": evidence,
            "candidates": candidates,
            "recommended_deep_checks": [],
        }
    try:
        response = await query_data(
            database,
            principal,
            project_id,
            DataQuery(
                device_id=manifest.node_label,
                properties=manifest.selected_properties,
                start_time=planning.window_start,
                end_time=planning.window_end,
            ),
        )
        summary = summarize_payload(response.data, planning.window_end)
        evidence["statistics"] = summary
        if not summary["sample_count"]:
            candidates.append(
                {
                    "code": "no_samples_in_window",
                    "category": "missingness",
                    "severity": "warning",
                    "detail": (
                        "The data source returned no numeric samples in the inspection window."
                    ),
                }
            )
        if summary.get("latest_timestamp") is None:
            candidates.append(
                {
                    "code": "freshness_not_verifiable",
                    "category": "data_freshness",
                    "severity": "warning",
                    "detail": "No timestamp was present, so data freshness cannot be verified.",
                }
            )
        elif int(summary.get("stale_seconds") or 0) > 3_600:
            candidates.append(
                {
                    "code": "stale_operational_data",
                    "category": "data_freshness",
                    "severity": "warning",
                    "detail": (
                        f"Latest observed timestamp is {summary['stale_seconds']} seconds before "
                        "the window end."
                    ),
                }
            )
        if summary.get("constant_series"):
            candidates.append(
                {
                    "code": "constant_signal_candidates",
                    "category": "anomaly",
                    "severity": "warning",
                    "detail": "One or more signals remained constant and require Agent review: "
                    + ", ".join(summary["constant_series"][:10]),
                }
            )
    except AppError as error:
        evidence["query_error"] = {"code": error.code, "message": error.message}
        candidates.append(
            {
                "code": "operational_query_failed",
                "category": "data_completeness",
                "severity": "warning",
                "detail": f"Operational query failed: {error.message}",
            }
        )
    needs_deep = any(
        item.get("category") in {"anomaly", "efficiency", "optimization"}
        or item.get("severity") == "critical"
        for item in candidates
    )
    return {
        "node_id": manifest.node_id,
        "node_label": manifest.node_label,
        "screening_status": "attention" if candidates else "normal",
        "evidence": evidence,
        "candidates": candidates,
        "recommended_deep_checks": (
            ["inspect related devices", "query a narrower time window", "compare related signals"]
            if needs_deep
            else []
        ),
    }
