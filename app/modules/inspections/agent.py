from __future__ import annotations

import asyncio
import json
from contextlib import suppress
from datetime import UTC, datetime
from typing import Any, cast

from pymongo import ReturnDocument
from pymongo.asynchronous.database import AsyncDatabase

from app.core.config import Settings
from app.core.errors import AppError
from app.core.ids import new_id
from app.core.security import Principal
from app.modules.agents.prompts import render_agent_system_prompt
from app.modules.agents.providers.openai_compatible import OpenAICompatibleProvider
from app.modules.agents.providers.registry import ProviderRegistry
from app.modules.agents.runtime.engine import AgentEngine, AgentRunRequest
from app.modules.agents.runtime.types import JsonObject, Message, ToolResult, Usage
from app.modules.agents.tools.base import Tool, ToolContext
from app.modules.agents.tools.project import project_tools
from app.modules.auth.model_settings import resolve_provider_runtime
from app.modules.inspections.schemas import (
    DeviceInspectionManifest,
    InspectionDimension,
    InspectionFinding,
    InspectionNodeResult,
    InspectionOverallConclusion,
    InspectionPlanningManifest,
    InspectionTaskEdge,
    InspectionTaskGraph,
    InspectionTaskNode,
)
from app.modules.inspections.screening import screen_device, summarize_payload
from app.modules.inspections.service import append_event
from app.modules.projects.data import query_data
from app.modules.projects.schemas import DataQuery

Document = dict[str, Any]
TERMINAL_STATES = {"completed", "partial", "failed", "cancelled"}
DIMENSIONS = (
    "operating_condition",
    "anomaly",
    "efficiency",
    "optimization",
    "data_completeness",
    "data_freshness",
    "missingness",
)
DIMENSION_STATUSES = ["normal", "attention", "critical", "not_assessable", "not_applicable"]


def _review_schema() -> JsonObject:
    dimension = {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": DIMENSION_STATUSES},
            "summary": {"type": "string", "minLength": 1, "maxLength": 2_000},
        },
        "required": ["status", "summary"],
        "additionalProperties": False,
    }
    finding = {
        "type": "object",
        "properties": {
            "code": {"type": "string", "minLength": 1, "maxLength": 120},
            "severity": {"type": "string", "enum": ["info", "warning", "critical"]},
            "category": {"type": "string", "minLength": 1, "maxLength": 120},
            "title": {"type": "string", "minLength": 1, "maxLength": 240},
            "detail": {"type": "string", "minLength": 1, "maxLength": 4_000},
            "evidence": {"type": "array", "items": {"type": "string"}, "maxItems": 20},
        },
        "required": ["code", "severity", "category", "title", "detail"],
        "additionalProperties": False,
    }
    review = {
        "type": "object",
        "properties": {
            "node_id": {"type": "string", "minLength": 1},
            "status": {
                "type": "string",
                "enum": ["normal", "warning", "critical", "inconclusive", "skipped"],
            },
            "summary": {"type": "string", "minLength": 1, "maxLength": 4_000},
            "dimensions": {
                "type": "object",
                "properties": {name: dimension for name in DIMENSIONS},
                "required": list(DIMENSIONS),
                "additionalProperties": False,
            },
            "findings": {"type": "array", "items": finding, "maxItems": 50},
            "recommendations": {"type": "array", "items": {"type": "string"}, "maxItems": 30},
            "assumptions": {"type": "array", "items": {"type": "string"}, "maxItems": 30},
            "limitations": {"type": "array", "items": {"type": "string"}, "maxItems": 30},
            "evidence_refs": {"type": "array", "items": {"type": "string"}, "maxItems": 50},
        },
        "required": ["node_id", "status", "summary", "dimensions"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "reviews": {"type": "array", "items": review, "minItems": 1, "maxItems": 30}
        },
        "required": ["reviews"],
        "additionalProperties": False,
    }


def _overall_schema() -> JsonObject:
    return {
        "type": "object",
        "properties": {
            "status": {
                "type": "string",
                "enum": ["normal", "warning", "critical", "inconclusive", "skipped"],
            },
            "executive_summary": {"type": "string", "minLength": 1, "maxLength": 8_000},
            "operating_assessment": {"type": "string", "minLength": 1, "maxLength": 4_000},
            "anomaly_assessment": {"type": "string", "minLength": 1, "maxLength": 4_000},
            "efficiency_assessment": {"type": "string", "minLength": 1, "maxLength": 4_000},
            "optimization_opportunities": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": 50,
            },
            "data_quality_assessment": {"type": "string", "minLength": 1, "maxLength": 4_000},
            "coverage_limitations": {"type": "array", "items": {"type": "string"}, "maxItems": 50},
            "recommended_actions": {"type": "array", "items": {"type": "string"}, "maxItems": 50},
        },
        "required": [
            "status",
            "executive_summary",
            "operating_assessment",
            "anomaly_assessment",
            "efficiency_assessment",
            "data_quality_assessment",
        ],
        "additionalProperties": False,
    }


def _assignment_plan_schema() -> JsonObject:
    return {
        "type": "object",
        "properties": {
            "objective": {"type": "string", "minLength": 1, "maxLength": 2_000},
            "scope_summary": {"type": "string", "minLength": 1, "maxLength": 2_000},
            "steps": {
                "type": "array",
                "minItems": 1,
                "maxItems": 12,
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string", "minLength": 1, "maxLength": 160},
                        "description": {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": 1_000,
                        },
                    },
                    "required": ["title", "description"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["objective", "scope_summary", "steps"],
        "additionalProperties": False,
    }


def _assignment_result_schema() -> JsonObject:
    schema = _overall_schema()
    properties = cast(JsonObject, schema["properties"])
    properties["report_markdown"] = {
        "type": "string",
        "minLength": 1,
        "maxLength": 40_000,
    }
    properties["findings"] = {
        "type": "array",
        "maxItems": 50,
        "items": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "minLength": 1, "maxLength": 120},
                "severity": {"type": "string", "enum": ["info", "warning", "critical"]},
                "category": {"type": "string", "minLength": 1, "maxLength": 120},
                "title": {"type": "string", "minLength": 1, "maxLength": 240},
                "detail": {"type": "string", "minLength": 1, "maxLength": 4_000},
                "node_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "maxItems": 30,
                },
                "evidence": {
                    "type": "array",
                    "items": {"type": "string"},
                    "maxItems": 30,
                },
            },
            "required": ["code", "severity", "category", "title", "detail"],
            "additionalProperties": False,
        },
    }
    cast(list[str], schema["required"]).extend(["report_markdown", "findings"])
    return schema


class InspectionCoordinator:
    def __init__(
        self,
        database: AsyncDatabase[Document],
        settings: Settings,
        providers: ProviderRegistry,
    ) -> None:
        self._database = database
        self._settings = settings
        self._providers = providers
        self._tasks: dict[str, asyncio.Task[None]] = {}

    async def start(self, principal: Principal, run_id: str) -> None:
        run = await self._database.inspection_runs.find_one(
            {"_id": run_id, "owner_id": principal.user_id}
        )
        if run is None:
            raise AppError(
                "inspection_run_not_found", "Inspection run was not found", status_code=404
            )
        if run["status"] in TERMINAL_STATES:
            return
        claimed = await self._database.inspection_runs.find_one_and_update(
            {"_id": run_id, "owner_id": principal.user_id, "status": {"$in": ["ready", "queued"]}},
            {
                "$set": {
                    "status": "running",
                    "started_at": datetime.now(UTC),
                    "updated_at": datetime.now(UTC),
                }
            },
            return_document=ReturnDocument.AFTER,
        )
        if claimed is None:
            if run.get("status") == "running":
                return
            raise AppError(
                "inspection_run_not_ready", "Inspection run cannot be started", status_code=409
            )
        task = asyncio.create_task(self._execute(run_id), name=f"inspection-run:{run_id}")
        self._tasks[run_id] = task
        task.add_done_callback(lambda completed: self._forget(run_id, completed))

    def _forget(self, run_id: str, task: asyncio.Task[None]) -> None:
        if self._tasks.get(run_id) is task:
            self._tasks.pop(run_id, None)

    async def cancel(self, principal: Principal, run_id: str) -> None:
        run = await self._database.inspection_runs.find_one(
            {"_id": run_id, "owner_id": principal.user_id}
        )
        if run is None:
            raise AppError(
                "inspection_run_not_found", "Inspection run was not found", status_code=404
            )
        task = self._tasks.get(run_id)
        if task is not None:
            task.cancel()
            return
        if run.get("status") not in TERMINAL_STATES:
            await self._finish(run_id, "cancelled")

    async def recover(self) -> None:
        interrupted = await self._database.inspection_runs.find({"status": "running"}).to_list(None)
        for run in interrupted:
            await self._database.inspection_runs.update_one(
                {"_id": run["_id"]},
                {
                    "$set": {
                        "status": "partial",
                        "error": "server_restarted_during_inspection",
                        "completed_at": datetime.now(UTC),
                    }
                },
            )
        queued = await self._database.inspection_runs.find({"status": "queued"}).to_list(None)
        for run in queued:
            await self.start(Principal(user_id=str(run["owner_id"]), username=""), str(run["_id"]))

    async def close(self) -> None:
        tasks = list(self._tasks.values())
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._tasks.clear()

    async def _execute(self, run_id: str) -> None:
        try:
            run = await self._database.inspection_runs.find_one({"_id": run_id})
            if run is None:
                return
            principal = Principal(user_id=str(run["owner_id"]), username="")
            runtime = await resolve_provider_runtime(
                self._database,
                principal.user_id,
                self._settings,
                requested_provider=run.get("requested_provider"),
                requested_api_style=run.get("requested_api_style"),
                requested_model=run.get("requested_model"),
                multimodal=False,
            )
            provider = self._providers.get(
                runtime.provider,
                runtime.api_style,
                api_key=runtime.api_key,
                base_url=runtime.base_url,
            )
            await self._database.inspection_runs.update_one(
                {"_id": run_id},
                {"$set": {"provider": runtime.provider, "model": runtime.model}},
            )
            run["provider"] = runtime.provider
            run["model"] = runtime.model
            await append_event(
                self._database,
                run_id,
                "run_started",
                {"provider": runtime.provider, "model": runtime.model},
            )
            planning = InspectionPlanningManifest.model_validate(run["planning_manifest"])
            snapshot = cast(Document, run["snapshot"])
            if run.get("trigger") == "assignment":
                await self._execute_assignment(
                    run, principal, provider, runtime.model, planning, snapshot
                )
                return
            await self._set_stage(run_id, "stage:screening", "running")
            screenings: dict[str, Document] = {}
            for index, manifest in enumerate(planning.devices, start=1):
                await append_event(
                    self._database,
                    run_id,
                    "node_scan_started",
                    {"node_id": manifest.node_id, "node_label": manifest.node_label},
                )
                screening = await screen_device(
                    self._database,
                    principal,
                    str(run["project_id"]),
                    snapshot,
                    planning,
                    manifest,
                )
                screenings[manifest.node_id] = screening
                await self._database.inspection_screenings.update_one(
                    {"run_id": run_id, "node_id": manifest.node_id},
                    {
                        "$set": {
                            **screening,
                            "run_id": run_id,
                            "project_id": run["project_id"],
                            "owner_id": run["owner_id"],
                            "created_at": datetime.now(UTC),
                        }
                    },
                    upsert=True,
                )
                await append_event(self._database, run_id, "node_scan_completed", screening)
                await self._update_progress(run_id, 0.35 * index / len(planning.devices))
            await self._set_stage(run_id, "stage:screening", "succeeded")
            await self._set_stage(run_id, "stage:review", "running")
            total_usage = Usage()
            batch_size = 12
            for start in range(0, len(planning.devices), batch_size):
                batch = planning.devices[start : start + batch_size]
                usage = await self._review_batch(
                    run, principal, provider, runtime.model, planning, snapshot, batch, screenings
                )
                total_usage += usage
                completed = min(start + len(batch), len(planning.devices))
                await self._update_usage(run_id, total_usage)
                await self._update_progress(run_id, 0.35 + 0.5 * completed / len(planning.devices))
            results = await (
                self._database.inspection_node_results.find({"run_id": run_id})
                .sort("node_label", 1)
                .to_list(None)
            )
            if len(results) != len(planning.devices):
                raise RuntimeError(
                    f"Agent review coverage mismatch: {len(results)}/{len(planning.devices)}"
                )
            await self._set_stage(run_id, "stage:review", "succeeded")
            await self._set_stage(run_id, "stage:summary", "running")
            overall, usage = await self._review_overall(
                run, principal, provider, runtime.model, planning, results
            )
            total_usage += usage
            await self._update_usage(run_id, total_usage)
            await self._set_stage(run_id, "stage:summary", "succeeded")
            await self._set_stage(run_id, "stage:report", "running")
            report = self._report(run, planning, results, overall, total_usage, runtime.model)
            findings = [finding for result in results for finding in result.get("findings", [])]
            await self._database.inspection_runs.update_one(
                {"_id": run_id},
                {
                    "$set": {
                        "node_results": [self._public_result(item) for item in results],
                        "findings": findings,
                        "overall_conclusion": overall,
                        "report": report,
                        "progress": 1,
                    }
                },
            )
            await self._set_stage(run_id, "stage:report", "succeeded")
            await append_event(self._database, run_id, "report_ready", {"report": report})
            await self._finish(run_id, "completed")
        except asyncio.CancelledError:
            await self._finish(run_id, "cancelled")
            raise
        except Exception as error:
            await self._finish(run_id, "failed", f"{type(error).__name__}: {error}")

    async def _execute_assignment(
        self,
        run: Document,
        principal: Principal,
        provider: OpenAICompatibleProvider,
        model: str,
        planning: InspectionPlanningManifest,
        snapshot: Document,
    ) -> None:
        run_id = str(run["_id"])
        plan_holder: dict[str, Document] = {}
        result_holder: dict[str, Document] = {}
        await self._set_stage(run_id, "stage:planning", "running")

        async def set_plan(arguments: JsonObject, _: ToolContext) -> ToolResult:
            if "value" in plan_holder:
                raise AppError(
                    "assignment_plan_already_set",
                    "The temporary assignment plan has already been submitted",
                    status_code=409,
                )
            steps = cast(list[Document], arguments["steps"])
            planned_steps: list[Document] = [
                {
                    "id": f"assignment:step:{index}",
                    "title": str(step["title"]),
                    "description": str(step["description"]),
                }
                for index, step in enumerate(steps, start=1)
            ]
            plan: Document = {
                "objective": str(arguments["objective"]),
                "scope_summary": str(arguments["scope_summary"]),
                "steps": planned_steps,
            }
            graph_nodes = [
                InspectionTaskNode(
                    id="stage:planning",
                    kind="stage",
                    title="Agent interpret assignment",
                    status="succeeded",
                    progress=1,
                ),
                *[
                    InspectionTaskNode(
                        id=str(step["id"]),
                        kind="stage",
                        title=str(step["title"]),
                        status="running" if index == 0 else "ready",
                    )
                    for index, step in enumerate(planned_steps)
                ],
                InspectionTaskNode(
                    id="stage:report",
                    kind="report",
                    title="Deliver assignment result",
                ),
            ]
            graph_edges: list[InspectionTaskEdge] = []
            for index in range(len(graph_nodes) - 1):
                source = graph_nodes[index].id
                target = graph_nodes[index + 1].id
                graph_edges.append(
                    InspectionTaskEdge(
                        id=f"flow:{source}:{target}",
                        source=source,
                        target=target,
                        relation="produces" if target == "stage:report" else "flow",
                    )
                )
            graph = InspectionTaskGraph(nodes=graph_nodes, edges=graph_edges).model_dump(
                mode="json"
            )
            plan_holder["value"] = plan
            await self._database.inspection_runs.update_one(
                {"_id": run_id},
                {"$set": {"assignment_plan": plan, "task_graph": graph}},
            )
            await append_event(
                self._database,
                run_id,
                "assignment_plan_ready",
                {"plan": plan, "task_graph": graph},
            )
            await self._update_progress(run_id, 0.12)
            return ToolResult(
                tool_call_id="",
                content=json.dumps({"accepted": True, "step_count": len(steps)}),
            )

        async def submit_result(arguments: JsonObject, _: ToolContext) -> ToolResult:
            if "value" not in plan_holder:
                raise AppError(
                    "assignment_plan_required",
                    "Submit the temporary assignment plan before the result",
                    status_code=422,
                )
            overall = InspectionOverallConclusion(
                status=cast(Any, arguments["status"]),
                executive_summary=str(arguments["executive_summary"]),
                operating_assessment=str(arguments["operating_assessment"]),
                anomaly_assessment=str(arguments["anomaly_assessment"]),
                efficiency_assessment=str(arguments["efficiency_assessment"]),
                optimization_opportunities=[
                    str(item) for item in arguments.get("optimization_opportunities", [])
                ],
                data_quality_assessment=str(arguments["data_quality_assessment"]),
                coverage_limitations=[
                    str(item) for item in arguments.get("coverage_limitations", [])
                ],
                recommended_actions=[
                    str(item) for item in arguments.get("recommended_actions", [])
                ],
                review_model=model,
                reviewed_at=datetime.now(UTC),
            ).model_dump(mode="json")
            findings = [
                InspectionFinding.model_validate(item).model_dump(mode="json")
                for item in cast(list[Document], arguments.get("findings", []))
            ]
            result_holder["value"] = {
                "overall": overall,
                "findings": findings,
                "report": {
                    "title": "Temporary assignment result",
                    "media_type": "text/markdown",
                    "content": str(arguments["report_markdown"]),
                    "created_at": datetime.now(UTC),
                },
            }
            return ToolResult(
                tool_call_id="",
                content="Temporary assignment result accepted",
                terminate=True,
            )

        tools = (
            Tool(
                name="set_assignment_plan",
                description=(
                    "Commit the temporary assignment objective, bounded scope, and execution steps "
                    "before using investigation tools."
                ),
                input_schema=_assignment_plan_schema(),
                execute=set_plan,
                effect="write",
                execution_mode="sequential",
                result_visibility="both",
                idempotent=True,
            ),
            *project_tools(self._database),
            Tool(
                name="submit_assignment_result",
                description=(
                    "Submit the final evidence-backed result for the temporary assignment."
                ),
                input_schema=_assignment_result_schema(),
                execute=submit_result,
                effect="write",
                execution_mode="sequential",
                result_visibility="both",
                idempotent=True,
            ),
        )
        usage = await self._agent_call(
            run,
            provider,
            model,
            new_id("assignment"),
            [
                Message(
                    role="user",
                    content=(
                        "Execute this temporary assignment. Decide its scope and method before "
                        "investigating. The Reality Model snapshot below is orientation context; "
                        "use project tools for authoritative RDF or operational data.\n\n"
                        + json.dumps(
                            {
                                "instruction": planning.instruction,
                                "window": {
                                    "start": planning.window_start.isoformat(),
                                    "end": planning.window_end.isoformat(),
                                },
                                "reality_revision": planning.reality_revision,
                                "nodes": [
                                    {
                                        "id": node.get("id"),
                                        "type": node.get("type"),
                                        "label": (
                                            node.get("data", {}).get("label")
                                            if isinstance(node.get("data"), dict)
                                            else None
                                        ),
                                    }
                                    for node in snapshot.get("nodes", [])
                                    if isinstance(node, dict)
                                ],
                            },
                            ensure_ascii=False,
                            separators=(",", ":"),
                        )
                    ),
                )
            ],
            tools,
            surface="assignment",
            max_turns=20,
        )
        if "value" not in plan_holder:
            raise RuntimeError("Temporary Assignment Agent did not submit an execution plan")
        if "value" not in result_holder:
            raise RuntimeError("Temporary Assignment Agent did not submit a final result")
        result = result_holder["value"]
        graph = await self._database.inspection_runs.find_one({"_id": run_id}, {"task_graph": 1})
        task_graph = cast(Document, (graph or {}).get("task_graph") or {})
        for node in task_graph.get("nodes", []):
            if isinstance(node, dict):
                node["status"] = "succeeded"
                node["progress"] = 1
        await self._database.inspection_runs.update_one(
            {"_id": run_id},
            {
                "$set": {
                    "task_graph": task_graph,
                    "node_results": [],
                    "findings": result["findings"],
                    "overall_conclusion": result["overall"],
                    "report": result["report"],
                    "progress": 1,
                }
            },
        )
        await self._update_usage(run_id, usage)
        await append_event(
            self._database,
            run_id,
            "assignment_completed",
            {"task_graph": task_graph, "conclusion": result["overall"]},
        )
        await append_event(
            self._database, run_id, "report_ready", {"report": result["report"]}
        )
        await self._finish(run_id, "completed")

    async def _review_batch(
        self,
        run: Document,
        principal: Principal,
        provider: OpenAICompatibleProvider,
        model: str,
        planning: InspectionPlanningManifest,
        snapshot: Document,
        batch: list[DeviceInspectionManifest],
        screenings: dict[str, Document],
    ) -> Usage:
        run_id = str(run["_id"])
        batch_id = new_id("batch")
        expected = {item.node_id: item for item in batch}

        async def deep_query(arguments: JsonObject, _: ToolContext) -> ToolResult:
            node_id = str(arguments["node_id"])
            manifest = expected.get(node_id)
            if manifest is None:
                raise AppError(
                    "device_outside_review_batch", "Device is outside this batch", status_code=422
                )
            requested = arguments.get("properties")
            selected = (
                [str(item) for item in requested if str(item) in manifest.available_properties]
                if isinstance(requested, list)
                else manifest.selected_properties
            )
            response = await query_data(
                self._database,
                principal,
                str(run["project_id"]),
                DataQuery(
                    device_id=manifest.node_label,
                    properties=selected or None,
                    start_time=planning.window_start,
                    end_time=planning.window_end,
                ),
            )
            summary = summarize_payload(response.data, planning.window_end)
            return ToolResult(
                tool_call_id="",
                content=json.dumps(summary, ensure_ascii=False, separators=(",", ":")),
                details={"node_id": node_id, "deep_inspection": True},
            )

        async def related(arguments: JsonObject, _: ToolContext) -> ToolResult:
            node_id = str(arguments["node_id"])
            manifest = expected.get(node_id)
            if manifest is None:
                raise AppError(
                    "device_outside_review_batch", "Device is outside this batch", status_code=422
                )
            all_manifests = {item.node_id: item for item in planning.devices}
            payload = [
                {
                    "manifest": all_manifests[related_id].model_dump(mode="json"),
                    "screening": screenings.get(related_id),
                }
                for related_id in manifest.related_node_ids
                if related_id in all_manifests
            ]
            return ToolResult(
                tool_call_id="",
                content=json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            )

        async def submit(arguments: JsonObject, _: ToolContext) -> ToolResult:
            reviews = arguments.get("reviews")
            if not isinstance(reviews, list):
                raise AppError("invalid_device_reviews", "reviews must be a list", status_code=422)
            received = {str(item.get("node_id")) for item in reviews if isinstance(item, dict)}
            if received != set(expected):
                raise AppError(
                    "incomplete_device_reviews",
                    "Submit exactly one review for every device in this batch",
                    status_code=422,
                    details={"expected": sorted(expected), "received": sorted(received)},
                )
            now = datetime.now(UTC)
            for raw in reviews:
                review = cast(Document, raw)
                manifest = expected[str(review["node_id"])]
                dimensions = {
                    name: InspectionDimension.model_validate(
                        review["dimensions"][name]
                    ).model_dump()
                    for name in DIMENSIONS
                }
                findings = [
                    InspectionFinding.model_validate(
                        {**item, "node_ids": [manifest.node_id]}
                    ).model_dump(mode="json")
                    for item in review.get("findings", [])
                ]
                result = InspectionNodeResult(
                    node_id=manifest.node_id,
                    node_label=manifest.node_label,
                    grade=manifest.grade,
                    status=review["status"],
                    summary=review["summary"],
                    dimensions=dimensions,
                    findings=[InspectionFinding.model_validate(item) for item in findings],
                    recommendations=[str(item) for item in review.get("recommendations", [])],
                    assumptions=[str(item) for item in review.get("assumptions", [])],
                    limitations=[str(item) for item in review.get("limitations", [])],
                    evidence_refs=[str(item) for item in review.get("evidence_refs", [])],
                    review_model=model,
                    reviewed_at=now,
                )
                document = {
                    **result.model_dump(mode="json"),
                    "run_id": run_id,
                    "project_id": run["project_id"],
                    "owner_id": run["owner_id"],
                    "batch_id": batch_id,
                }
                await self._database.inspection_node_results.update_one(
                    {"run_id": run_id, "node_id": manifest.node_id},
                    {"$set": document},
                    upsert=True,
                )
                await self._set_task(run_id, f"device:{manifest.node_id}", "succeeded", 1)
                await append_event(
                    self._database,
                    run_id,
                    "node_conclusion",
                    {"result": result.model_dump(mode="json")},
                )
            return ToolResult(
                tool_call_id="",
                content=json.dumps({"accepted": len(reviews)}),
                terminate=True,
            )

        tools = (
            Tool(
                name="query_inspection_device_data",
                description=(
                    "Run a bounded deep query for one device in this review batch and return "
                    "compact statistics."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "node_id": {"type": "string"},
                        "properties": {
                            "type": "array",
                            "items": {"type": "string"},
                            "maxItems": 50,
                        },
                    },
                    "required": ["node_id"],
                    "additionalProperties": False,
                },
                execute=deep_query,
                effect="external",
                result_visibility="both",
            ),
            Tool(
                name="get_inspection_related_devices",
                description=(
                    "Read locked RDF-derived neighboring device plans and screening evidence."
                ),
                input_schema={
                    "type": "object",
                    "properties": {"node_id": {"type": "string"}},
                    "required": ["node_id"],
                    "additionalProperties": False,
                },
                execute=related,
                effect="read",
                result_visibility="both",
            ),
            Tool(
                name="submit_device_reviews",
                description=(
                    "Submit the final Agent-reviewed conclusion for every device in this batch."
                ),
                input_schema=_review_schema(),
                execute=submit,
                effect="write",
                execution_mode="sequential",
                result_visibility="both",
                idempotent=True,
            ),
        )
        prompt_payload = {
            "run_premises": planning.premises,
            "window": {
                "start": planning.window_start.isoformat(),
                "end": planning.window_end.isoformat(),
            },
            "instruction": planning.instruction,
            "devices": [
                {
                    "manifest": item.model_dump(mode="json"),
                    "screening": screenings[item.node_id],
                }
                for item in batch
            ],
        }
        return await self._agent_call(
            run,
            provider,
            model,
            batch_id,
            [
                Message(
                    role="user",
                    content=(
                        "Review every device in this batch. Deep-check suspicious evidence when "
                        "useful, "
                        "then call submit_device_reviews exactly once with all devices.\n\n"
                        + json.dumps(prompt_payload, ensure_ascii=False, separators=(",", ":"))
                    ),
                )
            ],
            tools,
        )

    async def _review_overall(
        self,
        run: Document,
        principal: Principal,
        provider: OpenAICompatibleProvider,
        model: str,
        planning: InspectionPlanningManifest,
        results: list[Document],
    ) -> tuple[Document, Usage]:
        holder: dict[str, Document] = {}

        async def submit(arguments: JsonObject, _: ToolContext) -> ToolResult:
            overall = InspectionOverallConclusion(
                **arguments,
                review_model=model,
                reviewed_at=datetime.now(UTC),
            )
            holder["value"] = overall.model_dump(mode="json")
            await append_event(
                self._database,
                str(run["_id"]),
                "overall_conclusion",
                {"conclusion": holder["value"]},
            )
            return ToolResult(
                tool_call_id="", content="Overall conclusion accepted", terminate=True
            )

        counts: dict[str, int] = {}
        for result in results:
            counts[str(result["status"])] = counts.get(str(result["status"]), 0) + 1
        payload = {
            "premises": planning.premises,
            "scope": {
                "target_count": len(planning.devices),
                "reviewed_count": len(results),
                "status_counts": counts,
                "minimum_grade": planning.minimum_grade,
                "reality_revision": planning.reality_revision,
            },
            "device_results": [self._public_result(item) for item in results],
        }
        usage = await self._agent_call(
            run,
            provider,
            model,
            new_id("summary"),
            [
                Message(
                    role="user",
                    content=(
                        "Audit the complete set of Agent-reviewed device conclusions and call "
                        "submit_overall_conclusion. Report operational, anomaly, efficiency, "
                        "optimization, data completeness, freshness, and missingness outcomes. "
                        "Counts are authoritative.\n\n"
                        + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
                    ),
                )
            ],
            (
                Tool(
                    name="submit_overall_conclusion",
                    description=(
                        "Submit the final project-level inspection conclusion after auditing all "
                        "device reviews."
                    ),
                    input_schema=_overall_schema(),
                    execute=submit,
                    effect="write",
                    execution_mode="sequential",
                    result_visibility="both",
                    idempotent=True,
                ),
            ),
        )
        if "value" not in holder:
            raise RuntimeError("Inspection Agent did not submit an overall conclusion")
        return holder["value"], usage

    async def _agent_call(
        self,
        run: Document,
        provider: OpenAICompatibleProvider,
        model: str,
        operation_id: str,
        messages: list[Message],
        tools: tuple[Tool, ...],
        *,
        surface: str = "inspection",
        max_turns: int = 12,
    ) -> Usage:
        run_id = str(run["_id"])

        async def emit(event_type: str, data: JsonObject) -> None:
            mapped = {
                "message_start": "agent_message_start",
                "message_delta": "agent_message_delta",
                "message_end": "agent_message_end",
                "reasoning_delta": "agent_reasoning_delta",
                "run_start": "agent_operation_started",
                "run_end": "agent_operation_ended",
            }.get(event_type, event_type)
            await append_event(
                self._database,
                run_id,
                mapped,
                {**data, "operation_id": operation_id},
            )

        started = datetime.now(UTC)
        result = await AgentEngine(provider).run(
            AgentRunRequest(
                run_id=operation_id,
                session_id=run_id,
                project_id=str(run["project_id"]),
                user_id=str(run["owner_id"]),
                model=model,
                system_prompt=render_agent_system_prompt(
                    surface, timezone_name=self._settings.agent_timezone
                ),
                messages=tuple(messages),
                tools=tools,
                max_turns=max_turns,
                context_char_budget=self._settings.agent_context_char_budget,
                source="inspection",
                feature=surface,
            ),
            emit,
        )
        await self._database.agent_operations.insert_one(
            {
                "_id": operation_id,
                "session_id": run_id,
                "project_id": run["project_id"],
                "surface": "inspection",
                "workload_id": run_id,
                "owner_id": run["owner_id"],
                "status": "completed",
                "provider": run.get("provider"),
                "model": model,
                "usage": {
                    "input_tokens": result.usage.input_tokens,
                    "output_tokens": result.usage.output_tokens,
                    "cached_input_tokens": result.usage.cached_input_tokens,
                    "reasoning_tokens": result.usage.reasoning_tokens,
                },
                "created_at": started,
                "started_at": started,
                "updated_at": datetime.now(UTC),
                "completed_at": datetime.now(UTC),
            }
        )
        return result.usage

    async def _update_usage(self, run_id: str, usage: Usage) -> None:
        payload = {
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "cached_input_tokens": usage.cached_input_tokens,
            "reasoning_tokens": usage.reasoning_tokens,
        }
        await self._database.inspection_runs.update_one(
            {"_id": run_id}, {"$set": {"usage": payload}}
        )
        await append_event(self._database, run_id, "usage_update", payload)

    async def _update_progress(self, run_id: str, progress: float) -> None:
        bounded = max(0.0, min(1.0, progress))
        await self._database.inspection_runs.update_one(
            {"_id": run_id}, {"$set": {"progress": bounded}}
        )
        await append_event(
            self._database,
            run_id,
            "progress_update",
            {"progress": bounded},
        )

    async def _set_stage(self, run_id: str, task_id: str, status: str) -> None:
        await self._set_task(run_id, task_id, status, 1 if status == "succeeded" else 0)
        await append_event(
            self._database,
            run_id,
            "graph_updated",
            {"task_id": task_id, "status": status},
        )

    async def _set_task(self, run_id: str, task_id: str, status: str, progress: float) -> None:
        run = await self._database.inspection_runs.find_one({"_id": run_id}, {"task_graph": 1})
        if run is None:
            return
        graph = cast(Document, run.get("task_graph") or {"nodes": [], "edges": []})
        for node in graph.get("nodes", []):
            if isinstance(node, dict) and node.get("id") == task_id:
                node["status"] = status
                node["progress"] = progress
                break
        await self._database.inspection_runs.update_one(
            {"_id": run_id}, {"$set": {"task_graph": graph}}
        )

    async def _finish(self, run_id: str, status: str, error: str | None = None) -> None:
        now = datetime.now(UTC)
        await self._database.inspection_runs.update_one(
            {"_id": run_id},
            {
                "$set": {
                    "status": status,
                    "error": error,
                    "completed_at": now,
                    "updated_at": now,
                }
            },
        )
        with suppress(Exception):
            await append_event(
                self._database,
                run_id,
                "run_end",
                {"outcome": status, **({"error": error} if error else {})},
            )

    @staticmethod
    def _public_result(document: Document) -> Document:
        excluded = {"_id", "run_id", "project_id", "owner_id", "batch_id"}
        return {key: value for key, value in document.items() if key not in excluded}

    @staticmethod
    def _report(
        run: Document,
        planning: InspectionPlanningManifest,
        results: list[Document],
        overall: Document,
        usage: Usage,
        model: str,
    ) -> Document:
        counts: dict[str, int] = {}
        for result in results:
            counts[str(result["status"])] = counts.get(str(result["status"]), 0) + 1
        lines = [
            f"# {run.get('template_name', 'Operational inspection report')}",
            "",
            f"- Reality Model revision: {planning.reality_revision}",
            (
                f"- Inspection window: {planning.window_start.isoformat()} to "
                f"{planning.window_end.isoformat()}"
            ),
            f"- Minimum device grade: {planning.minimum_grade}",
            f"- Agent model: {model}",
            f"- Reviewed coverage: {len(results)}/{len(planning.devices)}",
            f"- Token usage: {usage.input_tokens} input / {usage.output_tokens} output",
            "",
            "## Overall conclusion",
            "",
            str(overall["executive_summary"]),
            "",
            f"**Status:** {overall['status']}",
            "",
            "## Assessments",
            "",
            f"- Operating condition: {overall['operating_assessment']}",
            f"- Anomalies: {overall['anomaly_assessment']}",
            f"- Efficiency: {overall['efficiency_assessment']}",
            f"- Data completeness, freshness and missingness: {overall['data_quality_assessment']}",
            "",
            "## Result counts",
            "",
            *[f"- {key}: {value}" for key, value in sorted(counts.items())],
            "",
            "## Device conclusions",
            "",
        ]
        for result in results:
            lines.extend(
                [
                    f"### {result['node_label']} · Grade {result['grade']} · {result['status']}",
                    "",
                    str(result["summary"]),
                    "",
                ]
            )
            for finding in result.get("findings", []):
                lines.append(f"- [{finding['severity']}] {finding['title']}: {finding['detail']}")
            lines.append("")
        return {
            "title": f"{run.get('template_name', 'Inspection')} report",
            "media_type": "text/markdown",
            "content": "\n".join(lines),
            "created_at": datetime.now(UTC),
        }
