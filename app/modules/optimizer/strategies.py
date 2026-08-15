from __future__ import annotations

import asyncio
import itertools
import math
from collections import Counter
from datetime import UTC, datetime
from typing import Any, cast

from pymongo import DESCENDING
from pymongo.asynchronous.database import AsyncDatabase

from app.core.errors import AppError
from app.core.ids import new_id
from app.core.security import Principal
from app.modules.optimizer.engineering import _pump_system, engineering_config
from app.modules.optimizer.modeling import optimization_artifact_compatible, predict_model
from app.modules.optimizer.schemas import (
    EngineeringConfigView,
    OptimizationPreflightIssue,
    OptimizationPreflightResult,
    OptimizationRange,
    OptimizationRunCounters,
    OptimizationRunStage,
    OptimizationRunView,
    OptimizationStrategyCreate,
    OptimizationStrategyList,
    OptimizationStrategyRow,
    OptimizationStrategySummary,
)
from app.modules.projects.data import owned_project

Document = dict[str, Any]

RUN_STAGES = (
    ("preflight", "Validate engineering inputs"),
    ("model_bundle", "Assemble representative models"),
    ("operating_grid", "Generate operating-condition grid"),
    ("chiller_combinations", "Generate chiller combinations"),
    ("equipment_simulation", "Traverse equipment candidates"),
    ("candidate_ranking", "Select minimum-power candidates"),
    ("strategy_matrix", "Build optimization strategy matrix"),
)

_tasks: set[asyncio.Task[None]] = set()


def _strategy_summary(document: Document) -> OptimizationStrategySummary:
    payload = dict(document)
    payload["id"] = str(payload.pop("_id"))
    return OptimizationStrategySummary.model_validate(payload)


def _run_view(document: Document) -> OptimizationRunView:
    payload = dict(document)
    payload["id"] = str(payload.pop("_id"))
    payload.pop("owner_id", None)
    payload.pop("strategy_snapshot", None)
    return OptimizationRunView.model_validate(payload)


def _range_values(value: OptimizationRange) -> list[float]:
    if value.maximum < value.minimum:
        raise ValueError("Range maximum must not be lower than minimum")
    count = int(math.floor((value.maximum - value.minimum) / value.step + 1e-9)) + 1
    return [round(value.minimum + index * value.step, 8) for index in range(count)]


def _supply_values(strategy: OptimizationStrategySummary) -> list[float]:
    supply = strategy.search_space.chilled_water_supply
    if not supply.enabled:
        return [supply.fixed]
    return _range_values(
        OptimizationRange(minimum=supply.minimum, maximum=supply.maximum, step=supply.step)
    )


async def create_strategy(
    database: AsyncDatabase[Document],
    principal: Principal,
    project_id: str,
    body: OptimizationStrategyCreate,
) -> OptimizationStrategySummary:
    await owned_project(database, principal, project_id)
    now = datetime.now(UTC)
    document: Document = {
        "_id": new_id("optstr"),
        "project_id": project_id,
        "owner_id": principal.user_id,
        "name": body.name.strip(),
        "description": body.description.strip(),
        "status": "draft",
        "revision": 1,
        "search_space": body.search_space.model_dump(mode="json"),
        "solver": body.solver.model_dump(mode="json"),
        "created_at": now,
        "updated_at": now,
    }
    await database.optimization_strategies.insert_one(document)
    return _strategy_summary(document)


async def list_strategies(
    database: AsyncDatabase[Document], principal: Principal, project_id: str
) -> OptimizationStrategyList:
    await owned_project(database, principal, project_id)
    documents = await (
        database.optimization_strategies.find(
            {"project_id": project_id, "owner_id": principal.user_id}
        )
        .sort("updated_at", DESCENDING)
        .to_list(None)
    )
    return OptimizationStrategyList(
        items=[_strategy_summary(document) for document in documents], total=len(documents)
    )


async def owned_strategy(
    database: AsyncDatabase[Document],
    principal: Principal,
    project_id: str,
    strategy_id: str,
) -> Document:
    document = await database.optimization_strategies.find_one(
        {"_id": strategy_id, "project_id": project_id, "owner_id": principal.user_id}
    )
    if document is None:
        raise AppError("optimization_strategy_not_found", "Strategy was not found", status_code=404)
    return document


async def update_strategy(
    database: AsyncDatabase[Document],
    principal: Principal,
    project_id: str,
    strategy_id: str,
    body: OptimizationStrategyCreate,
) -> OptimizationStrategySummary:
    current = await owned_strategy(database, principal, project_id, strategy_id)
    now = datetime.now(UTC)
    await database.optimization_strategies.update_one(
        {"_id": strategy_id},
        {
            "$set": {
                "name": body.name.strip(),
                "description": body.description.strip(),
                "search_space": body.search_space.model_dump(mode="json"),
                "solver": body.solver.model_dump(mode="json"),
                "status": "draft",
                "updated_at": now,
            },
            "$inc": {"revision": 1},
        },
    )
    current.update(
        {
            "name": body.name.strip(),
            "description": body.description.strip(),
            "search_space": body.search_space.model_dump(mode="json"),
            "solver": body.solver.model_dump(mode="json"),
            "status": "draft",
            "revision": int(current.get("revision", 1)) + 1,
            "updated_at": now,
        }
    )
    return _strategy_summary(current)


async def delete_strategy(
    database: AsyncDatabase[Document],
    principal: Principal,
    project_id: str,
    strategy_id: str,
) -> None:
    await owned_strategy(database, principal, project_id, strategy_id)
    active = await database.optimization_runs.find_one(
        {"strategy_id": strategy_id, "status": {"$in": ["queued", "running"]}}
    )
    if active:
        raise AppError(
            "optimization_strategy_running",
            "A strategy cannot be deleted while a run is active",
            status_code=409,
        )
    await database.optimization_strategies.delete_one({"_id": strategy_id})


def _average_value(values: list[Any], path: str = "artifact") -> Any:
    first = values[0]
    if all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in values):
        return math.fsum(float(value) for value in values) / len(values)
    if isinstance(first, dict) and all(isinstance(value, dict) for value in values):
        keys = set(first)
        if any(set(value) != keys for value in values):
            raise ValueError(f"{path} keys do not match")
        return {
            key: _average_value([value[key] for value in values], f"{path}.{key}")
            for key in first
        }
    if all(value == first for value in values):
        return first
    raise ValueError(f"{path} structures do not match")


def _model_is_optimization_compatible(model: Document, device_type: str) -> bool:
    artifact = model.get("artifact")
    return (
        model.get("algorithm") == "polynomial"
        and isinstance(artifact, dict)
        and device_type in models_by_type()
        and optimization_artifact_compatible(cast(Any, device_type), artifact)
    )


async def _context(
    database: AsyncDatabase[Document], principal: Principal, project_id: str
) -> tuple[Document, EngineeringConfigView, dict[str, Document]]:
    project = await owned_project(database, principal, project_id)
    engineering = await engineering_config(database, principal, project_id)
    model_ids = [binding.model_id for binding in engineering.model_bindings]
    documents = await database.models.find(
        {
            "_id": {"$in": model_ids},
            "project_id": project_id,
            "owner_id": principal.user_id,
            "status": "ready",
        }
    ).to_list(None) if model_ids else []
    return project, engineering, {str(item.get("_id")): item for item in documents}


def _preflight_from_context(
    strategy: OptimizationStrategySummary,
    project: Document,
    engineering: EngineeringConfigView,
    models: dict[str, Document],
) -> OptimizationPreflightResult:
    issues: list[OptimizationPreflightIssue] = []
    warnings: list[OptimizationPreflightIssue] = []
    binding_by_node = {item.node_id: item for item in engineering.model_bindings}
    supported_nodes = [item for item in engineering.nodes if item.device_type in models_by_type()]
    for node in supported_nodes:
        if not node.complete:
            issues.append(OptimizationPreflightIssue(
                code="node_parameters_incomplete",
                message=f"{node.label} has incomplete rated or operating parameters.",
                target=node.node_id,
            ))
        binding = binding_by_node.get(node.node_id)
        if not binding:
            issues.append(OptimizationPreflightIssue(
                code="model_binding_missing",
                message=f"{node.label} is not bound to a performance model.",
                target=node.node_id,
            ))
        elif binding.status != "ready" or binding.model_id not in models:
            issues.append(OptimizationPreflightIssue(
                code="model_binding_invalid",
                message=f"{node.label} has an unavailable or incompatible model binding.",
                target=node.node_id,
            ))
        elif not _model_is_optimization_compatible(models[binding.model_id], node.device_type):
            issues.append(OptimizationPreflightIssue(
                code="model_artifact_incompatible",
                message=(
                    f"{node.label} is not bound to a complete polynomial artifact supported "
                    "by traversal optimization."
                ),
                target=node.node_id,
            ))
    for parameter in engineering.water_system_parameters:
        if parameter.required and parameter.value is None:
            issues.append(OptimizationPreflightIssue(
                code="engineering_parameter_missing",
                message=f"{parameter.label} is required.",
                target=parameter.key,
            ))
        elif parameter.key in {"coef_resistance_chw", "coef_resistance_cw"} and (
            parameter.value is not None and parameter.value <= 0
        ):
            issues.append(OptimizationPreflightIssue(
                code="engineering_parameter_invalid",
                message=f"{parameter.label} must be greater than zero.",
                target=parameter.key,
            ))
    for topology in engineering.topologies:
        if topology.mode not in {"one_to_one", "parallel"}:
            issues.append(OptimizationPreflightIssue(
                code="topology_unsupported",
                message=f"{topology.label} must be one-to-one or common parallel.",
                target=topology.system,
            ))
    nodes_by_id = {
        str(item.get("id")): item
        for item in project.get("nodes", [])
        if isinstance(item, dict)
    }
    for group in engineering.model_groups:
        if group.device_type == "mixed":
            issues.append(OptimizationPreflightIssue(
                code="mixed_model_group",
                message=f"{group.label} mixes multiple device types.",
                target=group.group_id,
            ))
        if group.device_type == "pump":
            systems = {
                _pump_system(nodes_by_id[node_id])
                for node_id in group.member_node_ids
                if node_id in nodes_by_id
            }
            if None in systems or len(systems) != 1:
                issues.append(OptimizationPreflightIssue(
                    code="pump_group_role_ambiguous",
                    message=f"{group.label} must contain pumps from one recognizable water system.",
                    target=group.group_id,
                ))
        group_bindings = [binding_by_node.get(node_id) for node_id in group.member_node_ids]
        if any(binding is None or binding.model_id not in models for binding in group_bindings):
            continue
        group_models = [models[binding.model_id] for binding in group_bindings if binding]
        if group.device_type == "cooling_tower" and len({
            binding.model_id for binding in group_bindings if binding
        }) != 1:
            issues.append(OptimizationPreflightIssue(
                code="cooling_tower_group_model_mismatch",
                message=f"{group.label} must use one tower-group model for every member node.",
                target=group.group_id,
            ))
        elif group.device_type in {"chiller", "pump"}:
            try:
                _average_value([model["artifact"] for model in group_models])
            except (KeyError, ValueError) as error:
                issues.append(OptimizationPreflightIssue(
                    code="representative_model_incompatible",
                    message=f"{group.label} cannot form a representative model: {error}.",
                    target=group.group_id,
                ))
    try:
        wet_bulb = _range_values(strategy.search_space.wet_bulb)
        load_ratio = _range_values(strategy.search_space.load_ratio)
        supply = _supply_values(strategy)
        condenser_delta = _range_values(strategy.search_space.condenser_water_delta_t)
        approach = _range_values(strategy.search_space.cooling_tower_approach)
    except ValueError as error:
        issues.append(OptimizationPreflightIssue(code="search_range_invalid", message=str(error)))
        wet_bulb = load_ratio = supply = condenser_delta = approach = []
    if any(value <= 0 or value > 1 for value in load_ratio):
        issues.append(
            OptimizationPreflightIssue(
                code="load_ratio_invalid",
                message="Cooling-load ratios must remain in the interval (0, 1].",
            )
        )
    chiller_groups = [group for group in engineering.model_groups if group.device_type == "chiller"]
    possible_combinations = max(
        1,
        math.prod(len(group.member_node_ids) + 1 for group in chiller_groups) - 1,
    )
    combination_estimate = min(2, possible_combinations)
    external_count = len(wet_bulb) * len(load_ratio) * len(supply)
    candidate_count = (
        external_count * len(condenser_delta) * len(approach) * combination_estimate
    )
    if candidate_count > strategy.solver.maximum_candidates:
        issues.append(OptimizationPreflightIssue(
            code="candidate_limit_exceeded",
            message=(
                f"Estimated candidate count {candidate_count:,} exceeds the configured limit "
                f"of {strategy.solver.maximum_candidates:,}."
            ),
        ))
    if candidate_count > 500_000:
        warnings.append(OptimizationPreflightIssue(
            code="candidate_grid_large",
            message="This strategy has a large traversal grid and may take several minutes.",
        ))
    return OptimizationPreflightResult(
        ready=not issues,
        issues=issues,
        warnings=warnings,
        external_condition_count=external_count,
        estimated_candidate_count=candidate_count,
        model_group_count=len(engineering.model_groups),
    )


def models_by_type() -> set[str]:
    return {"chiller", "pump", "cooling_tower"}


async def preflight_strategy(
    database: AsyncDatabase[Document],
    principal: Principal,
    project_id: str,
    strategy_id: str,
) -> OptimizationPreflightResult:
    strategy = _strategy_summary(await owned_strategy(database, principal, project_id, strategy_id))
    project, engineering, models = await _context(database, principal, project_id)
    result = _preflight_from_context(strategy, project, engineering, models)
    await database.optimization_strategies.update_one(
        {"_id": strategy_id},
        {"$set": {"status": "ready" if result.ready else "draft", "updated_at": datetime.now(UTC)}},
    )
    return result


def _parameters(engineering: EngineeringConfigView) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for node in engineering.nodes:
        result[node.node_id] = {
            item.key: item.value for item in node.parameters if item.value is not None
        }
    return result


def _average_parameters(
    members: list[str], values: dict[str, dict[str, float]]
) -> dict[str, float]:
    keys = set.intersection(*(set(values[item]) for item in members))
    return {key: math.fsum(values[item][key] for item in members) / len(members) for key in keys}


def _representative_groups(
    project: Document,
    engineering: EngineeringConfigView,
    models: dict[str, Document],
) -> list[Document]:
    bindings = {item.node_id: item.model_id for item in engineering.model_bindings}
    parameter_values = _parameters(engineering)
    nodes_by_id = {
        str(item.get("id")): item
        for item in project.get("nodes", [])
        if isinstance(item, dict)
    }
    result: list[Document] = []
    for group in engineering.model_groups:
        if group.device_type not in models_by_type():
            continue
        model_ids = [bindings[node_id] for node_id in group.member_node_ids]
        artifacts = [models[model_id]["artifact"] for model_id in model_ids]
        if group.device_type == "cooling_tower" and len(set(model_ids)) != 1:
            raise ValueError(f"Cooling-tower group {group.label} must share one tower-group model")
        artifact = (
            artifacts[0]
            if group.device_type == "cooling_tower"
            else _average_value(artifacts)
        )
        pump_system = None
        if group.device_type == "pump":
            systems = {_pump_system(nodes_by_id[node_id]) for node_id in group.member_node_ids}
            pump_system = next(iter(systems))
        result.append({
            "group_id": group.group_id,
            "label": group.label,
            "device_type": group.device_type,
            "member_node_ids": group.member_node_ids,
            "available_count": len(group.member_node_ids),
            "parameters": _average_parameters(group.member_node_ids, parameter_values),
            "artifact": artifact,
            "pump_system": pump_system,
        })
    return result


def _capacity(group: Document, wet_bulb: float, supply_temperature: float) -> float:
    parameter = group["parameters"]
    factor = (
        parameter["coef_q_cool_rated_corr_a"] * supply_temperature**2
        + parameter["coef_q_cool_rated_corr_b"] * wet_bulb**2
        + parameter["coef_q_cool_rated_corr_c"] * supply_temperature * wet_bulb
        + parameter["coef_q_cool_rated_corr_d"] * supply_temperature
        + parameter["coef_q_cool_rated_corr_e"] * wet_bulb
        + parameter["coef_q_cool_rated_corr_f"]
    )
    capacity = float(parameter["q_cool_rated"] * factor)
    if not math.isfinite(capacity) or capacity <= 0:
        raise ValueError("invalid_capacity_correction")
    return capacity


def _chiller_combinations(
    groups: list[Document], wet_bulb: float, supply: float, load: float
) -> list[tuple[dict[str, int], dict[str, float]]]:
    capacities = {group["group_id"]: _capacity(group, wet_bulb, supply) for group in groups}
    candidates: list[tuple[dict[str, int], dict[str, float]]] = []
    for counts in itertools.product(*(range(group["available_count"] + 1) for group in groups)):
        if not any(counts):
            continue
        mapping = {group["group_id"]: count for group, count in zip(groups, counts, strict=True)}
        total_capacity = math.fsum(
            mapping[group["group_id"]] * capacities[group["group_id"]] for group in groups
        )
        if total_capacity + 1e-9 < load:
            continue
        candidates.append((mapping, capacities))
    def candidate_capacity(item: tuple[dict[str, int], dict[str, float]]) -> float:
        return math.fsum(
            item[0][group["group_id"]] * item[1][group["group_id"]] for group in groups
        )

    candidates.sort(key=lambda item: (sum(item[0].values()), candidate_capacity(item)))
    first = candidates[0] if candidates else None
    if first is None:
        return []
    first_capacity = candidate_capacity(first)
    higher_capacity = [
        item for item in candidates[1:] if candidate_capacity(item) > first_capacity + 1e-9
    ]
    second = min(higher_capacity, key=candidate_capacity) if higher_capacity else None
    return [first, second] if second else [first]


def _solve_chillers(
    groups: list[Document],
    counts: dict[str, int],
    capacities: dict[str, float],
    *,
    cooling_load: float,
    wet_bulb: float,
    supply_temperature: float,
    chilled_water_delta_t: float,
    condenser_water_delta_t: float,
    tower_approach: float,
    rho_w: float,
    cp_w: float,
    cop_initial: float,
    cop_tolerance: float,
    cop_max_iterations: int,
) -> Document:
    active = [group for group in groups if counts[group["group_id"]] > 0]
    total_flow = 3600 * cooling_load / (rho_w * cp_w * chilled_water_delta_t)
    weight_total = math.fsum(
        counts[group["group_id"]] * group["parameters"]["flow_chw_rated"] for group in active
    )
    result_groups: list[Document] = []
    total_power = 0.0
    total_condenser_flow = 0.0
    for group in active:
        group_id = group["group_id"]
        count = counts[group_id]
        weight = count * group["parameters"]["flow_chw_rated"] / weight_total
        group_load = cooling_load * weight
        unit_load = group_load / count
        unit_flow = total_flow * weight / count
        load_ratio = unit_load / capacities[group_id]
        if not (
            group["parameters"]["load_pct_min"]
            <= load_ratio
            <= group["parameters"]["load_pct_max"]
        ):
            raise ValueError("chiller_load_ratio")
        cop = cop_initial
        outputs: dict[str, float] = {}
        for _ in range(cop_max_iterations):
            q_reject = (1 + 1 / cop) * unit_load
            flow_cw = 3600 * q_reject / (rho_w * cp_w * condenser_water_delta_t)
            outputs = predict_model(
                "chiller",
                group["artifact"],
                {
                    "q_cool": unit_load,
                    "q_reject": q_reject,
                    "flow_chw": unit_flow,
                    "flow_cw": flow_cw,
                    "load_pct": load_ratio,
                    "t_chw_sup": supply_temperature,
                    "t_chw_ret": supply_temperature + chilled_water_delta_t,
                    "t_cw_sup": wet_bulb + tower_approach,
                    "t_cw_ret": wet_bulb + tower_approach + condenser_water_delta_t,
                },
            )
            calculated = outputs["cop"]
            if abs((calculated - cop) / calculated) < cop_tolerance:
                cop = calculated
                break
            cop = (cop + calculated) / 2
        else:
            raise ValueError("cop_not_converged")
        group_power = count * unit_load / cop
        total_power += group_power
        total_condenser_flow += count * flow_cw
        result_groups.append({
            "group_id": group_id,
            "count": count,
            "load_ratio": load_ratio,
            "cop": cop,
            "power": group_power,
        })
    return {
        "groups": result_groups,
        "power": total_power,
        "chilled_water_flow": total_flow,
        "condenser_water_flow": total_condenser_flow,
    }


def _solve_pump_system(
    groups: list[Document],
    target_flow: float,
    resistance: float,
    topology: str,
    chiller_count: int,
    rho_w: float,
) -> Document:
    if not groups or target_flow <= 0:
        raise ValueError("pump_group_unavailable")
    required_head_m = resistance * (target_flow / 3600) ** 2
    best: Document | None = None
    for counts in itertools.product(*(range(group["available_count"] + 1) for group in groups)):
        total_count = sum(counts)
        if not total_count or (topology == "one_to_one" and total_count != chiller_count):
            continue
        weight_total = math.fsum(
            count * group["parameters"]["flow_rated"]
            for group, count in zip(groups, counts, strict=True)
        )
        total_power = 0.0
        frequencies: list[float] = []
        feasible = True
        for group, count in zip(groups, counts, strict=True):
            if count == 0:
                continue
            parameter = group["parameters"]
            unit_flow = target_flow * count * parameter["flow_rated"] / weight_total / count
            low = parameter["freq_min"] / parameter["freq_rated"]
            high = parameter["freq_max"] / parameter["freq_rated"]

            def pump_head(
                ratio: float,
                bound_group: Document = group,
                bound_unit_flow: float = unit_flow,
            ) -> tuple[float, float]:
                output = predict_model(
                    "pump",
                    bound_group["artifact"],
                    {"flow": bound_unit_flow / max(ratio, 1e-9)},
                )
                return output["head"] * ratio**2, output["eff_pump"]

            high_head, _ = pump_head(high)
            if high_head < required_head_m:
                feasible = False
                break
            for _ in range(40):
                middle = (low + high) / 2
                middle_head, _ = pump_head(middle)
                if middle_head >= required_head_m:
                    high = middle
                else:
                    low = middle
            ratio = high
            _, efficiency = pump_head(ratio)
            if not 0 < efficiency <= 1:
                feasible = False
                break
            unit_power = (
                rho_w * 9.80665 * required_head_m * (unit_flow / 3600) / efficiency / 1000
            )
            total_power += count * unit_power
            frequencies.extend([ratio * parameter["freq_rated"]] * count)
        if feasible and total_power > 0 and (best is None or total_power < best["power"]):
            best = {
                "count": total_count,
                "frequency": math.fsum(frequencies) / len(frequencies),
                "power": total_power,
            }
    if best is None:
        raise ValueError("pump_solver_infeasible")
    return best


def _solve_towers(
    groups: list[Document], condenser_flow: float, wet_bulb: float, delta_t: float, approach: float,
    rho_w: float, rho_air: float,
) -> Document:
    if not groups:
        raise ValueError("cooling_tower_group_unavailable")
    efficiency = delta_t / (delta_t + approach)
    ratios = [
        predict_model("cooling_tower", group["artifact"], {"t_wb": wet_bulb, "eta": efficiency})[
            "air_water_ratio"
        ]
        for group in groups
    ]
    air_water_ratio = math.fsum(ratios) / len(ratios)
    if air_water_ratio <= 0:
        raise ValueError("cooling_tower_model_invalid")
    required_airflow = air_water_ratio * rho_w * condenser_flow / rho_air
    best: Document | None = None
    for counts in itertools.product(*(range(group["available_count"] + 1) for group in groups)):
        if not any(counts):
            continue
        rated_total = math.fsum(
            count * group["parameters"]["airflow_rated_unit"]
            for group, count in zip(groups, counts, strict=True)
        )
        total_power = 0.0
        frequencies: list[float] = []
        feasible = True
        for group, count in zip(groups, counts, strict=True):
            if count == 0:
                continue
            parameter = group["parameters"]
            group_airflow = required_airflow * count * parameter["airflow_rated_unit"] / rated_total
            frequency = (
                group_airflow / (count * parameter["airflow_rated_unit"]) * parameter["freq_rated"]
            )
            if not parameter["freq_min"] <= frequency <= parameter["freq_max"]:
                feasible = False
                break
            total_power += count * parameter["p_rated_unit"] * (
                frequency / parameter["freq_rated"]
            ) ** 3
            frequencies.extend([frequency] * count)
        if feasible:
            candidate = {
                "count": sum(counts),
                "frequency": math.fsum(frequencies) / len(frequencies),
                "power": total_power,
            }
            if best is None or candidate["count"] > best["count"] or (
                candidate["count"] == best["count"] and candidate["power"] < best["power"]
            ):
                best = candidate
    if best is None:
        raise ValueError("cooling_tower_solver_infeasible")
    return best


async def _set_stage(
    database: AsyncDatabase[Document], run_id: str, key: str, status: str, detail: str = ""
) -> None:
    document = await database.optimization_runs.find_one({"_id": run_id}, {"stages": 1})
    stages = list(document.get("stages", [])) if document else []
    for stage in stages:
        if stage["key"] == key:
            stage["status"] = status
            stage["detail"] = detail
    await database.optimization_runs.update_one(
        {"_id": run_id}, {"$set": {"stages": stages, "current_stage": key}}
    )


async def _execute_run(
    database: AsyncDatabase[Document], principal: Principal, project_id: str, run_id: str
) -> None:
    try:
        run = await database.optimization_runs.find_one({"_id": run_id})
        if run is None:
            return
        strategy = OptimizationStrategySummary.model_validate(run["strategy_snapshot"])
        await database.optimization_runs.update_one(
            {"_id": run_id},
            {"$set": {"status": "running", "started_at": datetime.now(UTC), "progress": 0.01}},
        )
        await _set_stage(database, run_id, "preflight", "running")
        project, engineering, models = await _context(database, principal, project_id)
        preflight = _preflight_from_context(strategy, project, engineering, models)
        if not preflight.ready:
            raise ValueError("; ".join(issue.message for issue in preflight.issues))
        await _set_stage(
            database, run_id, "preflight", "completed", "All required inputs are valid"
        )

        await _set_stage(database, run_id, "model_bundle", "running")
        groups = _representative_groups(project, engineering, models)
        await _set_stage(
            database,
            run_id,
            "model_bundle",
            "completed",
            f"{len(groups)} representative groups ready",
        )
        chiller_groups = [item for item in groups if item["device_type"] == "chiller"]
        tower_groups = [item for item in groups if item["device_type"] == "cooling_tower"]
        chilled_pumps = [
            item for item in groups if item.get("pump_system") == "chilled_water_pumps"
        ]
        condenser_pumps = [
            item for item in groups if item.get("pump_system") == "condenser_water_pumps"
        ]
        if not chiller_groups or not tower_groups or not chilled_pumps or not condenser_pumps:
            raise ValueError(
                "Chiller, chilled-water pump, condenser-water pump, and tower groups are required"
            )

        await _set_stage(database, run_id, "operating_grid", "running")
        conditions = list(itertools.product(
            _range_values(strategy.search_space.wet_bulb),
            _range_values(strategy.search_space.load_ratio),
            _supply_values(strategy),
        ))
        await _set_stage(
            database,
            run_id,
            "operating_grid",
            "completed",
            f"{len(conditions):,} external conditions",
        )
        await _set_stage(database, run_id, "chiller_combinations", "running")

        physical = {item.key: cast(float, item.value) for item in engineering.physical_properties}
        water = {item.key: cast(float, item.value) for item in engineering.water_system_parameters}
        topology = {item.system: item.mode for item in engineering.topologies}
        total_rated_capacity = math.fsum(
            group["parameters"]["q_cool_rated"] * group["available_count"]
            for group in chiller_groups
        )
        counters = OptimizationRunCounters(
            external_conditions_total=len(conditions),
            candidates_total=preflight.estimated_candidate_count,
        )
        rows: list[dict[str, Any]] = []
        failures: Counter[str] = Counter()
        await _set_stage(database, run_id, "chiller_combinations", "completed")
        await _set_stage(database, run_id, "equipment_simulation", "running")
        for condition_index, (wet_bulb, load_ratio, supply_temperature) in enumerate(conditions):
            cooling_load = total_rated_capacity * load_ratio
            condition_key = f"wb:{wet_bulb:g}|load:{load_ratio:g}|chw:{supply_temperature:g}"
            combinations = _chiller_combinations(
                chiller_groups, wet_bulb, supply_temperature, cooling_load
            )
            best: dict[str, Any] | None = None
            condition_failure = (
                "no_chiller_combination" if not combinations else "no_feasible_candidate"
            )
            for counts, capacities in combinations:
                for condenser_delta, approach in itertools.product(
                    _range_values(strategy.search_space.condenser_water_delta_t),
                    _range_values(strategy.search_space.cooling_tower_approach),
                ):
                    counters.candidates_evaluated += 1
                    try:
                        chillers = _solve_chillers(
                            chiller_groups,
                            counts,
                            capacities,
                            cooling_load=cooling_load,
                            wet_bulb=wet_bulb,
                            supply_temperature=supply_temperature,
                            chilled_water_delta_t=water["dt_hdr_chw_fixed"],
                            condenser_water_delta_t=condenser_delta,
                            tower_approach=approach,
                            rho_w=physical["rho_w"],
                            cp_w=physical["cp_w"],
                            cop_initial=strategy.solver.cop_initial,
                            cop_tolerance=strategy.solver.cop_tolerance,
                            cop_max_iterations=strategy.solver.cop_max_iterations,
                        )
                        chiller_count = sum(counts.values())
                        chwp = _solve_pump_system(
                            chilled_pumps,
                            chillers["chilled_water_flow"],
                            water["coef_resistance_chw"],
                            topology["chilled_water_pumps"],
                            chiller_count,
                            physical["rho_w"],
                        )
                        cwp = _solve_pump_system(
                            condenser_pumps,
                            chillers["condenser_water_flow"],
                            water["coef_resistance_cw"],
                            topology["condenser_water_pumps"],
                            chiller_count,
                            physical["rho_w"],
                        )
                        towers = _solve_towers(
                            tower_groups,
                            chillers["condenser_water_flow"],
                            wet_bulb,
                            condenser_delta,
                            approach,
                            physical["rho_w"],
                            physical["rho_air"],
                        )
                        total_power = (
                            chillers["power"]
                            + chwp["power"]
                            + cwp["power"]
                            + towers["power"]
                        )
                        if not math.isfinite(total_power) or total_power <= 0:
                            raise ValueError("total_power_invalid")
                        counters.candidates_feasible += 1
                        candidate = {
                            "counts": counts,
                            "condenser_delta": condenser_delta,
                            "approach": approach,
                            "chillers": chillers,
                            "chwp": chwp,
                            "cwp": cwp,
                            "towers": towers,
                            "total_power": total_power,
                        }
                        if best is None or total_power < best["total_power"]:
                            best = candidate
                    except (ValueError, KeyError, ZeroDivisionError, OverflowError) as error:
                        code = str(error) or error.__class__.__name__
                        failures[code] += 1
                        counters.candidates_rejected += 1
                        condition_failure = code
                    if counters.candidates_evaluated % 100 == 0:
                        await asyncio.sleep(0)
            if best is None:
                rows.append(OptimizationStrategyRow(
                    condition_key=condition_key,
                    wet_bulb=wet_bulb,
                    load_ratio=load_ratio,
                    cooling_load=cooling_load,
                    chilled_water_supply_temperature=supply_temperature,
                    success=False,
                    failure_code=condition_failure,
                ).model_dump(mode="json"))
            else:
                combination_code = "+".join(
                    f"{group_id}:{count}" for group_id, count in best["counts"].items() if count
                )
                rows.append(OptimizationStrategyRow(
                    condition_key=condition_key,
                    wet_bulb=wet_bulb,
                    load_ratio=load_ratio,
                    cooling_load=cooling_load,
                    chilled_water_supply_temperature=supply_temperature,
                    success=True,
                    chiller_combination_code=combination_code,
                    chiller_group_counts=best["counts"],
                    chilled_water_pump_count=best["chwp"]["count"],
                    chilled_water_pump_frequency=best["chwp"]["frequency"],
                    condenser_water_pump_count=best["cwp"]["count"],
                    condenser_water_pump_frequency=best["cwp"]["frequency"],
                    cooling_tower_count=best["towers"]["count"],
                    cooling_tower_frequency=best["towers"]["frequency"],
                    condenser_water_delta_t=best["condenser_delta"],
                    cooling_tower_approach=best["approach"],
                    chiller_power=best["chillers"]["power"],
                    chilled_water_pump_power=best["chwp"]["power"],
                    condenser_water_pump_power=best["cwp"]["power"],
                    cooling_tower_power=best["towers"]["power"],
                    total_power=best["total_power"],
                    eer=cooling_load / best["total_power"],
                ).model_dump(mode="json"))
            counters.external_conditions_completed = condition_index + 1
            progress = 0.2 + 0.65 * counters.external_conditions_completed / len(conditions)
            await database.optimization_runs.update_one(
                {"_id": run_id},
                {"$set": {
                    "progress": progress,
                    "current_condition": condition_key,
                    "counters": counters.model_dump(mode="json"),
                    "failure_summary": dict(failures.most_common(12)),
                    "rows": rows,
                }},
            )
        await _set_stage(
            database,
            run_id,
            "equipment_simulation",
            "completed",
            f"{counters.candidates_evaluated:,} candidates evaluated",
        )
        await _set_stage(database, run_id, "candidate_ranking", "running")
        successful = sum(bool(item["success"]) for item in rows)
        await _set_stage(
            database,
            run_id,
            "candidate_ranking",
            "completed",
            f"{successful:,} optimal rows selected",
        )
        await _set_stage(database, run_id, "strategy_matrix", "running")
        await _set_stage(database, run_id, "strategy_matrix", "completed", "Matrix is ready")
        final_status = (
            "completed" if successful == len(rows) else "partial" if successful else "failed"
        )
        await database.optimization_runs.update_one(
            {"_id": run_id},
            {"$set": {
                "status": final_status,
                "progress": 1,
                "current_stage": "strategy_matrix",
                "current_condition": "",
                "rows": rows,
                "counters": counters.model_dump(mode="json"),
                "failure_summary": dict(failures.most_common(12)),
                "completed_at": datetime.now(UTC),
            }},
        )
    except Exception as error:
        current = await database.optimization_runs.find_one(
            {"_id": run_id}, {"current_stage": 1}
        )
        current_stage = str(current.get("current_stage") or "") if current else ""
        if current_stage:
            await _set_stage(database, run_id, current_stage, "failed", str(error))
        await database.optimization_runs.update_one(
            {"_id": run_id},
            {"$set": {
                "status": "failed",
                "error": str(error),
                "completed_at": datetime.now(UTC),
            }},
        )


async def start_run(
    database: AsyncDatabase[Document],
    principal: Principal,
    project_id: str,
    strategy_id: str,
) -> OptimizationRunView:
    strategy_document = await owned_strategy(database, principal, project_id, strategy_id)
    strategy = _strategy_summary(strategy_document)
    project, engineering, models = await _context(database, principal, project_id)
    preflight = _preflight_from_context(strategy, project, engineering, models)
    if not preflight.ready:
        raise AppError(
            "optimization_preflight_failed",
            "Optimization inputs are not ready",
            status_code=422,
            details={"issues": [item.model_dump(mode="json") for item in preflight.issues]},
        )
    active = await database.optimization_runs.find_one(
        {
            "project_id": project_id,
            "owner_id": principal.user_id,
            "status": {"$in": ["queued", "running"]},
        }
    )
    if active:
        raise AppError(
            "optimization_run_active",
            "Only one optimization run can be active for a project",
            status_code=409,
        )
    now = datetime.now(UTC)
    run_id = new_id("optrun")
    stages = [
        OptimizationRunStage(key=key, label=label).model_dump(mode="json")
        for key, label in RUN_STAGES
    ]
    document: Document = {
        "_id": run_id,
        "project_id": project_id,
        "owner_id": principal.user_id,
        "strategy_id": strategy_id,
        "strategy_name": strategy.name,
        "strategy_snapshot": strategy.model_dump(mode="json"),
        "status": "queued",
        "progress": 0,
        "current_stage": "",
        "current_condition": "",
        "stages": stages,
        "counters": OptimizationRunCounters(
            external_conditions_total=preflight.external_condition_count,
            candidates_total=preflight.estimated_candidate_count,
        ).model_dump(mode="json"),
        "failure_summary": {},
        "rows": [],
        "error": "",
        "created_at": now,
        "started_at": None,
        "completed_at": None,
    }
    await database.optimization_runs.insert_one(document)
    task = asyncio.create_task(
        _execute_run(database, principal, project_id, run_id), name=f"optimization-run:{run_id}"
    )
    _tasks.add(task)
    task.add_done_callback(_tasks.discard)
    return _run_view(document)


async def read_run(
    database: AsyncDatabase[Document], principal: Principal, project_id: str, run_id: str
) -> OptimizationRunView:
    document = await database.optimization_runs.find_one(
        {"_id": run_id, "project_id": project_id, "owner_id": principal.user_id}
    )
    if document is None:
        raise AppError(
            "optimization_run_not_found", "Optimization run was not found", status_code=404
        )
    return _run_view(document)


async def latest_run(
    database: AsyncDatabase[Document], principal: Principal, project_id: str
) -> OptimizationRunView | None:
    document = await database.optimization_runs.find_one(
        {"project_id": project_id, "owner_id": principal.user_id}, sort=[("created_at", DESCENDING)]
    )
    return _run_view(document) if document else None
