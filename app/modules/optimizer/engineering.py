from __future__ import annotations

import json
import math
from collections import Counter, deque
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, cast

from pymongo import ReturnDocument
from pymongo.asynchronous.database import AsyncDatabase

from app.core.config import Settings
from app.core.errors import AppError
from app.core.security import Principal
from app.modules.agents.runtime.types import Message, ProviderRequest, TextDelta
from app.modules.auth.model_settings import (
    configured_auxiliary_model,
    read_model_settings,
    resolve_provider_runtime,
)
from app.modules.optimizer.schemas import (
    EngineeringConfigUpdate,
    EngineeringConfigView,
    EngineeringDerivedValue,
    EngineeringModelBinding,
    EngineeringModelGroup,
    EngineeringParameterValue,
    EngineeringTopology,
    EngineeringTopologyConnection,
    EngineeringTopologyInference,
    NodeEngineeringParameter,
    NodeEngineeringState,
)
from app.modules.projects.data import owned_project, project_rdf
from app.modules.studio.schemas import EngineeringParameterSchema
from app.modules.studio.service import _engineering_schema

if TYPE_CHECKING:
    from app.modules.agents.providers.registry import ProviderRegistry

Document = dict[str, Any]

PHYSICAL_PROPERTIES: tuple[Document, ...] = (
    {
        "key": "rho_w",
        "label": "Water density",
        "description": "Default density used by chilled- and condenser-water calculations.",
        "unit": "kg/m³",
        "value": 1_000.0,
    },
    {
        "key": "cp_w",
        "label": "Water specific heat",
        "description": "Default specific heat used by water-side heat-balance calculations.",
        "unit": "kJ/(kg·K)",
        "value": 4.186,
    },
    {
        "key": "rho_air",
        "label": "Air density",
        "description": "Default air density used by the cooling-tower airflow calculation.",
        "unit": "kg/m³",
        "value": 1.293,
    },
)

WATER_SYSTEM_PARAMETERS: tuple[Document, ...] = (
    {
        "key": "dt_hdr_chw_fixed",
        "label": "Chilled-water design ΔT",
        "description": "Default chilled-water supply/return temperature difference.",
        "unit": "°C",
        "default": 5.0,
        "required": True,
        "minimum": 0.1,
        "maximum": 20.0,
    },
    {
        "key": "dt_hdr_cw_default",
        "label": "Condenser-water design ΔT",
        "description": "Default condenser-water supply/return temperature difference.",
        "unit": "°C",
        "default": 5.0,
        "required": True,
        "minimum": 0.1,
        "maximum": 20.0,
    },
    {
        "key": "coef_resistance_chw",
        "label": "Chilled-water loop resistance",
        "description": "Project-specific resistance coefficient used by the pump solver.",
        "unit": "s²/m⁵",
        "default": None,
        "required": True,
        "minimum": 0.0,
        "maximum": None,
    },
    {
        "key": "coef_resistance_cw",
        "label": "Condenser-water loop resistance",
        "description": "Project-specific resistance coefficient used by the pump solver.",
        "unit": "s²/m⁵",
        "default": None,
        "required": True,
        "minimum": 0.0,
        "maximum": None,
    },
)

TOPOLOGY_LABELS: dict[str, str] = {
    "chilled_water_pumps": "Chilled-water pump topology",
    "condenser_water_pumps": "Condenser-water pump topology",
}

INFRASTRUCTURE_TYPES = {"pipe", "valve", "heat_exchanger"}


def _data(node: Document) -> Document:
    value = node.get("data")
    return value if isinstance(value, dict) else {}


def _label(node: Document) -> str:
    data = _data(node)
    return str(data.get("label") or data.get("name") or node.get("id") or "Equipment")


def _numeric(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


async def _engineering_schemas(
    database: AsyncDatabase[Document],
) -> list[EngineeringParameterSchema]:
    documents = await database.engineering_parameter_schemas.find({}).to_list(None)
    return sorted(
        (_engineering_schema(document) for document in documents),
        key=lambda schema: (schema.sort_order, schema.device_type),
    )


def _node_group_map(nodes: list[Document]) -> dict[str, str]:
    groups = {str(node.get("id")): node for node in nodes if node.get("type") == "group"}
    result: dict[str, str] = {}
    for node in nodes:
        node_id = str(node.get("id") or "")
        parent_id = str(node.get("parent_id") or "")
        if node_id and parent_id in groups:
            result[node_id] = parent_id
    for group_id, group in groups.items():
        children = _data(group).get("child")
        if not isinstance(children, list):
            continue
        for child_id in children:
            candidate = str(child_id)
            if candidate and candidate not in result:
                result[candidate] = group_id
    return result


def _node_states(
    nodes: list[Document], schemas: list[EngineeringParameterSchema]
) -> list[NodeEngineeringState]:
    schemas_by_type = {schema.device_type: schema for schema in schemas}
    nodes_by_id = {str(node.get("id")): node for node in nodes}
    group_by_node = _node_group_map(nodes)
    states: list[NodeEngineeringState] = []
    for node in nodes:
        if node.get("type") == "group":
            continue
        node_id = str(node.get("id") or "")
        device_type = str(node.get("type") or "node")
        schema = schemas_by_type.get(device_type)
        if schema is None:
            continue
        values = _data(node).get("engineering_parameters")
        configured = values if isinstance(values, dict) else {}
        parameters = [
            NodeEngineeringParameter(
                key=definition.key,
                label=definition.label,
                unit=definition.unit,
                value=_numeric(configured.get(definition.key)),
                required=definition.required,
            )
            for definition in schema.parameters
        ]
        required = [parameter for parameter in parameters if parameter.required]
        group_id = group_by_node.get(node_id)
        states.append(
            NodeEngineeringState(
                node_id=node_id,
                label=_label(node),
                device_type=device_type,
                group_id=group_id,
                group_label=_label(nodes_by_id[group_id]) if group_id in nodes_by_id else None,
                configured_count=sum(parameter.value is not None for parameter in parameters),
                required_count=len(required),
                complete=all(parameter.value is not None for parameter in required),
                parameters=parameters,
            )
        )
    return sorted(states, key=lambda item: (item.device_type, item.label.casefold()))


def _group_members(group_id: str, nodes: list[Document]) -> list[Document]:
    nodes_by_id = {str(node.get("id")): node for node in nodes}
    children: dict[str, set[str]] = {}
    for node in nodes:
        parent_id = str(node.get("parent_id") or "")
        if parent_id:
            children.setdefault(parent_id, set()).add(str(node.get("id")))
    group = nodes_by_id[group_id]
    explicit = _data(group).get("child")
    if isinstance(explicit, list):
        children.setdefault(group_id, set()).update(str(item) for item in explicit)

    result: list[Document] = []
    pending = list(children.get(group_id, set()))
    visited: set[str] = set()
    while pending:
        node_id = pending.pop()
        if node_id in visited or node_id not in nodes_by_id:
            continue
        visited.add(node_id)
        node = nodes_by_id[node_id]
        if node.get("type") == "group":
            pending.extend(children.get(node_id, set()))
        else:
            result.append(node)
    return sorted(result, key=lambda node: _label(node).casefold())


def _model_group(
    group_id: str, label: str, members: list[NodeEngineeringState]
) -> EngineeringModelGroup:
    device_types = {member.device_type for member in members}
    device_type = next(iter(device_types)) if len(device_types) == 1 else "mixed"
    derived: list[EngineeringDerivedValue] = []
    metric = {
        "chiller": ("q_cool_rated", "Total rated cooling capacity", "kW"),
        "pump": ("p_rated", "Total rated pump power", "kW"),
        "cooling_tower": ("p_rated_unit", "Total rated fan power", "kW"),
    }.get(device_type)
    if metric:
        key, derived_label, unit = metric
        values = [
            parameter.value
            for member in members
            for parameter in member.parameters
            if parameter.key == key and parameter.value is not None
        ]
        if len(values) == len(members):
            derived.append(
                EngineeringDerivedValue(
                    key=key,
                    label=derived_label,
                    value=sum(values),
                    unit=unit,
                )
            )
    return EngineeringModelGroup(
        group_id=group_id,
        label=label,
        device_type=device_type,
        member_node_ids=[member.node_id for member in members],
        member_labels=[member.label for member in members],
        complete=all(member.complete for member in members),
        derived_values=derived,
    )


def _model_groups(
    nodes: list[Document], node_states: list[NodeEngineeringState]
) -> list[EngineeringModelGroup]:
    state_by_id = {item.node_id: item for item in node_states}
    grouped_node_ids: set[str] = set()
    result: list[EngineeringModelGroup] = []
    for group in nodes:
        if group.get("type") != "group":
            continue
        group_id = str(group.get("id") or "")
        members = [
            state_by_id[str(member.get("id"))]
            for member in _group_members(group_id, nodes)
            if str(member.get("id")) in state_by_id
        ]
        if not members:
            continue
        grouped_node_ids.update(member.node_id for member in members)
        result.append(_model_group(group_id, _label(group), members))

    for node_state in node_states:
        if node_state.node_id in grouped_node_ids:
            continue
        result.append(
            _model_group(
                f"standalone:{node_state.node_id}",
                node_state.label,
                [node_state],
            )
        )
    return sorted(result, key=lambda item: (item.label.casefold(), item.group_id))


def _pump_system(node: Document) -> str | None:
    data = _data(node)
    text = " ".join(
        str(data.get(key) or "")
        for key in ("label", "category", "root_category", "modeling_note", "description")
    ).casefold()
    if any(token in text for token in ("chilled", "chwp", "冷冻", "冷水泵")):
        return "chilled_water_pumps"
    if any(token in text for token in ("condenser", "cooling water", "cwp", "冷却水泵", "冷却泵")):
        return "condenser_water_pumps"
    return None


def _adjacency(nodes: list[Document], edges: list[Document]) -> dict[str, set[str]]:
    known = {str(node.get("id")) for node in nodes}
    result: dict[str, set[str]] = {node_id: set() for node_id in known}
    for edge in edges:
        source = str(edge.get("source") or "")
        target = str(edge.get("target") or "")
        if source in known and target in known:
            result[source].add(target)
            result[target].add(source)
    return result


def _related_chillers(
    pump_id: str, nodes_by_id: dict[str, Document], adjacency: dict[str, set[str]]
) -> set[str]:
    found: set[str] = set()
    pending: deque[tuple[str, int]] = deque([(pump_id, 0)])
    visited = {pump_id}
    while pending:
        current, depth = pending.popleft()
        if depth >= 8:
            continue
        for neighbor in adjacency.get(current, set()):
            if neighbor in visited:
                continue
            visited.add(neighbor)
            node = nodes_by_id[neighbor]
            node_type = str(node.get("type") or "")
            if node_type == "chiller":
                found.add(neighbor)
            elif node_type in INFRASTRUCTURE_TYPES:
                pending.append((neighbor, depth + 1))
    return found


def infer_topologies(nodes: list[Document], edges: list[Document]) -> list[EngineeringTopology]:
    nodes_by_id = {str(node.get("id")): node for node in nodes}
    adjacency = _adjacency(nodes, edges)
    pumps = [node for node in nodes if node.get("type") == "pump"]
    result: list[EngineeringTopology] = []
    for system, label in TOPOLOGY_LABELS.items():
        system_pumps = [node for node in pumps if _pump_system(node) == system]
        connections: list[EngineeringTopologyConnection] = []
        associations: dict[str, set[str]] = {}
        for pump in system_pumps:
            pump_id = str(pump.get("id"))
            related = _related_chillers(pump_id, nodes_by_id, adjacency)
            associations[pump_id] = related
            connections.append(
                EngineeringTopologyConnection(
                    node_id=pump_id,
                    label=_label(pump),
                    related_node_ids=sorted(related),
                    related_labels=sorted(_label(nodes_by_id[item]) for item in related),
                )
            )

        nonempty = [related for related in associations.values() if related]
        chiller_degrees = Counter(item for related in nonempty for item in related)
        if not system_pumps:
            mode = "unknown"
            confidence = 0.1
            reason = "No pump nodes with a recognizable system role were found."
        elif len(nonempty) != len(system_pumps):
            mode = "mixed" if nonempty else "unknown"
            confidence = 0.45 if nonempty else 0.2
            reason = "Some pump nodes could not be traced to a chiller through the current graph."
        elif all(len(related) == 1 for related in nonempty) and all(
            degree == 1 for degree in chiller_degrees.values()
        ):
            mode = "one_to_one"
            confidence = 0.9
            reason = "Every pump resolves to one distinct chiller through the Reality Model."
        elif any(len(related) > 1 for related in nonempty) or any(
            degree > 1 for degree in chiller_degrees.values()
        ):
            mode = "parallel"
            confidence = 0.82
            reason = "Multiple pumps or chillers share a connected water-side network."
        else:
            mode = "mixed"
            confidence = 0.55
            reason = "The graph contains both dedicated and shared pump relationships."
        result.append(
            EngineeringTopology(
                system=cast(Any, system),
                label=label,
                mode=cast(Any, mode),
                inferred_mode=cast(Any, mode),
                source="reality_model",
                confidence=confidence,
                reason=reason,
                inferred_reason=reason,
                connections=connections,
            )
        )
    return result


def _physical_properties() -> list[EngineeringParameterValue]:
    return [
        EngineeringParameterValue(
            key=str(item["key"]),
            label=str(item["label"]),
            description=str(item["description"]),
            unit=str(item["unit"]),
            value=float(item["value"]),
            default_value=float(item["value"]),
            source="default",
            editable=False,
        )
        for item in PHYSICAL_PROPERTIES
    ]


def _water_system_values(saved: Document) -> list[EngineeringParameterValue]:
    return [
        EngineeringParameterValue(
            key=str(definition["key"]),
            label=str(definition["label"]),
            description=str(definition["description"]),
            unit=str(definition["unit"]),
            value=(
                _numeric(saved.get(str(definition["key"])))
                if str(definition["key"]) in saved
                else cast(float | None, definition["default"])
            ),
            default_value=cast(float | None, definition["default"]),
            source=(
                "user"
                if str(definition["key"]) in saved
                else "default"
                if definition["default"] is not None
                else "missing"
            ),
            editable=True,
            required=bool(definition["required"]),
            minimum=cast(float | None, definition["minimum"]),
            maximum=cast(float | None, definition["maximum"]),
        )
        for definition in WATER_SYSTEM_PARAMETERS
    ]


def _apply_topology_overrides(
    inferred: list[EngineeringTopology], config: Document
) -> list[EngineeringTopology]:
    raw = config.get("topologies")
    overrides = raw if isinstance(raw, dict) else {}
    result: list[EngineeringTopology] = []
    for topology in inferred:
        override = overrides.get(topology.system)
        if not isinstance(override, dict):
            result.append(topology)
            continue
        result.append(
            EngineeringTopology.model_validate(
                {
                    **topology.model_dump(mode="json"),
                    "mode": override.get("mode", topology.mode),
                    "source": override.get("source", "manual"),
                    "confidence": override.get("confidence", 1),
                    "reason": override.get("reason") or topology.reason,
                }
            )
        )
    return result


async def engineering_config(
    database: AsyncDatabase[Document], principal: Principal, project_id: str
) -> EngineeringConfigView:
    project = await owned_project(database, principal, project_id)
    nodes = [item for item in project.get("nodes", []) if isinstance(item, dict)]
    edges = [item for item in project.get("edges", []) if isinstance(item, dict)]
    schemas = await _engineering_schemas(database)
    node_states = _node_states(nodes, schemas)
    raw_config = project.get("optimizer_engineering_config")
    config = raw_config if isinstance(raw_config, dict) else {}
    water = config.get("water_system_parameters")
    saved_water = water if isinstance(water, dict) else {}
    inferred = infer_topologies(nodes, edges)
    raw_bindings = config.get("model_bindings")
    saved_bindings = raw_bindings if isinstance(raw_bindings, dict) else {}
    model_ids = [str(value) for value in saved_bindings.values() if value]
    models = await database.models.find(
        {
            "_id": {"$in": model_ids},
            "project_id": project_id,
            "owner_id": principal.user_id,
        }
    ).to_list(None) if model_ids else []
    models_by_id = {str(item.get("_id")): item for item in models}
    node_type_by_id = {str(item.get("id")): str(item.get("type")) for item in nodes}
    bindings: list[EngineeringModelBinding] = []
    for node_id, raw_model_id in saved_bindings.items():
        model_id = str(raw_model_id)
        model = models_by_id.get(model_id)
        expected_type = node_type_by_id.get(str(node_id), "")
        if model is None:
            status = "missing"
            model_name = "Missing model"
            known_type = (
                expected_type
                if expected_type in {"chiller", "pump", "cooling_tower"}
                else "pump"
            )
            device_type = cast(Any, known_type)
        else:
            device_type = cast(Any, model.get("device_type"))
            status = "ready" if str(model.get("device_type")) == expected_type else "incompatible"
            model_name = str(model.get("name") or model_id)
        bindings.append(
            EngineeringModelBinding(
                node_id=str(node_id),
                model_id=model_id,
                model_name=model_name,
                device_type=device_type,
                status=cast(Any, status),
            )
        )
    return EngineeringConfigView(
        project_id=project_id,
        graph_revision=int(project.get("graph_revision", 0)),
        physical_properties=_physical_properties(),
        water_system_parameters=_water_system_values(saved_water),
        nodes=node_states,
        model_groups=_model_groups(nodes, node_states),
        model_bindings=sorted(bindings, key=lambda item: item.node_id),
        topologies=_apply_topology_overrides(inferred, config),
        updated_at=config.get("updated_at"),
    )


def _validated_water_values(values: dict[str, float]) -> dict[str, float]:
    definitions = {str(item["key"]): item for item in WATER_SYSTEM_PARAMETERS}
    unknown = sorted(set(values) - set(definitions))
    if unknown:
        raise AppError(
            "engineering_parameter_unknown",
            "Unknown water-system engineering parameters",
            status_code=422,
            details={"keys": unknown},
        )
    result: dict[str, float] = {}
    for key, value in values.items():
        numeric = _numeric(value)
        definition = definitions[key]
        if numeric is None:
            raise AppError(
                "engineering_parameter_invalid", f"{key} must be finite", status_code=422
            )
        minimum = cast(float | None, definition["minimum"])
        maximum = cast(float | None, definition["maximum"])
        if minimum is not None and numeric < minimum:
            raise AppError(
                "engineering_parameter_invalid",
                f"{key} must be at least {minimum:g}",
                status_code=422,
            )
        if maximum is not None and numeric > maximum:
            raise AppError(
                "engineering_parameter_invalid",
                f"{key} must be at most {maximum:g}",
                status_code=422,
            )
        result[key] = numeric
    return result


async def update_engineering_config(
    database: AsyncDatabase[Document],
    principal: Principal,
    project_id: str,
    body: EngineeringConfigUpdate,
) -> EngineeringConfigView:
    water = _validated_water_values(body.water_system_parameters)
    current_project = await owned_project(database, principal, project_id)
    supported_nodes = {
        str(item.get("id")): str(item.get("type"))
        for item in current_project.get("nodes", [])
        if isinstance(item, dict) and item.get("type") in {"chiller", "pump", "cooling_tower"}
    }
    unknown_nodes = sorted(set(body.model_bindings) - set(supported_nodes))
    if unknown_nodes:
        raise AppError(
            "engineering_model_binding_node_unknown",
            "Model bindings reference unknown optimizer nodes",
            status_code=422,
            details={"node_ids": unknown_nodes},
        )
    model_ids = list(set(body.model_bindings.values()))
    models = await database.models.find(
        {
            "_id": {"$in": model_ids},
            "project_id": project_id,
            "owner_id": principal.user_id,
        }
    ).to_list(None) if model_ids else []
    models_by_id = {str(item.get("_id")): item for item in models}
    missing_models = sorted(set(model_ids) - set(models_by_id))
    if missing_models:
        raise AppError(
            "engineering_model_binding_model_missing",
            "One or more bound performance models do not exist",
            status_code=422,
            details={"model_ids": missing_models},
        )
    incompatible = [
        node_id
        for node_id, model_id in body.model_bindings.items()
        if str(models_by_id[model_id].get("device_type")) != supported_nodes[node_id]
    ]
    if incompatible:
        raise AppError(
            "engineering_model_binding_incompatible",
            "A performance model must match the bound node device type",
            status_code=422,
            details={"node_ids": sorted(incompatible)},
        )
    topology_systems = [item.system for item in body.topologies]
    if len(topology_systems) != len(set(topology_systems)):
        raise AppError(
            "engineering_topology_duplicate",
            "Each water system can have only one topology definition",
            status_code=422,
        )
    now = datetime.now(UTC)
    config = {
        "graph_revision": body.graph_revision,
        "water_system_parameters": water,
        "model_bindings": body.model_bindings,
        "topologies": {
            item.system: item.model_dump(mode="json", exclude={"system"})
            for item in body.topologies
        },
        "updated_at": now,
        "updated_by": principal.user_id,
    }
    project = await database.projects.find_one_and_update(
        {
            "_id": project_id,
            "owner_id": principal.user_id,
            "graph_revision": body.graph_revision,
        },
        {"$set": {"optimizer_engineering_config": config, "updated_at": now}},
        return_document=ReturnDocument.AFTER,
    )
    if project is None:
        owned = await database.projects.find_one(
            {"_id": project_id, "owner_id": principal.user_id}, {"graph_revision": 1}
        )
        if owned is None:
            raise AppError("project_not_found", "Project was not found", status_code=404)
        raise AppError(
            "engineering_graph_revision_conflict",
            "The Reality Model changed after engineering configuration was loaded",
            status_code=409,
            details={"current_revision": int(owned.get("graph_revision", 0))},
        )
    return await engineering_config(database, principal, project_id)


def _json_object(value: str) -> Document:
    cleaned = value.strip().strip("`").strip()
    if cleaned.casefold().startswith("json"):
        cleaned = cleaned[4:].strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("The model response did not contain a JSON object")
    parsed = json.loads(cleaned[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("The model response must be a JSON object")
    return parsed


async def infer_topologies_with_llm(
    database: AsyncDatabase[Document],
    principal: Principal,
    project_id: str,
    settings: Settings,
    providers: ProviderRegistry,
) -> EngineeringTopologyInference:
    project = await owned_project(database, principal, project_id)
    nodes = [item for item in project.get("nodes", []) if isinstance(item, dict)]
    edges = [item for item in project.get("edges", []) if isinstance(item, dict)]
    deterministic = infer_topologies(nodes, edges)
    model_settings = await read_model_settings(database, principal.user_id, settings)
    provider_id = model_settings.active_provider
    auxiliary = await configured_auxiliary_model(database, principal.user_id, provider_id)
    runtime = await resolve_provider_runtime(
        database,
        principal.user_id,
        settings,
        requested_provider=provider_id,
        requested_api_style=None,
        requested_model=auxiliary or None,
        multimodal=False,
    )
    provider = providers.get(
        runtime.provider,
        runtime.api_style,
        api_key=runtime.api_key,
        base_url=runtime.base_url,
    )
    summary = [
        {
            "system": item.system,
            "mode": item.mode,
            "reason": item.reason,
            "connections": [connection.model_dump(mode="json") for connection in item.connections],
        }
        for item in deterministic
    ]
    request = ProviderRequest(
        model=runtime.model,
        system_prompt=(
            "You review an EnerAI Reality Model RDF graph to classify pump topology. "
            "Return JSON only with this exact shape: "
            '{"topologies":[{"system":"chilled_water_pumps|condenser_water_pumps",'
            '"mode":"one_to_one|parallel|mixed|unknown","confidence":0.0,'
            '"reason":"short evidence-based explanation"}]}. '
            "Return exactly one item for each system. Treat brick:feed/isFedBy as "
            "directed node relationships and brick:hasPart/isPartOf as grouping. "
            "Do not invent nodes or links."
        ),
        messages=(
            Message(
                role="user",
                content=(
                    "Rule-based inference:\n"
                    + json.dumps(summary, ensure_ascii=False)
                    + "\n\nProject RDF:\n"
                    + project_rdf(project)[:28_000]
                ),
            ),
        ),
        temperature=0,
        max_output_tokens=900,
    )
    chunks: list[str] = []
    try:
        async for event in provider.stream(request):
            if isinstance(event, TextDelta):
                chunks.append(event.delta)
        parsed = _json_object("".join(chunks))
        raw_items = parsed.get("topologies")
        if not isinstance(raw_items, list):
            raise ValueError("The model response did not include topologies")
        by_system = {str(item.get("system")): item for item in raw_items if isinstance(item, dict)}
        suggestions: list[EngineeringTopology] = []
        for base in deterministic:
            raw = by_system.get(base.system)
            if raw is None:
                raise ValueError(f"The model omitted {base.system}")
            suggestions.append(
                EngineeringTopology.model_validate(
                    {
                        **base.model_dump(mode="json"),
                        "mode": raw.get("mode"),
                        "source": "llm",
                        "confidence": max(0, min(1, float(raw.get("confidence", 0.5)))),
                        "reason": str(raw.get("reason") or "LLM review returned no explanation"),
                    }
                )
            )
        validated = suggestions
    except AppError:
        raise
    except Exception as error:
        raise AppError(
            "engineering_llm_inference_failed",
            "The configured model could not return a valid topology review",
            status_code=502,
            details={"error": type(error).__name__},
        ) from error
    return EngineeringTopologyInference(
        graph_revision=int(project.get("graph_revision", 0)),
        topologies=validated,
    )
