import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.core.errors import AppError
from app.modules.studio.schemas import (
    EngineeringParameterDefinition,
    EngineeringParameterSchema,
    GraphEdge,
    GraphNode,
    StudioGraphUpdate,
)
from app.modules.studio.service import (
    _engineering_schema,
    _normalize_categories,
    validate_engineering_parameters,
    validate_graph,
)

RESOURCE_ROOT = Path(__file__).parents[1] / "resources" / "engineering_parameters"


def node(node_id: str, parent_id: str | None = None, node_type: str = "equipment") -> GraphNode:
    return GraphNode(
        id=node_id,
        type=node_type,
        position={"x": 0, "y": 0},
        parent_id=parent_id,
    )


def test_graph_requires_unique_ids_and_valid_endpoints() -> None:
    request = StudioGraphUpdate(
        revision=0,
        nodes=[node("a"), node("b")],
        edges=[GraphEdge(id="edge", source="a", target="b")],
    )
    validate_graph(request)

    with pytest.raises(AppError) as error:
        validate_graph(
            StudioGraphUpdate(
                revision=0,
                nodes=[node("a")],
                edges=[GraphEdge(id="edge", source="a", target="missing")],
            )
        )
    assert error.value.code == "invalid_edge_endpoint"


def test_graph_validates_group_references_without_moving_nodes() -> None:
    request = StudioGraphUpdate(
        revision=3,
        nodes=[node("group", node_type="group"), node("pump", parent_id="group")],
        edges=[],
    )
    validate_graph(request)
    assert request.nodes[1].position == {"x": 0.0, "y": 0.0}


def test_graph_rejects_non_group_parents_and_group_cycles() -> None:
    with pytest.raises(AppError) as wrong_type:
        validate_graph(
            StudioGraphUpdate(
                revision=0,
                nodes=[node("pump"), node("sensor", parent_id="pump")],
                edges=[],
            )
        )
    assert wrong_type.value.code == "invalid_parent_type"

    with pytest.raises(AppError) as cycle:
        validate_graph(
            StudioGraphUpdate(
                revision=0,
                nodes=[
                    node("a", parent_id="b", node_type="group"),
                    node("b", parent_id="a", node_type="group"),
                ],
                edges=[],
            )
        )
    assert cycle.value.code == "cyclic_node_parent"


def test_category_documents_support_grouped_and_flat_collections() -> None:
    categories = _normalize_categories(
        [
            {
                "parent": "Chiller",
                "children": [{"label": "Centrifugal chiller", "value": "centrifugal"}],
            },
            {
                "root_category": "Pump",
                "category": "primary_pump",
                "category_cn": "Primary pump",
            },
        ]
    )

    assert [(group.parent, group.children[0].value) for group in categories] == [
        ("Chiller", "centrifugal"),
        ("Pump", "primary_pump"),
    ]


def test_graph_rejects_source_to_source_handle_connections() -> None:
    with pytest.raises(AppError) as error:
        validate_graph(
            StudioGraphUpdate(
                revision=0,
                nodes=[node("a"), node("b")],
                edges=[
                    GraphEdge(
                        id="invalid-handles",
                        source="a",
                        target="b",
                        data={"sourceHandle": "source-0", "targetHandle": "source-0"},
                    )
                ],
            )
        )

    assert error.value.code == "invalid_edge_handle"


def test_equipment_inspection_grade_defaults_to_b_and_is_validated() -> None:
    equipment = node("pump")

    assert equipment.data["inspection"] == {"grade": "B", "enabled": True}

    with pytest.raises(ValidationError):
        GraphNode(
            id="invalid",
            type="equipment",
            position={"x": 0, "y": 0},
            data={"inspection": {"grade": "D"}},
        )


def test_engineering_parameters_follow_device_schema() -> None:
    schema = EngineeringParameterSchema(
        device_type="chiller",
        label="Chiller",
        version=1,
        parameters=[
            EngineeringParameterDefinition(
                key="q_cool_rated",
                label="Rated cooling capacity",
                unit="kW",
                required=True,
                exclusive_minimum=0,
            ),
            EngineeringParameterDefinition(
                key="load_pct_min",
                label="Minimum load ratio",
                unit="1",
                required=True,
                minimum=0,
                maximum=1,
                less_than_or_equal_to="load_pct_max",
            ),
            EngineeringParameterDefinition(
                key="load_pct_max",
                label="Maximum load ratio",
                unit="1",
                required=True,
                minimum=0,
                maximum=1,
            ),
        ],
    )
    equipment = GraphNode(
        id="chiller-1",
        type="chiller",
        position={"x": 0, "y": 0},
        data={
            "engineering_parameters": {
                "q_cool_rated": 1_200,
                "load_pct_min": 0.2,
                "load_pct_max": 1,
            }
        },
    )

    validate_engineering_parameters([equipment], [schema])


def test_engineering_parameters_reject_missing_bounds_and_unknown_fields() -> None:
    schema = EngineeringParameterSchema(
        device_type="pump",
        label="Pump",
        version=1,
        parameters=[
            EngineeringParameterDefinition(
                key="freq_min",
                label="Minimum frequency",
                unit="Hz",
                required=True,
                minimum=0,
                less_than_or_equal_to="freq_max",
            ),
            EngineeringParameterDefinition(
                key="freq_max",
                label="Maximum frequency",
                unit="Hz",
                required=True,
                exclusive_minimum=0,
            ),
        ],
    )
    equipment = GraphNode(
        id="pump-1",
        type="pump",
        position={"x": 0, "y": 0},
        data={
            "engineering_parameters": {
                "freq_min": 50,
                "freq_max": 40,
                "legacy_frequency": 45,
            }
        },
    )

    with pytest.raises(AppError) as error:
        validate_engineering_parameters([equipment], [schema])

    assert error.value.code == "engineering_parameters_invalid"
    issues = error.value.details["nodes"][0]["issues"]
    assert "Unknown fields: legacy_frequency" in issues
    assert "freq_min must not exceed freq_max" in issues


@pytest.mark.parametrize("device_type", ["chiller", "pump", "cooling_tower"])
def test_engineering_parameter_resources_match_database_schema(device_type: str) -> None:
    document = json.loads((RESOURCE_ROOT / f"{device_type}.json").read_text(encoding="utf-8"))

    schema = _engineering_schema(document)

    assert schema.device_type == device_type
    assert schema.parameters
    assert all(parameter.unit for parameter in schema.parameters)
