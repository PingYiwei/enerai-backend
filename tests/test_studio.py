import pytest
from pydantic import ValidationError

from app.core.errors import AppError
from app.modules.studio.schemas import GraphEdge, GraphNode, StudioGraphUpdate
from app.modules.studio.service import _normalize_categories, validate_graph


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
