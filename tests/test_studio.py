import pytest

from app.core.errors import AppError
from app.modules.studio.schemas import GraphEdge, GraphNode, StudioGraphUpdate
from app.modules.studio.service import validate_graph


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
