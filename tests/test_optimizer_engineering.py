import pytest

from app.modules.optimizer.engineering import (
    _json_object,
    _model_groups,
    _node_states,
    _physical_properties,
    _water_system_values,
    infer_topologies,
)
from app.modules.studio.schemas import (
    EngineeringParameterDefinition,
    EngineeringParameterSchema,
)


def _node(
    node_id: str,
    node_type: str,
    label: str,
    *,
    parent_id: str | None = None,
    parameters: dict[str, float] | None = None,
    children: list[str] | None = None,
) -> dict[str, object]:
    data: dict[str, object] = {"label": label}
    if parameters is not None:
        data["engineering_parameters"] = parameters
    if children is not None:
        data["child"] = children
    result: dict[str, object] = {"id": node_id, "type": node_type, "data": data}
    if parent_id is not None:
        result["parent_id"] = parent_id
    return result


def _edge(source: str, target: str) -> dict[str, str]:
    return {"id": f"{source}-{target}", "source": source, "target": target}


def test_group_derivation_aggregates_node_engineering_parameters() -> None:
    schema = EngineeringParameterSchema(
        device_type="chiller",
        label="Chiller",
        version=1,
        parameters=[
            EngineeringParameterDefinition(
                key="q_cool_rated",
                label="Rated cooling capacity",
                label_zh="额定制冷量",
                unit="kW",
                required=True,
            )
        ],
    )
    nodes = [
        _node("group-a", "group", "Chiller model A", children=["ch-1", "ch-2"]),
        _node(
            "ch-1",
            "chiller",
            "CH-1",
            parent_id="group-a",
            parameters={
                "q_cool_rated": 1_000,
            },
        ),
        _node(
            "ch-2",
            "chiller",
            "CH-2",
            parent_id="group-a",
            parameters={
                "q_cool_rated": 1_200,
            },
        ),
    ]

    node_states = _node_states(nodes, [schema])
    groups = _model_groups(nodes, node_states)

    assert len(groups) == 1
    assert groups[0].member_node_ids == ["ch-1", "ch-2"]
    assert groups[0].complete is True
    assert groups[0].derived_values[0].value == pytest.approx(2_200)
    assert node_states[0].parameters[0].label == "Rated cooling capacity"


def test_topology_derivation_distinguishes_dedicated_and_shared_networks() -> None:
    nodes = [
        _node("ch-1", "chiller", "CH-1"),
        _node("ch-2", "chiller", "CH-2"),
        _node("chwp-1", "pump", "冷冻水泵 1"),
        _node("chwp-2", "pump", "冷冻水泵 2"),
        _node("cwp-1", "pump", "冷却水泵 1"),
        _node("cwp-2", "pump", "冷却水泵 2"),
        _node("header", "pipe", "冷却水共管"),
    ]
    edges = [
        _edge("chwp-1", "ch-1"),
        _edge("chwp-2", "ch-2"),
        _edge("cwp-1", "header"),
        _edge("cwp-2", "header"),
        _edge("header", "ch-1"),
        _edge("header", "ch-2"),
    ]

    by_system = {item.system: item for item in infer_topologies(nodes, edges)}

    assert by_system["chilled_water_pumps"].mode == "one_to_one"
    assert by_system["chilled_water_pumps"].confidence == pytest.approx(0.9)
    assert by_system["condenser_water_pumps"].mode == "parallel"
    assert all(
        connection.related_node_ids == ["ch-1", "ch-2"]
        for connection in by_system["condenser_water_pumps"].connections
    )


def test_parameter_layers_keep_physical_defaults_read_only_and_water_editable() -> None:
    physical = _physical_properties()
    water = {
        item.key: item
        for item in _water_system_values(
            {
                "dt_hdr_chw_fixed": 6.5,
                "coef_resistance_chw": 0.003,
            }
        )
    }

    assert physical
    assert all(item.source == "default" and item.editable is False for item in physical)
    assert water["dt_hdr_chw_fixed"].source == "user"
    assert water["dt_hdr_chw_fixed"].value == pytest.approx(6.5)
    assert water["dt_hdr_cw_default"].source == "default"
    assert water["coef_resistance_chw"].source == "user"
    assert water["coef_resistance_cw"].source == "missing"
    assert all(item.editable for item in water.values())


def test_llm_json_parser_accepts_fenced_json_only() -> None:
    assert _json_object('```json\n{"topologies": []}\n```') == {"topologies": []}
    with pytest.raises(ValueError):
        _json_object("The topology is parallel.")
