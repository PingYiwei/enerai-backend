from app.modules.inspections.service import inspect_graph


def test_empty_graph_returns_actionable_finding() -> None:
    findings = inspect_graph({"nodes": [], "edges": []})
    assert [finding.code for finding in findings] == ["graph_empty"]


def test_graph_reports_isolated_and_unmapped_sensor_nodes() -> None:
    findings = inspect_graph(
        {
            "nodes": [
                {"id": "pump", "type": "pump", "data": {}},
                {"id": "sensor", "type": "sensor", "data": {}},
            ],
            "edges": [],
        }
    )
    assert {finding.code for finding in findings} == {
        "isolated_equipment",
        "sensor_property_missing",
    }
    assert findings[0].node_ids == ["pump", "sensor"]
