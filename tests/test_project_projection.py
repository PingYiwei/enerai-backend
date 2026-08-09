from app.modules.projects.data import point_scheme, point_scheme_csv, project_rdf


def project() -> dict[str, object]:
    return {
        "_id": "prj_1",
        "name": 'Plant "A"',
        "nodes": [
            {
                "id": "sensor-1",
                "type": "sensor",
                "data": {
                    "label": "Supply temperature",
                    "property": "chw.supply_temperature",
                    "unit": "°C",
                },
            },
            {"id": "pump-1", "type": "pump", "data": {"label": "Pump"}},
        ],
        "edges": [{"source": "pump-1", "target": "sensor-1"}],
    }


def test_point_scheme_only_contains_explicit_property_mappings() -> None:
    scheme = point_scheme(project())
    assert scheme.total == 1
    assert scheme.items[0].property == "chw.supply_temperature"
    assert b"node_id,node_name,node_type,property,unit" in point_scheme_csv(scheme)


def test_rdf_projection_escapes_labels_and_preserves_connections() -> None:
    rdf = project_rdf(project())
    assert 'nodex:name "Plant \\"A\\""' in rdf
    assert "pump-1 nodex:connectedTo" in rdf
