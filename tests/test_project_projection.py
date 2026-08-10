import io
import zipfile
from xml.etree import ElementTree

from app.modules.projects.data import (
    _property_catalog_items,
    _property_query_params,
    point_scheme,
    point_scheme_csv,
    point_scheme_xlsx,
    project_rdf,
)


def project() -> dict[str, object]:
    return {
        "_id": "prj_1",
        "name": 'Plant "A"',
        "nodes": [
            {
                "id": "chiller-1",
                "type": "chiller",
                "data": {
                    "label": "CH-1",
                    "root_category": "Chiller",
                    "category": "Centrifugal_Chiller",
                    "sensors": [
                        {
                            "id": "sensor-1",
                            "name": "Supply temperature",
                            "category": "Temperature_Sensor",
                            "category_cn": "Supply temperature",
                        }
                    ],
                },
            },
            {"id": "pump-1", "type": "pump", "data": {"label": "Pump 1"}},
        ],
        "edges": [{"source": "pump-1", "target": "chiller-1"}],
    }


def test_point_scheme_generates_category_properties_and_sensors() -> None:
    scheme = point_scheme(
        project(),
        [
            {
                "root_category": "Chiller",
                "properties": [
                    {
                        "name": "rated_cooling_capacity",
                        "cn_name": "Rated cooling capacity",
                        "unit": "kW",
                        "data_type": "number",
                        "min_value": 0,
                        "is_inherent": True,
                    },
                    {
                        "name": "coefficient_of_performance",
                        "cn_name": "COP",
                        "data_type": "number",
                        "is_inherent": False,
                    },
                ],
            }
        ],
    )
    assert scheme.total == 3
    assert scheme.inherent[0].point_name == "CH-1-rcc"
    assert scheme.calculate[0].property_name == "coefficient_of_performance"
    assert scheme.sensor[0].device_name == "CH-1"
    exported = point_scheme_csv(scheme)
    assert b"section,point_name,device_name,property_name" in exported
    assert b"inherent" in exported
    assert b"sensor" in exported

    workbook = point_scheme_xlsx(scheme)
    assert workbook.startswith(b"PK")
    with zipfile.ZipFile(io.BytesIO(workbook)) as archive:
        worksheet = ElementTree.fromstring(archive.read("xl/worksheets/sheet1.xml"))
        styles = ElementTree.fromstring(archive.read("xl/styles.xml"))

    namespace = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    columns = worksheet.findall("x:cols/x:col", namespace)
    assert len(columns) == 8
    assert all(float(column.attrib["width"]) >= 12 for column in columns)

    cells = {
        cell.attrib["r"]: cell
        for cell in worksheet.findall("x:sheetData/x:row/x:c", namespace)
    }
    assert cells["A1"].attrib["s"] == "1"
    assert cells["H2"].attrib == {"r": "H2", "s": "2", "t": "inlineStr"}
    assert styles.find("x:fonts/x:font[2]/x:b", namespace) is not None
    header_fill = styles.find("x:fills/x:fill[3]/x:patternFill/x:fgColor", namespace)
    assert header_fill is not None
    assert header_fill.attrib["rgb"] == "FF4472C4"
    text_style = styles.find("x:cellXfs/x:xf[3]", namespace)
    assert text_style is not None
    assert text_style.attrib["numFmtId"] == "49"


def test_rdf_projection_escapes_labels_and_preserves_connections() -> None:
    rdf = project_rdf(project())
    assert "@prefix brick:" in rdf
    assert "@prefix rdf:" in rdf
    assert "@prefix rdfs:" in rdf
    assert "@prefix enerai: <https://enerai.ai/projects/Plant_%22A%22#>" in rdf
    assert 'enerai:project a brick:Building ;\n    rdfs:label "Plant \\"A\\"" .' in rdf
    assert "enerai:CH-1 a brick:Centrifugal_Chiller ;" in rdf
    assert "    brick:hasPoint enerai:Supply_temperature ;" in rdf
    assert "enerai:Supply_temperature a brick:Temperature_Sensor ;" in rdf
    assert "    brick:isPointOf enerai:CH-1 ." in rdf
    assert "enerai:Pump_1 a brick:pump ;" in rdf
    assert "    brick:feed enerai:CH-1 ." in rdf


def test_property_catalog_uses_node_names_and_flattens_device_response() -> None:
    params = _property_query_params(project())
    assert params == {"device_ids": "CH-1,Pump 1"}

    items = _property_catalog_items(
        {
            "code": 200,
            "message": "success",
            "data": {
                "devices": [
                    {
                        "device_id": "CH-1",
                        "properties": [
                            {"name": "temperature", "data_type": "number", "unit": "°C"},
                            "status",
                        ],
                    }
                ]
            },
        }
    )
    assert items == [
        {"name": "temperature", "data_type": "number", "unit": "°C", "device_id": "CH-1"},
        {"name": "status", "device_id": "CH-1"},
    ]
