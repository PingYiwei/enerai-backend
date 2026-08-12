from __future__ import annotations

import csv
import io
import re
import time
import zipfile
from datetime import UTC, datetime
from typing import Any
from urllib.parse import quote, urljoin, urlparse

import httpx
from gridfs import AsyncGridFSBucket
from pymongo.asynchronous.database import AsyncDatabase

from app.core.errors import AppError
from app.core.object_storage import get_minio_storage
from app.core.security import Principal
from app.modules.projects.schemas import (
    DataQuery,
    DataQueryResult,
    DataSourceTestResult,
    DataSourceUpdate,
    DataSourceView,
    PointScheme,
    PropertyCatalog,
    PropertyPoint,
    SensorPoint,
)

Document = dict[str, Any]


async def owned_project(
    database: AsyncDatabase[Document], principal: Principal, project_id: str
) -> Document:
    document = await database.projects.find_one({"_id": project_id, "owner_id": principal.user_id})
    if document is None:
        raise AppError("project_not_found", "Project was not found", status_code=404)
    return document


def _validate_base_url(value: str) -> str:
    url = value.strip().rstrip("/")
    if urlparse(url).scheme not in {"http", "https"}:
        raise AppError(
            "invalid_data_source_url",
            "Data source URL must use http or https",
            status_code=422,
        )
    return url


def _view(config: Document) -> DataSourceView:
    return DataSourceView(
        base_url=config["base_url"],
        properties_path=config["properties_path"],
        query_path=config["query_path"],
        token_present=bool(config.get("bearer_token")),
        updated_at=config["updated_at"],
    )


async def save_data_source(
    database: AsyncDatabase[Document],
    principal: Principal,
    project_id: str,
    request: DataSourceUpdate,
) -> DataSourceView:
    project = await owned_project(database, principal, project_id)
    previous = project.get("data_source", {})
    token = request.bearer_token
    if token is None and isinstance(previous, dict):
        token = previous.get("bearer_token")
    config: Document = {
        "base_url": _validate_base_url(request.base_url),
        "properties_path": request.properties_path.strip(),
        "query_path": request.query_path.strip(),
        "bearer_token": token.strip() if isinstance(token, str) else None,
        "updated_at": datetime.now(UTC),
    }
    await database.projects.update_one(
        {"_id": project_id, "owner_id": principal.user_id},
        {"$set": {"data_source": config, "updated_at": config["updated_at"]}},
    )
    return _view(config)


async def get_data_source(
    database: AsyncDatabase[Document], principal: Principal, project_id: str
) -> DataSourceView:
    project = await owned_project(database, principal, project_id)
    return _view(_configured_source(project))


def _configured_source(project: Document) -> Document:
    source = project.get("data_source")
    if not isinstance(source, dict):
        raise AppError(
            "data_source_not_configured",
            "Configure the project data source first",
            status_code=409,
        )
    return source


def _headers(source: Document) -> dict[str, str]:
    token = source.get("bearer_token")
    return {"Authorization": f"Bearer {token}"} if token else {}


def _endpoint(source: Document, path_key: str) -> str:
    return urljoin(f"{source['base_url'].rstrip('/')}/", str(source[path_key]).lstrip("/"))


async def test_data_source(
    database: AsyncDatabase[Document], principal: Principal, project_id: str
) -> DataSourceTestResult:
    project = await owned_project(database, principal, project_id)
    source = _configured_source(project)
    requirements = await _node_property_requirements(database, project)
    endpoint = _endpoint(source, "properties_path")
    started = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            response = await client.get(
                endpoint,
                headers=_headers(source),
                params=_property_query_params(project),
            )
    except httpx.HTTPError as error:
        raise AppError("data_source_unreachable", str(error), status_code=502) from error
    elapsed_ms = round((time.perf_counter() - started) * 1_000)
    payload: Any = None
    response_error = ""
    if response.is_success:
        try:
            payload = response.json()
        except ValueError:
            response_error = "Property endpoint did not return valid JSON"
    else:
        response_error = f"Property endpoint returned HTTP {response.status_code}"

    items_value = _property_catalog_items(payload) if payload is not None else []
    items = items_value if isinstance(items_value, list) else []
    catalog_items = [item for item in items if isinstance(item, dict)]
    provided_by_device = _device_property_map(payload)
    nodes: list[Document] = []
    for requirement in requirements:
        required = set(requirement["required_properties"])
        provided = provided_by_device.get(requirement["device_id"])
        if response_error:
            status, status_text, message = "failed", "Test failed", response_error
            provided_names: list[str] = []
            missing = sorted(required)
        elif provided is None:
            status, status_text = "failed", "Device unavailable"
            message = "The property response did not include this node"
            provided_names = []
            missing = sorted(required)
        else:
            provided_names = sorted(provided)
            missing = sorted(required - provided)
            if not required:
                status, status_text = "failed", "Model incomplete"
                message = "No required properties are configured for this node category"
            elif not missing:
                status, status_text = "complete", "Data complete"
                message = "All properties required by the current point scheme are available"
            elif required & provided:
                status, status_text = "partial", "Partially connected"
                message = "Only some properties required by the current point scheme are available"
            else:
                status, status_text = "failed", "No required data"
                message = "None of the required properties are available"
        nodes.append(
            {
                **requirement,
                "provided_properties": provided_names,
                "missing_properties": missing,
                "status": status,
                "status_text": status_text,
                "message": message,
            }
        )

    healthy_count = sum(node["status"] in {"complete", "partial"} for node in nodes)
    if nodes and all(node["status"] == "complete" for node in nodes):
        overall_status, overall_status_text = "complete", "Data connection complete"
    elif healthy_count:
        overall_status, overall_status_text = "partial", "Data connection partially complete"
    else:
        overall_status, overall_status_text = "failed", "Data connection failed"
    return DataSourceTestResult(
        ok=response.is_success and not response_error,
        status_code=response.status_code,
        elapsed_ms=elapsed_ms,
        project_id=project_id,
        endpoint=endpoint,
        overall_status=overall_status,
        overall_status_text=overall_status_text,
        node_count=len(nodes),
        completed_node_count=healthy_count,
        nodes=nodes,
        items=catalog_items,
    )


async def properties(
    database: AsyncDatabase[Document],
    principal: Principal,
    project_id: str,
    device_ids: list[str] | None = None,
) -> PropertyCatalog:
    project = await owned_project(database, principal, project_id)
    source = _configured_source(project)
    payload = await _request_json(
        "GET",
        _endpoint(source, "properties_path"),
        _headers(source),
        params=_property_query_params(project, device_ids),
    )
    items = _property_catalog_items(payload)
    if not isinstance(items, list) or not all(isinstance(item, dict) for item in items):
        raise AppError(
            "invalid_data_source_response",
            "Property endpoint must return a list or an object with an items list",
            status_code=502,
        )
    return PropertyCatalog(items=items, total=len(items))


def _property_query_params(
    project: Document, device_ids: list[str] | None = None
) -> dict[str, str]:
    node_names: list[str] = []
    candidates = device_ids if device_ids is not None else _project_device_ids(project)
    for device_id in candidates:
        name = str(device_id).strip()
        if name and name not in node_names:
            node_names.append(name)
    return {"device_ids": ",".join(node_names)} if node_names else {}


def _project_device_ids(project: Document) -> list[str]:
    node_names: list[str] = []
    for node in project.get("nodes", []):
        if node.get("type") == "group":
            continue
        data = node.get("data")
        if not isinstance(data, dict):
            continue
        name = str(data.get("name") or data.get("label") or "").strip()
        if name and name not in node_names:
            node_names.append(name)
    return node_names


def _property_catalog_items(payload: Any) -> Any:
    if not isinstance(payload, dict):
        return payload
    data = payload.get("data")
    devices = data.get("devices") if isinstance(data, dict) else None
    if isinstance(devices, list):
        items: list[Document] = []
        for device in devices:
            if not isinstance(device, dict):
                continue
            device_id = str(device.get("device_id") or "")
            properties_value = device.get("properties")
            if not isinstance(properties_value, list):
                continue
            for property_value in properties_value:
                item = (
                    dict(property_value)
                    if isinstance(property_value, dict)
                    else {"name": property_value}
                )
                item["device_id"] = device_id
                items.append(item)
        return items
    return payload.get("items", payload)


def _device_property_map(payload: Any) -> dict[str, set[str]]:
    result: dict[str, set[str]] = {}
    if isinstance(payload, dict):
        data = payload.get("data")
        devices = data.get("devices") if isinstance(data, dict) else None
        if not isinstance(devices, list):
            devices = payload.get("devices")
        if isinstance(devices, list):
            for device in devices:
                if not isinstance(device, dict):
                    continue
                device_id = str(device.get("device_id") or "").strip()
                if device_id:
                    result[device_id] = _property_names(device.get("properties"))
            return result

    items = _property_catalog_items(payload)
    if isinstance(items, list):
        for item in items:
            if not isinstance(item, dict):
                continue
            device_id = str(item.get("device_id") or "").strip()
            property_name = str(item.get("name") or item.get("property_id") or "").strip()
            if device_id and property_name:
                result.setdefault(device_id, set()).add(property_name)
    return result


def _property_names(value: Any) -> set[str]:
    if not isinstance(value, list):
        return set()
    names: set[str] = set()
    for item in value:
        name = item.get("name") or item.get("property_id") if isinstance(item, dict) else item
        if isinstance(name, str) and name.strip():
            names.add(name.strip())
    return names


async def _node_property_requirements(
    database: AsyncDatabase[Document], project: Document
) -> list[Document]:
    root_categories = {
        str(data["root_category"])
        for node in project.get("nodes", [])
        if node.get("type") != "group"
        and isinstance((data := node.get("data")), dict)
        and data.get("root_category")
    }
    documents = (
        await database.properties.find({"root_category": {"$in": list(root_categories)}}).to_list(
            None
        )
        if root_categories
        else []
    )
    properties_by_category = {
        str(document.get("root_category")): sorted(_property_names(document.get("properties")))
        for document in documents
    }
    requirements: list[Document] = []
    for node in project.get("nodes", []):
        if node.get("type") == "group":
            continue
        data = node.get("data")
        if not isinstance(data, dict):
            continue
        node_id = str(node.get("id") or "").strip()
        node_name = str(data.get("name") or data.get("label") or node_id).strip()
        if not node_id or not node_name:
            continue
        requirements.append(
            {
                "node_id": node_id,
                "node_name": node_name,
                "device_id": node_name,
                "required_properties": properties_by_category.get(
                    str(data.get("root_category") or ""), []
                ),
            }
        )
    return requirements


async def query_data(
    database: AsyncDatabase[Document],
    principal: Principal,
    project_id: str,
    request: DataQuery,
) -> DataQueryResult:
    if request.start_time.tzinfo is None or request.end_time.tzinfo is None:
        raise AppError(
            "invalid_time_range",
            "start_time and end_time must include a timezone",
            status_code=422,
        )
    if request.end_time <= request.start_time:
        raise AppError("invalid_time_range", "end_time must be after start_time", status_code=422)
    project = await owned_project(database, principal, project_id)
    source = _configured_source(project)
    payload = await _request_json(
        "POST",
        _endpoint(source, "query_path"),
        _headers(source),
        json={
            "device_id": request.device_id,
            "start_time": request.start_time.isoformat(),
            "end_time": request.end_time.isoformat(),
            **({"properties": request.properties} if request.properties else {}),
        },
    )
    return DataQueryResult(data=payload)


async def _request_json(method: str, url: str, headers: dict[str, str], **kwargs: Any) -> Any:
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.request(method, url, headers=headers, **kwargs)
            response.raise_for_status()
            return response.json()
    except (httpx.HTTPError, ValueError) as error:
        raise AppError("data_source_request_failed", str(error), status_code=502) from error


def point_scheme(
    project: Document, property_documents: list[Document] | None = None
) -> PointScheme:
    inherent: list[PropertyPoint] = []
    calculate: list[PropertyPoint] = []
    sensors: list[SensorPoint] = []
    properties_by_category = {
        str(document.get("root_category")): document.get("properties", [])
        for document in property_documents or []
        if document.get("root_category") and isinstance(document.get("properties"), list)
    }
    for node in project.get("nodes", []):
        data = node.get("data", {})
        if not isinstance(data, dict):
            continue
        device_name = str(data.get("name") or data.get("label") or node.get("id") or "")
        for sensor in data.get("sensors", []):
            if not isinstance(sensor, dict):
                continue
            sensors.append(
                SensorPoint(
                    sensor_name=str(sensor.get("name") or sensor.get("id") or ""),
                    device_name=device_name,
                    category=str(sensor.get("category") or ""),
                    category_cn=str(sensor.get("category_cn") or ""),
                    description=str(sensor.get("description") or sensor.get("note") or ""),
                )
            )
        root_category = str(data.get("root_category") or "")
        for property_document in properties_by_category.get(root_category, []):
            if not isinstance(property_document, dict) or not property_document.get("name"):
                continue
            property_name = str(property_document["name"])
            item = PropertyPoint(
                point_name=f"{device_name}-{_brief_name(property_name)}",
                device_name=device_name,
                property_name=property_name,
                property_name_cn=str(property_document.get("cn_name") or ""),
                unit=str(property_document.get("unit") or ""),
                data_type=str(property_document.get("data_type") or ""),
                range=_format_range(
                    property_document.get("min_value"), property_document.get("max_value")
                ),
            )
            (inherent if property_document.get("is_inherent") else calculate).append(item)
    return PointScheme(
        inherent=inherent,
        calculate=calculate,
        sensor=sensors,
        total=len(inherent) + len(calculate) + len(sensors),
    )


async def project_point_scheme(database: AsyncDatabase[Document], project: Document) -> PointScheme:
    root_categories = {
        str(data["root_category"])
        for node in project.get("nodes", [])
        if isinstance((data := node.get("data")), dict) and data.get("root_category")
    }
    property_documents = (
        await database.properties.find({"root_category": {"$in": list(root_categories)}}).to_list(
            None
        )
        if root_categories
        else []
    )
    return point_scheme(project, property_documents)


def _point_scheme_rows(scheme: PointScheme) -> list[list[str]]:
    rows = [
        [
            "section",
            "point_name",
            "device_name",
            "property_name",
            "property_name_cn",
            "unit",
            "data_type",
            "range",
        ]
    ]
    for section, items in (("inherent", scheme.inherent), ("calculate", scheme.calculate)):
        for item in items:
            rows.append(
                [
                    section,
                    item.point_name,
                    item.device_name,
                    item.property_name,
                    item.property_name_cn,
                    item.unit,
                    item.data_type,
                    item.range,
                ]
            )
    for sensor in scheme.sensor:
        rows.append(
            [
                "sensor",
                sensor.sensor_name,
                sensor.device_name,
                sensor.category,
                sensor.category_cn,
                "",
                "",
                sensor.description,
            ]
        )
    return rows


def point_scheme_csv(scheme: PointScheme) -> bytes:
    target = io.StringIO(newline="")
    writer = csv.writer(target)
    writer.writerows(_point_scheme_rows(scheme))
    return target.getvalue().encode("utf-8-sig")


def _excel_column_name(index: int) -> str:
    result = ""
    while index:
        index, remainder = divmod(index - 1, 26)
        result = chr(65 + remainder) + result
    return result


def _xml_text(value: str) -> str:
    return (
        value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
    )


def _display_width(value: str) -> int:
    return sum(2 if ord(character) > 255 else 1 for character in value)


def point_scheme_xlsx(scheme: PointScheme) -> bytes:
    rows = _point_scheme_rows(scheme)
    widths = [
        min(60, max(12, max(_display_width(row[index]) for row in rows) + 2))
        for index in range(len(rows[0]))
    ]
    columns = "".join(
        f'<col min="{index}" max="{index}" width="{width}" customWidth="1"/>'
        for index, width in enumerate(widths, start=1)
    )
    sheet_rows: list[str] = []
    for row_index, row in enumerate(rows, start=1):
        cells = []
        for column_index, value in enumerate(row, start=1):
            reference = f"{_excel_column_name(column_index)}{row_index}"
            style = 1 if row_index == 1 else 2 if column_index == 8 else 0
            cells.append(
                f'<c r="{reference}" s="{style}" t="inlineStr">'
                f'<is><t xml:space="preserve">{_xml_text(value)}</t></is></c>'
            )
        sheet_rows.append(f'<row r="{row_index}">{"".join(cells)}</row>')

    worksheet = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        '<sheetViews><sheetView workbookViewId="0">'
        '<pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/>'
        "</sheetView></sheetViews>"
        f"<cols>{columns}</cols><sheetData>{''.join(sheet_rows)}</sheetData>"
        f'<autoFilter ref="A1:H{len(rows)}"/></worksheet>'
    )
    styles = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        '<fonts count="2"><font><sz val="11"/><name val="Calibri"/></font>'
        '<font><b/><color rgb="FFFFFFFF"/><sz val="11"/><name val="Calibri"/></font></fonts>'
        '<fills count="3"><fill><patternFill patternType="none"/></fill>'
        '<fill><patternFill patternType="gray125"/></fill>'
        '<fill><patternFill patternType="solid"><fgColor rgb="FF4472C4"/>'
        '<bgColor indexed="64"/></patternFill></fill></fills>'
        '<borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>'
        '<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/>'
        "</cellStyleXfs>"
        '<cellXfs count="3">'
        '<xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>'
        '<xf numFmtId="0" fontId="1" fillId="2" borderId="0" xfId="0" applyFont="1" '
        'applyFill="1" applyAlignment="1"><alignment horizontal="center" vertical="center"/></xf>'
        '<xf numFmtId="49" fontId="0" fillId="0" borderId="0" xfId="0" applyNumberFormat="1"/>'
        '</cellXfs><cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/>'
        "</cellStyles></styleSheet>"
    )
    content_types = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" '
        'ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/xl/workbook.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        '<Override PartName="/xl/worksheets/sheet1.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        '<Override PartName="/xl/styles.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>'
        "</Types>"
    )
    root_relationships = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="xl/workbook.xml"/></Relationships>'
    )
    workbook = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        '<sheets><sheet name="Point Scheme" sheetId="1" r:id="rId1"/></sheets></workbook>'
    )
    workbook_relationships = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
        'Target="worksheets/sheet1.xml"/>'
        '<Relationship Id="rId2" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" '
        'Target="styles.xml"/></Relationships>'
    )

    target = io.BytesIO()
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types)
        archive.writestr("_rels/.rels", root_relationships)
        archive.writestr("xl/workbook.xml", workbook)
        archive.writestr("xl/_rels/workbook.xml.rels", workbook_relationships)
        archive.writestr("xl/worksheets/sheet1.xml", worksheet)
        archive.writestr("xl/styles.xml", styles)
    return target.getvalue()


def project_rdf(project: Document) -> str:
    project_name = str(project.get("name") or project.get("_id") or "project").strip()
    namespace_name = quote(re.sub(r"\s+", "_", project_name), safe="._-~") or "project"
    prefixes = [
        "@prefix brick: <https://brickschema.org/schema/Brick#> .",
        f"@prefix enerai: <https://enerai.ai/projects/{namespace_name}#> .",
        "@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .",
        "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .",
    ]
    statements: dict[str, list[tuple[str, str]]] = {}

    def add(subject: str, predicate: str, value: str) -> None:
        statements.setdefault(subject, []).append((predicate, value))

    add("enerai:project", "a", "brick:Building")
    add("enerai:project", "rdfs:label", f'"{_turtle(project_name)}"')
    node_by_id = {str(node.get("id")): node for node in project.get("nodes", [])}
    for node in project.get("nodes", []):
        data = node.get("data", {}) if isinstance(node.get("data"), dict) else {}
        name = str(data.get("label") or data.get("name") or node.get("id") or "")
        if not name:
            continue
        node_ref = f"enerai:{_uri_part(name)}"
        category = _brick_class(node, data)
        add(node_ref, "a", f"brick:{category}")
        add(node_ref, "rdfs:label", f'"{_turtle(name)}"')
        description = data.get("description") or data.get("note")
        if description:
            add(node_ref, "rdfs:comment", f'"{_turtle(description)}"')
        for sensor in data.get("sensors", []):
            if not isinstance(sensor, dict):
                continue
            sensor_name = str(sensor.get("name") or sensor.get("id") or "")
            if not sensor_name:
                continue
            sensor_ref = f"enerai:{_uri_part(sensor_name)}"
            sensor_class = _class_part(sensor.get("category") or "Sensor")
            add(sensor_ref, "a", f"brick:{sensor_class}")
            add(sensor_ref, "rdfs:label", f'"{_turtle(sensor_name)}"')
            if sensor.get("id"):
                add(sensor_ref, "enerai:sourceId", f'"{_turtle(sensor["id"])}"')
            sensor_description = sensor.get("description") or sensor.get("note")
            if sensor_description:
                add(sensor_ref, "rdfs:comment", f'"{_turtle(sensor_description)}"')
            add(sensor_ref, "brick:isPointOf", node_ref)
            add(node_ref, "brick:hasPoint", sensor_ref)
        child_ids = data.get("child") if isinstance(data.get("child"), list) else []
        if child_ids is None:
            continue
        for child_id in child_ids:
            child = node_by_id.get(str(child_id))
            if not child:
                continue
            child_data = child.get("data", {}) if isinstance(child.get("data"), dict) else {}
            child_name = str(child_data.get("label") or child_data.get("name") or child_id)
            child_ref = f"enerai:{_uri_part(child_name)}"
            add(node_ref, "brick:hasPart", child_ref)
            add(child_ref, "brick:isPartOf", node_ref)
    for edge in project.get("edges", []):
        source = node_by_id.get(str(edge.get("source")))
        target = node_by_id.get(str(edge.get("target")))
        if not source or not target:
            continue
        source_data = source.get("data", {}) if isinstance(source.get("data"), dict) else {}
        target_data = target.get("data", {}) if isinstance(target.get("data"), dict) else {}
        source_name = source_data.get("label") or source_data.get("name") or source.get("id")
        target_name = target_data.get("label") or target_data.get("name") or target.get("id")
        source_ref = f"enerai:{_uri_part(source_name)}"
        target_ref = f"enerai:{_uri_part(target_name)}"
        add(source_ref, "brick:feed", target_ref)
        add(target_ref, "brick:isFedBy", source_ref)

    blocks = [_turtle_subject(subject, predicates) for subject, predicates in statements.items()]
    return "\n".join(prefixes) + "\n\n" + "\n\n".join(blocks) + "\n"


def _turtle_subject(subject: str, predicates: list[tuple[str, str]]) -> str:
    priority = {"a": 0, "rdfs:label": 1, "rdfs:comment": 2}
    ordered = [
        item
        for _, item in sorted(
            enumerate(predicates),
            key=lambda entry: (priority.get(entry[1][0], 3), entry[0]),
        )
    ]
    lines: list[str] = []
    for index, (predicate, value) in enumerate(ordered):
        prefix = f"{subject} " if index == 0 else "    "
        terminator = " ." if index == len(ordered) - 1 else " ;"
        lines.append(f"{prefix}{predicate} {value}{terminator}")
    return "\n".join(lines)


def _brief_name(value: str) -> str:
    return "".join(word[0] for word in value.split("_") if word)


def _format_range(minimum: Any, maximum: Any) -> str:
    if minimum is None and maximum is None:
        return ""
    if minimum is None:
        return f"≤ {maximum}"
    if maximum is None:
        return f"≥ {minimum}"
    return f"{minimum} - {maximum}"


def _class_part(value: Any) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_]+", "_", str(value).strip()).strip("_")
    return normalized or "Equipment"


def _brick_class(node: Document, data: Document) -> str:
    if node.get("type") == "group":
        return {"area": "Area", "space": "Space", "group": "Group"}.get(
            str(data.get("category") or "").lower(), "Group"
        )
    return _class_part(data.get("category") or data.get("root_category") or node.get("type"))


def _uri_part(value: Any) -> str:
    return quote(re.sub(r"\s+", "_", str(value).strip()), safe="._-~")


def _turtle(value: Any) -> str:
    return (
        str(value)
        .replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\r", "\\r")
        .replace("\n", "\\n")
    )


async def cleanup_project_resources(
    database: AsyncDatabase[Document], owner_id: str, project_id: str
) -> None:
    datasets = await database.datasets.find(
        {"project_id": project_id, "owner_id": owner_id}, {"file_id": 1}
    ).to_list(None)
    bucket = AsyncGridFSBucket(database, bucket_name="optimizer_files")
    for dataset in datasets:
        if dataset.get("file_id") is not None:
            await bucket.delete(dataset["file_id"])
    attachments = await database.chat_attachments.find(
        {"project_id": project_id, "owner_id": owner_id},
        {"file_id": 1, "storage_bucket": 1, "object_name": 1},
    ).to_list(None)
    chat_bucket = AsyncGridFSBucket(database, bucket_name="chat_files")
    for attachment in attachments:
        if attachment.get("object_name"):
            await get_minio_storage().delete_object(
                bucket=attachment.get("storage_bucket"),
                object_name=attachment["object_name"],
            )
        elif attachment.get("file_id") is not None:
            await chat_bucket.delete(attachment["file_id"])
    await database.chat_attachments.delete_many({"project_id": project_id, "owner_id": owner_id})
    artifacts = await database.artifacts.find(
        {"project_id": project_id, "owner_id": owner_id},
        {"file_id": 1, "storage_bucket": 1, "object_name": 1},
    ).to_list(None)
    artifact_bucket = AsyncGridFSBucket(database, bucket_name="agent_artifacts")
    for artifact in artifacts:
        if artifact.get("object_name"):
            await get_minio_storage().delete_object(
                bucket=artifact.get("storage_bucket"),
                object_name=artifact["object_name"],
            )
        elif artifact.get("file_id") is not None:
            await artifact_bucket.delete(artifact["file_id"])
    await database.artifacts.delete_many({"project_id": project_id, "owner_id": owner_id})

    sessions = await database.agent_sessions.find(
        {"project_id": project_id, "owner_id": owner_id}, {"_id": 1}
    ).to_list(None)
    session_ids = [document["_id"] for document in sessions]
    operations = await database.agent_operations.find(
        {"project_id": project_id, "owner_id": owner_id}, {"_id": 1}
    ).to_list(None)
    run_ids = [document["_id"] for document in operations]
    if run_ids:
        await database.agent_events.delete_many({"run_id": {"$in": run_ids}})
        await database.agent_records.delete_many({"operation_id": {"$in": run_ids}})
    if session_ids:
        await database.agent_entries.delete_many({"session_id": {"$in": session_ids}})
        await database.agent_lanes.delete_many({"session_id": {"$in": session_ids}})
    await database.agent_operations.delete_many({"project_id": project_id, "owner_id": owner_id})
    await database.agent_sessions.delete_many({"project_id": project_id, "owner_id": owner_id})
    for collection in (
        database.studio_graph_versions,
        database.inspection_policies,
        database.inspection_runs,
        database.models,
        database.datasets,
    ):
        await collection.delete_many({"project_id": project_id, "owner_id": owner_id})
