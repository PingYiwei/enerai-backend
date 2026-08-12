from __future__ import annotations

from app.modules.inspections.schemas import InspectionTemplate

TEMPLATES: dict[str, InspectionTemplate] = {
    "full_inspection": InspectionTemplate(
        id="full_inspection",
        version=1,
        name="Full operational inspection",
        description=(
            "Review every enabled Reality Model device for operating condition, anomalies, "
            "efficiency, optimization opportunities, and data-quality exceptions."
        ),
        default_minimum_grade="C",
        objectives=[
            "operating_condition",
            "anomaly_detection",
            "efficiency",
            "optimization",
            "data_completeness",
            "data_freshness",
            "missingness",
        ],
    ),
    "critical_equipment": InspectionTemplate(
        id="critical_equipment",
        version=1,
        name="Critical equipment inspection",
        description=(
            "Review devices at the selected grade or above and use their RDF relationships "
            "when investigating abnormal operation."
        ),
        default_minimum_grade="A",
        objectives=[
            "operating_condition",
            "anomaly_detection",
            "efficiency",
            "optimization",
            "data_completeness",
            "data_freshness",
            "missingness",
        ],
    ),
}


def template(template_id: str) -> InspectionTemplate:
    return TEMPLATES.get(template_id, TEMPLATES["full_inspection"])
