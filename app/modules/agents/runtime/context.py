from __future__ import annotations

from typing import Any

from pymongo.asynchronous.database import AsyncDatabase

from app.core.errors import AppError
from app.core.security import Principal
from app.modules.agents.schemas import ContextOption, ContextOptions, ContextReference

Document = dict[str, Any]
SKILLS = {
    "energy-system-analysis": (
        "Energy system analysis",
        "Ground conclusions in project topology and operational evidence; label inferences.",
    ),
    "timeseries-data-analysis": (
        "Time-series data analysis",
        "Use bounded project queries, report time ranges, units, data gaps, and compact evidence.",
    ),
}


async def context_options(
    database: AsyncDatabase[Document], principal: Principal, project_id: str
) -> ContextOptions:
    project = await database.projects.find_one(
        {"_id": project_id, "owner_id": principal.user_id},
        {"name": 1, "nodes": 1},
    )
    if project is None:
        raise AppError("project_not_found", "Project was not found", status_code=404)
    items = [
        ContextOption(type="project", id=project_id, name=project["name"]),
        *[
            ContextOption(
                type="node",
                id=str(node["id"]),
                name=str(node.get("data", {}).get("label") or node["id"]),
                description=str(node.get("type", "equipment")),
            )
            for node in project.get("nodes", [])
        ],
        *[
            ContextOption(type="skill", id=skill_id, name=name, description=description)
            for skill_id, (name, description) in SKILLS.items()
        ],
    ]
    return ContextOptions(items=items)


async def validate_references(
    database: AsyncDatabase[Document],
    principal: Principal,
    project_id: str,
    references: list[ContextReference],
) -> list[ContextReference]:
    if not references:
        return []
    options = await context_options(database, principal, project_id)
    by_key = {(option.type, option.id): option for option in options.items}
    resolved: list[ContextReference] = []
    seen: set[tuple[str, str]] = set()
    for reference in references:
        key = (reference.type, reference.id)
        option = by_key.get(key)
        if option is None:
            raise AppError(
                "invalid_context_reference",
                "A referenced project, node, or skill is unavailable",
                status_code=422,
                details={"type": reference.type, "id": reference.id},
            )
        if key in seen:
            continue
        seen.add(key)
        resolved.append(ContextReference(type=option.type, id=option.id, name=option.name))
    return resolved


def contextual_content(content: str, references: list[ContextReference]) -> str:
    if not references:
        return content
    lines = [f"- {reference.type}: {reference.name} ({reference.id})" for reference in references]
    skill_guidance = [
        SKILLS[reference.id][1] for reference in references if reference.type == "skill"
    ]
    guidance = "\n".join(f"- {item}" for item in skill_guidance)
    suffix = f"\nSkill guidance:\n{guidance}" if guidance else ""
    return f"Referenced project context:\n{'\n'.join(lines)}{suffix}\n\nUser request:\n{content}"
