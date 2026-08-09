from app.modules.agents.context import contextual_content
from app.modules.agents.schemas import ContextReference


def test_context_references_are_projected_without_changing_user_text() -> None:
    content = contextual_content(
        "Compare the last day.",
        [
            ContextReference(type="node", id="chiller-1", name="Chiller 1"),
            ContextReference(
                type="skill",
                id="timeseries-data-analysis",
                name="Time-series data analysis",
            ),
        ],
    )
    assert "node: Chiller 1 (chiller-1)" in content
    assert "report time ranges, units, data gaps" in content
    assert content.endswith("User request:\nCompare the last day.")
