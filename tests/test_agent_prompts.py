from datetime import UTC, datetime

from app.modules.agents.prompts import render_agent_system_prompt


def test_insight_prompt_contains_analysis_contract_and_runtime_clock() -> None:
    prompt = render_agent_system_prompt(
        "insight",
        now=datetime(2026, 8, 11, 4, 5, 6, tzinfo=UTC),
        timezone_name="Asia/Shanghai",
    )

    assert "EnerAI Insight Agent" in prompt
    assert "query_project_device_data" in prompt
    assert "explicitly references a node with `@`" in prompt
    assert "observed evidence" in prompt
    assert "2026-08-11T04:05:06+00:00" in prompt
    assert "2026-08-11T12:05:06+08:00" in prompt
    assert "Asia/Shanghai" in prompt


def test_studio_prompt_contains_atomic_graph_contract_and_runtime_clock() -> None:
    prompt = render_agent_system_prompt(
        "studio",
        now=datetime(2026, 8, 11, 23, 30, tzinfo=UTC),
        timezone_name="Asia/Shanghai",
    )
    normalized = " ".join(prompt.split())

    assert "EnerAI Studio Agent" in prompt
    assert "get_project_graph" in prompt
    assert "exactly one create, update, or delete tool call" in normalized
    assert "Preserve existing node positions" in prompt
    assert "2026-08-12T07:30:00+08:00" in prompt
    assert "Local calendar date: 2026-08-12 (Wednesday)" in prompt


def test_inspection_surface_uses_a_dedicated_read_only_profile() -> None:
    prompt = render_agent_system_prompt(
        "inspection",
        now=datetime(2026, 8, 11, tzinfo=UTC),
    )

    assert "EnerAI Auto-inspection Agent" in prompt
    assert "planned unavailable properties" in prompt
    assert "data freshness" in prompt
    assert "EnerAI Studio Agent" not in prompt
