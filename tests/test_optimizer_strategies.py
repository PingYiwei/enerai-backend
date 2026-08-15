from datetime import UTC, datetime

import pytest

from app.modules.optimizer.schemas import OptimizationRange, OptimizationStrategySummary
from app.modules.optimizer.strategies import (
    _average_value,
    _chiller_combinations,
    _range_values,
    _supply_values,
)


def test_range_values_are_inclusive_and_stable_for_decimal_steps() -> None:
    values = _range_values(OptimizationRange(minimum=0.2, maximum=0.5, step=0.1))
    assert values == [0.2, 0.3, 0.4, 0.5]


def test_fixed_supply_temperature_produces_one_external_dimension_value() -> None:
    now = datetime.now(UTC)
    strategy = OptimizationStrategySummary.model_validate({
        "id": "optstr_1",
        "project_id": "prj_1",
        "name": "Fixed supply",
        "status": "draft",
        "revision": 1,
        "search_space": {
            "chilled_water_supply": {
                "enabled": False,
                "minimum": 5,
                "maximum": 9,
                "step": 1,
                "fixed": 6.5,
            }
        },
        "solver": {},
        "created_at": now,
        "updated_at": now,
    })
    assert _supply_values(strategy) == [6.5]


def test_representative_model_averages_matching_coefficient_positions() -> None:
    artifact = _average_value([
        {"kind": "pump_polynomial", "coefficients": {"head": {"intercept": 10.0}}},
        {"kind": "pump_polynomial", "coefficients": {"head": {"intercept": 14.0}}},
    ])
    assert artifact["kind"] == "pump_polynomial"
    assert artifact["coefficients"]["head"]["intercept"] == pytest.approx(12.0)


def test_chiller_combination_generator_keeps_two_nearest_feasible_options() -> None:
    groups = [
        {
            "group_id": "large",
            "available_count": 3,
            "parameters": {
                "q_cool_rated": 500.0,
                "coef_q_cool_rated_corr_a": 0.0,
                "coef_q_cool_rated_corr_b": 0.0,
                "coef_q_cool_rated_corr_c": 0.0,
                "coef_q_cool_rated_corr_d": 0.0,
                "coef_q_cool_rated_corr_e": 0.0,
                "coef_q_cool_rated_corr_f": 1.0,
            },
        }
    ]
    combinations = _chiller_combinations(groups, wet_bulb=25, supply=7, load=700)
    assert [item[0]["large"] for item in combinations] == [2, 3]


def test_second_chiller_combination_has_more_capacity_than_first() -> None:
    correction = {
        "coef_q_cool_rated_corr_a": 0.0,
        "coef_q_cool_rated_corr_b": 0.0,
        "coef_q_cool_rated_corr_c": 0.0,
        "coef_q_cool_rated_corr_d": 0.0,
        "coef_q_cool_rated_corr_e": 0.0,
        "coef_q_cool_rated_corr_f": 1.0,
    }
    groups = [
        {
            "group_id": "large",
            "available_count": 2,
            "parameters": {"q_cool_rated": 500.0, **correction},
        },
        {
            "group_id": "small",
            "available_count": 3,
            "parameters": {"q_cool_rated": 300.0, **correction},
        },
    ]
    combinations = _chiller_combinations(groups, wet_bulb=25, supply=7, load=850)
    capacities = [
        sum(counts[group["group_id"]] * rated[group["group_id"]] for group in groups)
        for counts, rated in combinations
    ]
    assert capacities == [1000.0, 1100.0]
