import pytest

from app.modules.optimizer.modeling import predict_model, train_model


def pump_rows() -> list[dict[str, float]]:
    return [
        {
            "flow": float(flow),
            "eff_pump": 0.55 + 0.03 * flow - 0.0008 * flow * flow,
            "head": 30.0 + 2.0 * flow + 0.5 * flow * flow,
        }
        for flow in range(1, 21)
    ]


def test_pump_model_trains_two_polynomial_submodels_and_predicts() -> None:
    result = train_model("polynomial", "pump", pump_rows())
    assert [item["key"] for item in result.series] == ["flow_efficiency", "flow_head"]
    assert all(item["metrics"]["r2"] == pytest.approx(1.0) for item in result.series)
    outputs = predict_model("pump", result.artifact, {"flow": 10.0})
    assert outputs["eff_pump"] == pytest.approx(0.77)
    assert outputs["head"] == pytest.approx(100.0)


def test_cooling_tower_model_uses_wet_bulb_and_efficiency() -> None:
    rows = []
    for index in range(16):
        t_wb = 20.0 + index // 4 * 3.0
        eta = 0.4 + index % 4 * 0.1
        t_cw_in = t_wb + 10.0
        rows.append({
            "t_cw_in": t_cw_in,
            "t_cw_out": t_cw_in - eta * 10.0,
            "t_wb_wea": t_wb,
            "air_water_ratio": 0.7 + 0.02 * t_wb + 0.5 * eta,
        })
    result = train_model("polynomial", "cooling_tower", rows)
    outputs = predict_model(
        "cooling_tower", result.artifact, {"t_wb": 25.0, "eta": 0.55}
    )
    assert outputs["air_water_ratio"] == pytest.approx(1.475, abs=1e-5)


def test_chiller_model_builds_submodels_and_chained_evaluations() -> None:
    rows = [
        {
            "t_chw_ret": 12.0 + index * 0.02,
            "t_chw_sup": 7.0,
            "t_cw_sup": 28.0,
            "t_cw_ret": 32.0 + index * 0.01,
            "flow_chw": 600.0 + index * 8.0,
            "flow_cw": 720.0 + index * 9.0,
            "q_cool": 500.0 + index * 25.0,
            "q_reject": 620.0 + index * 29.0,
            "load_pct": 35.0 + index * 2.0,
            "t_evap": 4.5 + index * 0.01,
            "t_cond": 36.0 + index * 0.03,
        }
        for index in range(20)
    ]
    result = train_model("polynomial", "chiller", rows)
    assert [item["kind"] for item in result.series[:3]] == [
        "submodel",
        "submodel",
        "submodel",
    ]
    assert len([item for item in result.series if item["kind"] == "evaluation"]) == 4
    outputs = predict_model("chiller", result.artifact, rows[10])
    assert outputs["cop"] > 0
    assert outputs["power"] > 0
