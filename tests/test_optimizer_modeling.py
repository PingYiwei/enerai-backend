import pytest

from app.modules.optimizer.modeling import train_model


def pump_rows() -> list[dict[str, float]]:
    return [
        {"flow": float(flow), "head": 30.0 + 2.0 * flow + 0.5 * flow * flow}
        for flow in range(1, 21)
    ]


def test_polynomial_model_fits_quadratic_pump_curve() -> None:
    result = train_model("polynomial", "pump", pump_rows())
    assert result.metrics["r2"] == pytest.approx(1.0)
    assert result.metrics["rmse"] < 1e-6
    assert result.artifact["target_name"] == "head"


def test_gradient_boosting_produces_real_predictions_and_metrics() -> None:
    result = train_model("gradient_boosting", "pump", pump_rows())
    assert len(result.artifact["stumps"]) > 0
    assert len(result.predictions) == 20
    assert result.metrics["r2"] > 0.8
