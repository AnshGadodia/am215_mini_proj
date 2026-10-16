import numpy as np
import pandas as pd
import pytest

from am215_mini_proj.models import (
    BernoulliRandomWalk,
    estimate_bernoulli_params,
    estimate_bernoulli_params_from_returns,
    estimate_rolling_parameters,
)


def test_estimate_bernoulli_params_from_returns():
    # 2 positive returns, 3 non-positive (including zero)
    returns = np.array([0.02, -0.01, 0.04, -0.03, 0.0])
    p, delta_up, delta_down = estimate_bernoulli_params_from_returns(returns)

    assert p == pytest.approx(2 / 5)
    assert delta_up == pytest.approx(0.03)
    assert delta_down == pytest.approx(-0.04 / 3)


def test_estimate_bernoulli_params_from_prices():
    prices = [100.0, 102.0, 100.98, 105.0192]
    # Daily returns: +0.02, -0.01, +0.04
    p, delta_up, delta_down = estimate_bernoulli_params(prices)

    assert p == pytest.approx(2 / 3)
    assert delta_up == pytest.approx(0.03)
    assert delta_down == pytest.approx(-0.01)


def test_bernoulli_random_walk_from_prices():
    prices = pd.Series([100.0, 102.0, 100.98, 105.0192])
    model = BernoulliRandomWalk.from_prices(prices)

    assert isinstance(model, BernoulliRandomWalk)
    assert model.p == pytest.approx(2 / 3)
    assert model.delta_up == pytest.approx(0.03)
    assert model.delta_down == pytest.approx(-0.01)

    # Verify simulate & predict function properly with fitted model
    paths = model.simulate(s0=100.0, n_days=5, n_paths=20)
    assert paths.shape == (6, 20)
    prob = model.predict(s0=100.0, threshold=0.03, n_days=5, n_paths=500)
    assert 0.0 <= prob <= 1.0


def test_estimate_params_validation():
    with pytest.raises(ValueError, match="At least 2 price observations"):
        estimate_bernoulli_params([100.0])

    with pytest.raises(ValueError, match="Returns sequence cannot be empty"):
        estimate_bernoulli_params_from_returns([])

    # All positive returns
    p, delta_up, delta_down = estimate_bernoulli_params_from_returns([0.01, 0.02])
    assert p == 1.0
    assert delta_up == pytest.approx(0.015)
    assert delta_down == 0.0

    # All negative returns
    p, delta_up, delta_down = estimate_bernoulli_params_from_returns([-0.01, -0.02])
    assert p == 0.0
    assert delta_up == 0.0
    assert delta_down == pytest.approx(-0.015)


def test_estimate_rolling_parameters():
    dates = pd.date_range("2024-01-01", periods=60, freq="B")
    prices = pd.Series(
        100.0 * (1.002 ** np.arange(60)),
        index=dates,
    )

    df = estimate_rolling_parameters(
        prices,
        window_size=20,
        step_size=10,
        forecast_horizon=5,
        target_gain=0.01,
    )

    expected_cols = {
        "forecast_date",
        "s0",
        "p",
        "delta_up",
        "delta_down",
        "eval_date",
        "s_eval",
        "actual_return",
        "realized_gain_target",
    }
    assert expected_cols.issubset(set(df.columns))
    assert len(df) > 0
    # Prices always increase here, so p should be 1.0
    assert (df["p"] == 1.0).all()

    # Test error if price history is shorter than window_size + 1
    short_prices = prices.iloc[:15]
    with pytest.raises(ValueError, match="Need at least"):
        estimate_rolling_parameters(short_prices, window_size=20)
