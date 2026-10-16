import numpy as np
import pandas as pd


def estimate_bernoulli_params_from_returns(
    returns: np.ndarray | pd.Series,
) -> tuple[float, float, float]:
    """
    Estimate Bernoulli random walk parameters (p, delta_up, delta_down)
    from a sequence of daily returns.

    Parameters:
    - returns: sequence or 1D array/Series of daily simple returns

    Returns:
    - (p, delta_up, delta_down):
        - p: fraction of positive daily returns (r > 0)
        - delta_up: mean return on positive days
        - delta_down: mean return on non-positive days (r <= 0)
    """
    if isinstance(returns, pd.Series):
        clean_returns = returns.dropna().to_numpy(dtype=float)
    else:
        clean_returns = np.asarray(returns, dtype=float)
        clean_returns = clean_returns[~np.isnan(clean_returns)]

    if len(clean_returns) == 0:
        raise ValueError("Returns sequence cannot be empty.")

    up_returns = clean_returns[clean_returns > 0]
    down_returns = clean_returns[clean_returns <= 0]

    p = float(len(up_returns) / len(clean_returns))
    delta_up = float(np.mean(up_returns)) if len(up_returns) > 0 else 0.0
    delta_down = float(np.mean(down_returns)) if len(down_returns) > 0 else 0.0

    return p, delta_up, delta_down


def estimate_bernoulli_params(
    prices: np.ndarray | pd.Series | list[float],
) -> tuple[float, float, float]:
    """
    Estimate Bernoulli random walk parameters (p, delta_up, delta_down)
    from a historical window of stock prices (e.g. 252 trading days).

    Parameters:
    - prices: sequence or 1D array/Series of historical daily prices

    Returns:
    - (p, delta_up, delta_down)
    """
    if isinstance(prices, pd.Series):
        clean_prices = prices.dropna().to_numpy(dtype=float)
    elif isinstance(prices, pd.DataFrame):
        if prices.shape[1] != 1:
            raise ValueError(
                f"Expected 1-dimensional price series, got DataFrame with {prices.shape[1]} columns"
            )
        clean_prices = prices.iloc[:, 0].dropna().to_numpy(dtype=float)
    else:
        clean_prices = np.asarray(prices, dtype=float)
        clean_prices = clean_prices[~np.isnan(clean_prices)]

    if len(clean_prices) < 2:
        raise ValueError(
            "At least 2 price observations are required to compute returns."
        )

    # Simple daily percentage returns: (P_t - P_{t-1}) / P_{t-1}
    returns = (clean_prices[1:] - clean_prices[:-1]) / clean_prices[:-1]
    return estimate_bernoulli_params_from_returns(returns)


def estimate_rolling_parameters(
    prices: pd.Series,
    window_size: int = 252,
    step_size: int = 21,
    forecast_horizon: int = 21,
    target_gain: float = 0.03,
) -> pd.DataFrame:
    """
    Compute rolling-window Bernoulli parameter estimates and realized forward returns.

    Parameters:
    - prices: pandas Series with DatetimeIndex of daily adjusted closing prices for a single stock.
    - window_size: number of return observations used for estimation (default: 252 trading days).
    - step_size: number of trading days to step forward between forecast dates (default: 21).
    - forecast_horizon: number of trading days forward to evaluate the outcome (default: 21).
    - target_gain: target fractional gain threshold (default: 0.03 for 3% gain).

    Returns:
    - pd.DataFrame containing for each rolling window:
        - forecast_date: timestamp of the final day in the estimation window (t0)
        - s0: price at forecast_date
        - p: estimated probability of positive daily return
        - delta_up: mean positive return
        - delta_down: mean non-positive return
        - eval_date: timestamp at forecast_horizon (if available)
        - s_eval: price at eval_date (if available)
        - actual_return: realized return over the horizon
        - realized_gain_target: boolean indicating whether actual_return >= target_gain
    """
    if not isinstance(prices, pd.Series):
        prices = pd.Series(prices)

    clean_prices = prices.dropna()
    n_obs = len(clean_prices)

    # Need at least window_size + 1 prices for the first estimation window
    min_required = window_size + 1
    if n_obs < min_required:
        raise ValueError(
            f"Need at least {min_required} price observations for window_size={window_size}, got {n_obs}."
        )

    records = []
    # i tracks the start index of the estimation window
    i = 0
    while i + window_size < n_obs:
        # Estimation window covers indices [i : i + window_size + 1] (window_size returns)
        hist_window = clean_prices.iloc[i : i + window_size + 1]
        forecast_idx = i + window_size
        forecast_date = clean_prices.index[forecast_idx]
        s0 = float(clean_prices.iloc[forecast_idx])

        p, delta_up, delta_down = estimate_bernoulli_params(hist_window)

        # Check forward evaluation horizon
        eval_idx = forecast_idx + forecast_horizon
        if eval_idx < n_obs:
            eval_date = clean_prices.index[eval_idx]
            s_eval = float(clean_prices.iloc[eval_idx])
            actual_return = (s_eval - s0) / s0
            realized_gain = bool(actual_return >= target_gain)
        else:
            eval_date = None
            s_eval = np.nan
            actual_return = np.nan
            realized_gain = None

        records.append({
            "forecast_date": forecast_date,
            "s0": s0,
            "p": p,
            "delta_up": delta_up,
            "delta_down": delta_down,
            "eval_date": eval_date,
            "s_eval": s_eval,
            "actual_return": actual_return,
            "realized_gain_target": realized_gain,
        })

        i += step_size

    return pd.DataFrame(records)


class BernoulliRandomWalk:
    """
    Bernoulli Random Walk model for daily stock returns

    Parameters:     
    - p = probability of positive daily return
    - delta_up = return on positive day 
    - delta_down = return on non-positive day
    """
    def __init__(self, p, delta_up, delta_down):
        self.p = p
        self.delta_up = delta_up
        self.delta_down = delta_down

    @classmethod
    def from_prices(cls, prices: np.ndarray | pd.Series | list[float]):
        """
        Fit a BernoulliRandomWalk model from a window of historical stock prices.

        Parameters:
        - prices: sequence or Series of daily prices (e.g. 252 trading days)

        Returns:
        - BernoulliRandomWalk instance initialized with estimated parameters
        """
        p, delta_up, delta_down = estimate_bernoulli_params(prices)
        return cls(p=p, delta_up=delta_up, delta_down=delta_down)

    @classmethod
    def from_returns(cls, returns: np.ndarray | pd.Series):
        """
        Fit a BernoulliRandomWalk model from a sequence of daily returns.

        Parameters:
        - returns: sequence or Series of daily returns

        Returns:
        - BernoulliRandomWalk instance initialized with estimated parameters
        """
        p, delta_up, delta_down = estimate_bernoulli_params_from_returns(returns)
        return cls(p=p, delta_up=delta_up, delta_down=delta_down)

    @property
    def p(self):
        return self._p

    @p.setter
    def p(self, value):
        if not 0 <= value <= 1:
            raise ValueError("p must be between 0 and 1")
        self._p = value

    def simulate(self, s0, n_days=21, n_paths=10000, rng=None):
        """
        Simulate 21 day stock price path using random walk model

        Parameters:
        - s0 = initial stock price
        - n_days = number of days to simulate 
        - n_paths = number of paths 
        - rng = NumPy random number generator (optional)

        Returns
        - paths = array of shape (n_days + 1, n_paths) containing
          simulated stock prices, including the initial price
        """

        if rng is None:
            rng = np.random.default_rng()
        
        moves = rng.binomial(n=1, p=self.p, size=(n_days, n_paths)) # Generate Bernoulli movements (1 = up, 0 = down)
        returns = np.where(moves, self.delta_up, self.delta_down) 
        paths = s0 * np.cumprod(1 + returns, axis=0)

        paths = np.vstack([np.full((1, n_paths), s0), paths]) # Append initial stock price as Day 0

        return paths

    def predict(self, s0, threshold=0.03, n_days=21, n_paths=10000, rng=None):
        """
        Estimate the probability of a stock gaining at least threshold over n_days trading days.

        Parameters:
        - s0 = initial stock price
        - threshold = target fractional gain
        - n_days = number of days to simulate 
        - n_paths = number of paths
        - rng = NumPy random number generator (optional)

        Returns
        - probability = Monte Carlo estimate of the probability that the final stock price is at least s0 * (1 + threshold)

        """
        paths = self.simulate(s0, n_days, n_paths, rng)

        final_prices = paths[-1, :] 
        probability = np.mean(final_prices >= s0 * (1 + threshold))

        return probability
