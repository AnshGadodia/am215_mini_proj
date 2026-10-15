import numpy as np

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
