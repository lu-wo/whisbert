import numpy as np


class DummyModel:
    def __init__(self, target_values, nb_sig=3):
        self.target_values = target_values
        self.nb_sig = nb_sig
        self.lower, self.upper = self._compute_bounds()

    def _compute_bounds(self):
        max_val = max(self.target_values)
        min_val = min(self.target_values)
        std_dev = np.std(self.target_values)
        avg_val = np.mean(self.target_values)

        lower = max(0, avg_val - self.nb_sig * std_dev)
        upper = min(max_val, avg_val + self.nb_sig * std_dev)

        return lower, upper

    def predict(self, nb_predictions=1):
        if nb_predictions == 1:
            return np.random.choice(np.arange(self.lower, self.upper, 0.01))
        else:
            return np.random.choice(
                np.arange(self.lower, self.upper, 0.01), nb_predictions
            )

    def compute_mse(self, predictions):
        mse = np.mean((self.target_values - predictions) ** 2)
        return mse
