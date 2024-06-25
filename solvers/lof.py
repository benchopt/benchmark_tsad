# Local Outlier Factor

from benchopt import BaseSolver
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    from sklearn.neighbors import LocalOutlierFactor
    import numpy as np


class Solver(BaseSolver):
    name = "LocalOutlierFactor"

    install_cmd = "conda"
    requirements = ["scikit-learn"]

    parameters = {
        "contamination": [0.1, 0.2, 0.3],
        "n_neighbors": [5, 10, 20, 25, 40],
        "window": [True],
        "window_size": [20],
        "stride": [1],
    }

    sampling_strategy = "run_once"

    def set_objective(self, X_train, y_test, X_test):
        self.X_train = X_train
        self.X_test, self.y_test = X_test, y_test
        self.clf = LocalOutlierFactor(
            novelty=True,
            n_neighbors=self.n_neighbors,
            contamination=self.contamination,
        )

    def run(self, _):
        if self.window:
            # We need to transform the data to have a rolling window
            if self.X_train is not None:
                self.X_train = np.lib.stride_tricks.sliding_window_view(
                    self.X_train, window_shape=self.window_size, axis=0
                )[::self.stride].transpose(0, 2, 1)

            if self.X_test is not None:
                self.X_test = np.lib.stride_tricks.sliding_window_view(
                    self.X_test, window_shape=self.window_size, axis=0
                )[::self.stride].transpose(0, 2, 1)

            if self.y_test is not None:
                self.y_test = np.lib.stride_tricks.sliding_window_view(
                    self.y_test, window_shape=self.window_size, axis=0
                )[::self.stride]

            raw_y_hat = []
            raw_anomaly_score = []
            for i in range(len(self.X_test)):
                self.clf.fit(self.X_test[i])
                raw_y_hat.append(self.clf.predict(self.X_test[i]))
                # Decision function : Negative scores are outliers
                # We take the opposite to have larger scores as outliers
                raw_anomaly_score.append(
                    -self.clf.decision_function(self.X_test[i]))

            self.raw_y_hat = np.array(raw_y_hat)
            self.raw_anomaly_score = np.array(raw_anomaly_score)

    def skip(self, X_train, y_test, X_test):
        if self.n_neighbors > self.window_size:
            return True, "Number of neighbors greater than number of samples."
        return False, None

    def get_result(self):
        self.y_hat = np.append(
            self.raw_y_hat[0], self.raw_y_hat[1:, -1:])
        self.y_hat = np.where(self.y_hat == -1, 1, 0)
        return dict(y_hat=self.y_hat)
