# ABOD solver

from benchopt import BaseSolver
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    from pyod.models.abod import ABOD
    import numpy as np


class Solver(BaseSolver):
    name = "ABOD"

    install_cmd = "conda"
    requirements = ["pip:pyod"]

    parameters = {
        "contamination": [5e-4, 0.1, 0.2, 0.3],
        "n_neighbors": [5, 10, 15, 20, 30],
        "window": [True],
        "window_size": [20],
        "stride": [1],
    }

    sampling_strategy = "run_once"

    def set_objective(self, X_train, y_test, X_test):
        self.X_train = X_train
        self.X_test, self.y_test = X_test, y_test

    def run(self, _):
        clf = ABOD(
            n_neighbors=self.n_neighbors,
            contamination=self.contamination,
            method="fast"
        )
        # clf.fit(self.X)
        # self.y_hat = clf.predict(self.X)

        if self.window:
            # We need to transform the data to have a rolling window
            self.X_train = np.lib.stride_tricks.sliding_window_view(
                self.X_train, window_shape=self.window_size, axis=0
            )[::self.stride].transpose(0, 2, 1)

            self.X_test = np.lib.stride_tricks.sliding_window_view(
                self.X_test, window_shape=self.window_size, axis=0
            )[::self.stride].transpose(0, 2, 1)

            self.y_test = np.lib.stride_tricks.sliding_window_view(
                self.y_test, window_shape=self.window_size, axis=0
            )[::self.stride]

            raw_y_hat = []
            raw_anomaly_score = []
            for i in range(len(self.X_test)):
                clf.fit(self.X_test[i])
                raw_y_hat.append(clf.predict(self.X_test[i]))
                # Decision function : Larger scores are outliers
                raw_anomaly_score.append(
                    clf.decision_function(self.X_test[i]))

            self.raw_y_hat = np.array(raw_y_hat)
            self.raw_anomaly_score = np.array(raw_anomaly_score)

    def skip(self, X_train, X_test, y_test):
        if self.n_neighbors >= self.window_size:
            return True, "Number of neighbors greater than number of samples."
        return False, None

    def get_result(self):
        # Anomaly : 1
        # Inlier : 0
        self.y_hat = np.append(
            self.raw_y_hat[0], self.raw_y_hat[1:, -1:])
        return {"y_hat": self.y_hat}
