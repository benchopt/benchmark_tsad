# Cluster Based Local Outlier Factor (CBLOF) solver

from benchopt import BaseSolver
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    from pyod.models.cblof import CBLOF
    import numpy as np


class Solver(BaseSolver):
    name = "CBLOF"

    install_cmd = "conda"
    requirements = ["pip:pyod"]

    parameters = {
        "contamination": [5e-4, 0.01, 0.02, 0.03, 0.04],
        "window": [True],
        "window_size": [20],
        "stride": [1],
    }

    sampling_strategy = "run_once"

    def set_objective(self, X_train, y_test, X_test):
        self.X_train = X_train
        self.X_test, self.y_test = X_test, y_test
        self.clf = CBLOF(contamination=self.contamination, n_clusters=8)

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
                # Decision function : Larger scores are outliers
                raw_anomaly_score.append(
                    self.clf.decision_function(self.X_test[i]))

            self.raw_y_hat = np.array(raw_y_hat)
            self.raw_anomaly_score = np.array(raw_anomaly_score)

    def get_result(self):
        # Anomaly : 1
        # Inlier : 0
        self.y_hat = np.append(
            self.raw_y_hat[0], self.raw_y_hat[1:, -1:])
        return {"y_hat": self.y_hat}
