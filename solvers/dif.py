# Deep Isolation Forest
from benchopt import BaseSolver
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    from pyod.models.dif import DIF
    import numpy as np


class Solver(BaseSolver):
    name = "DIF"

    install_cmd = "conda"
    requirements = ["pyod"]

    parameters = {
        "contamination": [0.05, 0.1, 0.2],
        "window": [True],
        "window_size": [20],
        "stride": [1],
    }

    sampling_strategy = "run_once"

    def set_objective(self, X_train, y_test, X_test):
        self.X_train = X_train
        self.X_test, self.y_test = X_test, y_test

    def run(self, _):
        clf = DIF(contamination=self.contamination, device="cuda")
        # clf.fit(self.X)
        # self.y_hat = clf.predict(self.X_test)

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

    def skip(self, X_train, X_test, y_test):
        # If cuda is not available, we skip the test because deep method
        import torch
        if not torch.cuda.is_available():
            return True, "Cuda is not available"
        return False, None

    def get_result(self):
        # Anomaly : 1
        # Inlier : 0
        self.y_hat = np.append(
            self.raw_y_hat[0], self.raw_y_hat[1:, -1:])
        return {"y_hat": self.y_hat}
