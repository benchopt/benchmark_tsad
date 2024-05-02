# Isolation Forest solver

from benchopt import BaseSolver
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from sklearn.ensemble import IsolationForest


class Solver(BaseSolver):
    name = "IsolationForest"

    install_cmd = "pip"
    requirements = ["scikit-learn"]

    parameters = {
        "contamination": [5e-4, 5e-3, 5e-2, 0.1, 0.2, 0.3, 0.4, 0.5],
    }

    def set_objective(self, X, y):
        self.X, self.y = X, y

    def run(self, _):
        clf = IsolationForest(contamination=self.contamination)
        clf.fit(self.X)
        self.y_hat = clf.predict(self.X)
        # Map map({1: 0, -1: 1})
        self.y_hat[self.y_hat == 1] = 0
        self.y_hat[self.y_hat == -1] = 1

    def get_result(self):
        return {"y_hat": self.y_hat}
