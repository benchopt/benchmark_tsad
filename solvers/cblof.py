# Cluster Based Local Outlier Factor (CBLOF) solver

from benchopt import BaseSolver
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    from pyod.models.cblof import CBLOF


class Solver(BaseSolver):
    name = "CBLOF"

    install_cmd = "pip"
    requirements = ["pyod"]

    parameters = {
        "contamination": [5e-4, 0.01, 0.02, 0.03, 0.04],
    }

    sampling_strategy = "run_once"

    def set_objective(self, X, y):
        self.X, self.y = X, y

    def run(self, _):
        clf = CBLOF(contamination=self.contamination)
        clf.fit(self.X)
        self.y_hat = clf.predict(self.X)

    def get_result(self):
        return {"y_hat": self.y_hat}
