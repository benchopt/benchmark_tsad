# Deep Isolation Forest
from benchopt import BaseSolver
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    from pyod.models.dif import DIF


class Solver(BaseSolver):
    name = "DIF"

    install_cmd = "conda"
    requirements = ["pyod"]

    parameters = {
        "contamination": [0.05, 0.1, 0.2],
    }

    sampling_strategy = "run_once"

    def set_objective(self, X, y, X_test=None):
        # y is y_test, the learning is unsupervised
        self.X = X
        self.X_test = X_test
        self.y = y

    def run(self, _):
        clf = DIF(contamination=self.contamination)
        clf.fit(self.X)
        self.y_hat = clf.predict(self.X_test)

    def get_result(self):
        return {"y_hat": self.y_hat}
