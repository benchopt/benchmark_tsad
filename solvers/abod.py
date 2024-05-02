# ABOD solver

from benchopt import BaseSolver
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    from pyod.models.abod import ABOD


class Solver(BaseSolver):
    name = "ABOD"

    install_cmd = "pip"
    requirements = ["pyod"]

    parameters = {
        "contamination": [5e-4, 0.1, 0.2, 0.3],
        "n_neighbors": [10, 20, 30, 40],
    }

    def set_objective(self, X, y):
        self.X, self.y = X, y

    def run(self, _):
        clf = ABOD(
            n_neighbors=self.n_neighbors,
            contamination=self.contamination,
        )
        clf.fit(self.X)
        self.y_hat = clf.predict(self.X)

    def get_result(self):
        return {"y_hat": self.y_hat}
