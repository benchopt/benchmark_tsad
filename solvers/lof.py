# Local Outlier Factor

from benchopt import BaseSolver
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    from sklearn.neighbors import LocalOutlierFactor


class Solver(BaseSolver):
    name = "LocalOutlierFactor"

    install_cmd = "conda"
    requirements = ["scikit-learn"]

    parameters = {
        "contamination": [5e-4, 0.1, 0.2, 0.3],
        "n_neighbors": [5, 10, 20, 25, 40],
    }

    sampling_strategy = "run_once"

    def set_objective(self, X, y):
        self.X, self.y = X, y

    def run(self, _):
        clf = LocalOutlierFactor(
            novelty=True,
            n_neighbors=self.n_neighbors,
            contamination=self.contamination,
        )
        clf.fit(self.X)
        self.y_hat = clf.predict(self.X)

    def get_result(self):
        self.y_hat[self.y_hat == 1] = 0
        self.y_hat[self.y_hat == -1] = 1

        return {"y_hat": self.y_hat}
