from benchopt import BaseSolver
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    from sktime.annotation.lof import SubLOF


class Solver(BaseSolver):
    name = "SubLOF"

    install_cmd = "conda"
    requirements = ["sktime"]

    parameters = {
        "n_neighbors": [5, 10, 20, 25, 40],
        "window_size": [20, 64, 128],
        "leaf_size": [30, 40],
        "contamination": ["auto", 0.1, 0.2, 0.3],
    }

    sampling_strategy = "run_once"

    def set_objective(self, X_train, y_test, X_test):
        self.X_train = X_train
        self.X_test, self.y_test = X_test, y_test
        self.clf = SubLOF(
            n_neighbors=self.n_neighbors,
            window_size=self.window_size,
            leaf_size=self.leaf_size,
            contamination=self.contamination,
            n_jobs=-1,
        )

    def run(self, _):
        self.clf.fit(self.X_train)
        self.raw_y_hat = self.clf.predict(self.X_test)
        self.raw_anomaly_score = self.clf.predict_score(self.X_test)

    def skip(self, X_train, y_test, X_test):
        if self.n_neighbors > self.window_size:
            return True, "Number of neighbors greater than window size"
        if self.n_neighbors > X_train.shape[0]:
            return True, "Number of neighbors greater than number of samples"
        if self.leaf_size > X_train.shape[0]:
            return True, "Leaf size greater than number of samples"
        if self.window_size > X_train.shape[0]:
            return True, "Window size greater than number of samples"
        return False, None

    def get_result(self):
        self.y_hat = self.raw_y_hat
        return dict(y_hat=self.y_hat)
