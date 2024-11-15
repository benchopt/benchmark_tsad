from benchopt import BaseDataset, safe_import_context

with safe_import_context() as import_ctx:
    from sklearn.datasets import make_regression
    import numpy as np


class Dataset(BaseDataset):
    name = "Simulated"

    install_cmd = "conda"
    requirements = ["scikit-learn"]

    parameters = {
        "n_samples": [10000],
        "n_features": [5],
        "noise": [0.1],
        "n_anomaly": [90],
    }

    test_parameters = {
        "n_samples": [500],
        "n_features": [5],
        "noise": [0.1],
        "n_anomaly": [90],
    }

    def get_data(self):
        X_train, _ = make_regression(
            n_samples=self.n_samples,
            n_features=self.n_features,
            noise=self.noise,
        )

        X_test, _ = make_regression(
            n_samples=self.n_samples,
            n_features=self.n_features,
            noise=self.noise,
        )

        assert X_test.shape == (self.n_samples, self.n_features)

        y_test = np.zeros(self.n_samples)
        for i in range(self.n_anomaly):
            idx = np.random.randint(self.n_samples)
            y_test[idx] = 1

        X_test = (
            X_test
            + np.random.randint(0, 2, (self.n_samples, self.n_features))
            * y_test[:, None]
            * 10
        )

        return dict(X_train=X_train, y_test=y_test, X_test=X_test)
