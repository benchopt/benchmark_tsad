from benchopt import BaseDataset, safe_import_context

with safe_import_context() as import_ctx:
    from sklearn.datasets import make_regression
    import numpy as np


class Dataset(BaseDataset):
    name = "Simulated"

    install_cmd = "conda"
    requirements = ["scikit-learn"]

    n_samples = 1000
    n_features = 5
    noise = 0.1
    n_anomaly = 90

    def get_data(self):
        x1, _ = make_regression(
            n_samples=self.n_samples,
            n_features=self.n_features,
            noise=self.noise,
        )

        assert x1.shape == (self.n_samples, self.n_features)

        y = np.zeros(self.n_samples)
        for i in range(self.n_anomaly):
            idx = np.random.randint(self.n_samples)
            y[idx] = 1

        x1 = (
            x1
            + np.random.randint(0, 2, (self.n_samples, self.n_features))
            * y[:, None]
            * 10
        )

        X = x1
        return dict(X=X, y=y)
