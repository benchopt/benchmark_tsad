from benchopt.base import BaseDataset, safe_import_context

with safe_import_context() as import_ctx:
    # import module to generate normal 1d data
    from sklearn.datasets import make_regression


class Dataset(BaseDataset):
    name = "Simulated"
    is_classification = False
    n_samples = 1000
    n_features = 5
    noise = 0.1

    def get_data(self):
        X, y = make_regression(
            n_samples=self.n_samples,
            n_features=self.n_features,
            noise=self.noise,
        )
        return dict(X=X, y=y)
