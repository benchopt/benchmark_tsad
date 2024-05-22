from benchopt import BaseDataset
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np


class Dataset(BaseDataset):

    name = "Glass"

    def get_data(self):
        data = np.load("datasets/14_glass.npz")
        X, y = data["X"], data["y"]
        return dict(X=X, y=y, X_test=None)
