from benchopt import BaseDataset
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import os
    import pandas as pd


class Dataset(BaseDataset):
    name = "SMAP"

    install_cmd = "conda"
    requirements = ["pandas"]

    def get_data(self):

        path = "/storage/store/work/jyehya/Benchmarks/processing/processed/SMAP"
        dataset = "SMAP"

        X_train = pd.read_pickle(os.path.join(path, dataset + "_train.pkl"))
        X_test = pd.read_pickle(os.path.join(path, dataset + "_test.pkl"))
        y_test = pd.read_pickle(os.path.join(path, dataset + "_test_label.pkl"))

        return dict(X=X_train, y=y_test, X_test=X_test)
