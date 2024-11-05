from benchopt import BaseDataset, safe_import_context

with safe_import_context() as import_ctx:
    import pandas as pd

# No URLs temporarily, local data is used
URL_XTRAIN = "./data/SWAT/swat_train2.csv"
URL_XTEST = "./data/SWAT/swat2.csv"


class Dataset(BaseDataset):
    name = "SWAT"

    install_cmd = "conda"
    requirements = ["pandas", "scikit-learn"]

    parameters = {
        "debug": [False],
        "n_splits": [5],
        "validation_size": [0.2],
    }

    def get_data(self):
        # path = config.get_data_path(key="SMAP")

        # Load the data
        X_train = pd.read_csv(URL_XTRAIN)
        X_test = pd.read_csv(URL_XTEST)

        # Extract the target
        y_test = X_test["Normal/Attack"].values
        X_test = X_test.drop(columns=["Normal/Attack"])
        X_test = X_test.to_numpy()

        X_train = X_train.drop(columns=["Normal/Attack"])
        X_train = X_train.to_numpy()

        # Limiting the size of the dataset for testing purposes
        if self.debug:
            X_train = X_train[:1000]
            X_test = X_test[:1000]
            y_test = y_test[:1000]

        return dict(
            X_train=X_train, y_test=y_test, X_test=X_test
        )
