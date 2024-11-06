from benchopt import BaseDataset, safe_import_context
from benchopt.config import get_data_path

with safe_import_context() as import_ctx:
    import pandas as pd

URL_XTRAIN = "./data/WADI/WADI_14days_new.csv"
URL_XTEST = "./data/WADI/WADI_attackdataLABLE.csv"


class Dataset(BaseDataset):
    name = "WADI"

    install_cmd = "conda"
    requirements = ["pandas"]

    parameters = {
        "debug": [False],
        "n_splits": [5],
        "validation_size": [0.2],
    }

    def get_data(self):
        path = get_data_path(key="WADI")

        if not (path / "WADI_14days_new.csv").exists():
            raise FileNotFoundError(
                "Train data not found. Please download the data "
                f"from the official repository and place it in {path}"
            )

        if not (path / "WADI_attackdataLABLE.csv").exists():
            raise FileNotFoundError(
                "Test data not found. Please download the data "
                f"from the official repository and place it in {path}"
            )

        # Load the data
        X_train = pd.read_csv(path / "WADI_14days_new.csv")
        X_test = pd.read_csv(path / "WADI_attackdataLABLE.csv", header=1)

        # Data processing
        todrop = [
            "2_LS_001_AL",
            "2_LS_002_AL",
            "2_P_001_STATUS",
            "2_P_002_STATUS",
            "Date",
            "Time",
        ]
        X_train.columns = X_train.columns.str.strip()
        X_train.drop(columns=todrop, inplace=True)
        X_train.ffill(inplace=True)
        X_train = X_train.to_numpy()

        # Extract the target
        X_test.columns = X_test.columns.str.strip()
        y_test = X_test["Attack LABLE (1:No Attack, -1:Attack)"].values
        X_test.drop(
            columns=todrop + [
                     "Attack LABLE (1:No Attack, -1:Attack)"],
            inplace=True
        )
        X_test.ffill(inplace=True)
        X_test = X_test.to_numpy()

        # Limiting the size of the dataset for testing purposes
        if self.debug:
            X_train = X_train[:1000]
            X_test = X_test[:1000]
            y_test = y_test[:1000]

        return dict(
            X_train=X_train, y_test=y_test, X_test=X_test
        )
