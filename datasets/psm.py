from benchopt import BaseDataset, safe_import_context, config

with safe_import_context() as import_ctx:
    import requests
    import pandas as pd
    import pathlib

URL_XTRAIN = (
    "https://drive.google.com/uc?&id=1d3tAbYTj0CZLhB7z3IDTfTRg3E7qj_tw"
    "&export=download"
)
URL_XTEST = (
    "https://drive.google.com/uc?&id=1RQH7igHhm_0GAgXyVpkJk6TenDl9rd53"
    "&export=download"
)
URL_YTEST = (
    "https://drive.google.com/uc?&id=1SYgcRt0DH--byFbvkKTkezJKU5ZENZhw"
    "&export=download"
)


class Dataset(BaseDataset):
    name = "PSM"
    install_cmd = "conda"
    requirements = ["pandas", "pathlib"]
    parameters = {
        "debug": [False],
    }

    def get_data(self):
        # Check if the data is already here
        train_path = config.get_data_path(key="psm_train")
        test_path = config.get_data_path(key="psm_test")
        test_label_path = config.get_data_path(key="psm_test_label")

        if not pathlib.Path.exists(train_path):
            response = requests.get(URL_XTRAIN)
            with open(train_path, "wb") as f:
                f.write(response.content)
            response = requests.get(URL_XTEST)
            with open(test_path, "wb") as f:
                f.write(response.content)
            response = requests.get(URL_YTEST)
            with open(test_label_path, "wb") as f:
                f.write(response.content)

        X_train = pd.read_csv(train_path)
        X_train.fillna(X_train.mean(), inplace=True)
        X_train = X_train.to_numpy()

        X_test = pd.read_csv(test_path)
        X_test.fillna(X_test.mean(), inplace=True)
        X_test = X_test.to_numpy()

        y_test = pd.read_csv(test_label_path).to_numpy()[:, 1]

        # Limiting the size of the dataset for testing purposes
        if self.debug:
            X_train = X_train[:1000]
            X_test = X_test[:1000]
            y_test = y_test[:1000]

        return dict(
            X_train=X_train, y_test=y_test, X_test=X_test
        )
