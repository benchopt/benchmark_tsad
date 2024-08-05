from benchopt import BaseDataset, safe_import_context, config

with safe_import_context() as import_ctx:
    import pathlib
    import numpy as np
    import requests
    # from sklearn.model_selection import TimeSeriesSplit

URL_XTRAIN = (
    "https://drive.google.com/uc?&id=1e_JhpIURD"
    "Lluw4IcHJF-dgtjjJXsPEKE&export=download"
)
URL_XTEST = (
    "https://drive.google.com/uc?&id=10"
    "-r-Zm0nfQJp0i-mVg3iXs6x0u9Ua25a&export=download"
)
URL_YTEST = (
    "https://drive.google.com/uc?&id=1uYiXqmK3C"
    "gyxk4U6-LgUni7JddQnlggs&export=download"
)


class Dataset(BaseDataset):
    name = "SMAP"

    install_cmd = "conda"
    requirements = ["pandas", "scikit-learn"]

    parameters = {
        "debug": [False],
        "n_splits": [5],
        "validation_size": [0.2],
    }

    def get_data(self):
        train_path = config.get_data_path(key="smap_train")
        test_path = config.get_data_path(key="smap_test")
        test_label_path = config.get_data_path(key="smap_test_label")

        # Check if the data is already here
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

        X_train = np.load(train_path)
        X_test = np.load(test_path)
        y_test = np.load(test_label_path)

        # Limiting the size of the dataset for testing purposes
        if self.debug:
            X_train = X_train[:1000]
            X_test = X_test[:1000]
            y_test = y_test[:1000]

        return dict(
            X_train=X_train, y_test=y_test, X_test=X_test
        )
