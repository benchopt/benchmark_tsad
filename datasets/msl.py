from benchopt import BaseDataset, safe_import_context, config

with safe_import_context() as import_ctx:
    import pathlib
    import numpy as np
    import requests

# Create global variables to store the urls
URL_XTRAIN = (
    "https://drive.google.com/uc?&id="
    "1PMzjODVFblVnwq8xo7pKHrdbczPxdqTa&export=download"
)

URL_XTEST = (
    "https://drive.google.com/uc?&id="
    "1OcNc0YQsOMw9jQIIHgiOXVG03wjXbEiM&export=download"
)

URL_YTEST = (
    "https://drive.google.com/uc?&id="
    "19vR0QvKluuiIT2H5mCFNIJh6xGVwshDd&export=download"
)


class Dataset(BaseDataset):
    name = "MSL"

    install_cmd = "conda"
    requirements = ["pandas", "requests", "pathlib"]

    parameters = {
        "debug": [False],
    }

    def get_data(self):
        # Adding get_data_path method soon
        train_path = config.get_data_path(key="msl_train")
        test_path = config.get_data_path(key="msl_test")
        test_label_path = config.get_data_path(key="msl_test_label")

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

        print(X_train.shape, X_test.shape, y_test.shape)

        return dict(
            X_train=X_train, y_test=y_test, X_test=X_test
        )
