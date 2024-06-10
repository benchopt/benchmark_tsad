from benchopt import BaseDataset, safe_import_context

with safe_import_context() as import_ctx:
    import os
    import numpy as np
    import requests


class Dataset(BaseDataset):
    name = "MSL"

    install_cmd = "conda"
    requirements = ["pandas", "requests"]

    def get_data(self):

        # Check if the data is already here
        if not os.path.exists("data/MSL/MSL_train.npy"):
            os.makedirs("data/MSL", exist_ok=True)
            url_xtrain = (
                "https://drive.google.com/uc?&id="
                "1PMzjODVFblVnwq8xo7pKHrdbczPxdqTa&export=download"
            )
            url_xtest = (
                "https://drive.google.com/uc?&id="
                "1OcNc0YQsOMw9jQIIHgiOXVG03wjXbEiM&export=download"
            )
            url_ytest = (
                "https://drive.google.com/uc?&id="
                "19vR0QvKluuiIT2H5mCFNIJh6xGVwshDd&export=download"
            )

            response = requests.get(url_xtrain)
            with open("data/MSL/MSL_train.npy", "wb") as f:
                f.write(response.content)

            response = requests.get(url_xtest)
            with open("data/MSL/MSL_test.npy", "wb") as f:
                f.write(response.content)

            response = requests.get(url_ytest)
            with open("data/MSL/MSL_test_label.npy", "wb") as f:
                f.write(response.content)

        X_train = np.load("data/MSL/MSL_train.npy")
        X_test = np.load("data/MSL/MSL_test.npy")
        y_test = np.load("data/MSL/MSL_test_label.npy")

        print(X_train.shape, X_test.shape, y_test.shape)

        X_train = np.lib.stride_tricks.sliding_window_view(
            X_train, window_shape=self.window_size, axis=0
        )[::self.stride]

        X_test = np.lib.stride_tricks.sliding_window_view(
            X_test, window_shape=self.window_size, axis=0
        )[::self.stride]

        y_test = np.lib.stride_tricks.sliding_window_view(
            y_test, window_shape=self.window_size, axis=0
        )[::self.stride]

        return dict(
            X=X_train, y=y_test, X_test=X_test
        )
