from benchopt import BaseDataset, safe_import_context

with safe_import_context() as import_ctx:
    import os
    import numpy as np


class Dataset(BaseDataset):
    name = "MSL"

    install_cmd = "conda"
    requirements = ["pandas"]

    def get_data(self):

        # Check if the data is already here
        if not os.path.exists("data/MSL/MSL_train.npy"):
            os.makedirs("data/MSL", exist_ok=True)
            os.system(
                """
                wget -O data/MSL/MSL_train.npy "https://drive.google.com/uc?&id=1PMzjODVFblVnwq8xo7pKHrdbczPxdqTa&export=download"
                wget -O data/MSL/MSL_test.npy "https://drive.google.com/uc?&id=1OcNc0YQsOMw9jQIIHgiOXVG03wjXbEiM&export=download"
                wget -O data/MSL/MSL_test_label.npy "https://drive.google.com/uc?&id=19vR0QvKluuiIT2H5mCFNIJh6xGVwshDd&export=download"
            """  # noqa
            )

        X_train = np.load("data/MSL/MSL_train.npy")
        X_test = np.load("data/MSL/MSL_test.npy")
        y_test = np.load("data/MSL/MSL_test_label.npy")

        print(X_train.shape, X_test.shape, y_test.shape)

        return dict(
            X=X_train, y=y_test, X_test=X_test
        )
