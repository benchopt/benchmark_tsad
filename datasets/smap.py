from benchopt import BaseDataset, safe_import_context

with safe_import_context() as import_ctx:
    import os
    import numpy as np


class Dataset(BaseDataset):
    name = "SMAP"

    install_cmd = "conda"
    requirements = ["pandas"]

    def get_data(self):

        # Check if the data is already here
        if not os.path.exists("data/SMAP/SMAP_train.npy"):
            os.makedirs("data/SMAP", exist_ok=True)
            os.system(
                """
                wget -O data/SMAP/SMAP_train.npy "https://drive.google.com/uc?&id=1e_JhpIURDLluw4IcHJF-dgtjjJXsPEKE&export=download"
                wget -O data/SMAP/SMAP_test.npy "https://drive.google.com/uc?&id=10-r-Zm0nfQJp0i-mVg3iXs6x0u9Ua25a&export=download"
                wget -O data/SMAP/SMAP_test_label.npy "https://drive.google.com/uc?&id=1uYiXqmK3Cgyxk4U6-LgUni7JddQnlggs&export=download"
            """  # noqa
            )

        X_train = np.load("data/SMAP/SMAP_train.npy")
        X_test = np.load("data/SMAP/SMAP_test.npy")
        y_test = np.load("data/SMAP/SMAP_test_label.npy")

        print(X_train.shape, X_test.shape, y_test.shape)

        return dict(
            X=X_train, y=y_test, X_test=X_test
        )
