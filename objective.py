from benchopt import BaseObjective, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from sklearn.metrics import (
        precision_score, recall_score, f1_score, zero_one_loss
    )


class Objective(BaseObjective):
    name = "Anomaly detection"

    install_cmd = "conda"
    requirements = ["scikit-learn"]

    def get_one_result(self):
        """Return one solution for which the objective can be computed,
        Used to get the shape of the result.
        Our algorithms will return an array of labels of shape (n_samples,)
        """
        return np.zeros(self.X_test.shape[0])

    def set_data(self, X_train, y_test, X_test):
        "Set the data to compute the objective."
        self.X_train = X_train
        self.X_test, self.y_test = X_test, y_test

    def evaluate_result(self, y_hat):
        "Evaluate the result provided by the solver."
        precision = precision_score(self.y_test, y_hat, zero_division=0)
        recall = recall_score(self.y_test, y_hat, zero_division=0)
        f1 = f1_score(self.y_test, y_hat, zero_division=0)
        zoloss = zero_one_loss(self.y_test, y_hat)

        return {
            "value": zoloss,  # having zoloss twice because of the API
            "zoloss": zoloss,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    def get_objective(self):
        return dict(
            X_train=self.X_train, y_test=None, X_test=self.X_test
        )
