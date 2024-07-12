from benchopt import BaseObjective, safe_import_context
from benchmark_utils.metrics import soft_precision as soft_precision_score
from benchmark_utils.metrics import soft_recall as soft_recall_score

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
        return dict(y_hat=np.ones(self.X_test.shape[0]))

    def set_data(self, X_train, y_test, X_test):
        "Set the data to compute the objective."
        self.X_train = X_train
        self.X_test, self.y_test = X_test, y_test

    def evaluate_result(self, y_hat):
        "Evaluate the result provided by the solver."

        to_discard = (y_hat == -1).sum()
        self.y_test = self.y_test[to_discard:]
        y_hat = y_hat[to_discard:]

        detection_range = 1

        precision = precision_score(self.y_test, y_hat, zero_division=0)
        recall = recall_score(self.y_test, y_hat, zero_division=0)
        f1 = f1_score(self.y_test, y_hat, zero_division=0)
        zoloss = zero_one_loss(self.y_test, y_hat)
        soft_precision = soft_precision_score(
            self.y_test, y_hat, detection_range=detection_range
            )
        soft_recall = soft_recall_score(
            self.y_test, y_hat, detection_range=detection_range
            )

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "soft_precision": soft_precision,
            "soft_recall": soft_recall,
            "zoloss": zoloss,
            "value": zoloss,  # having zoloss twice because of the API
        }

    def get_objective(self):
        return dict(
            X_train=self.X_train, y_test=None, X_test=self.X_test
        )
