from benchopt import BaseObjective, safe_import_context
from benchmark_utils.metrics import (
    soft_precision as soft_precision_score,
    soft_recall as soft_recall_score,
    soft_f1 as soft_f1_score,
    ctt, ttc
)

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

        precision = precision_score(self.y_test, y_hat, zero_division=0)
        recall = recall_score(self.y_test, y_hat, zero_division=0)
        f1 = f1_score(self.y_test, y_hat, zero_division=0)
        zoloss = zero_one_loss(self.y_test, y_hat)

        soft_precision1 = soft_precision_score(
            self.y_test, y_hat, detection_range=1
        )
        soft_recall1 = soft_recall_score(
            self.y_test, y_hat, detection_range=1
        )
        soft_f1_1 = soft_f1_score(soft_precision1, soft_recall1)

        soft_precision3 = soft_precision_score(
            self.y_test, y_hat, detection_range=3
        )
        soft_recall3 = soft_recall_score(
            self.y_test, y_hat, detection_range=3
        )
        soft_f1_3 = soft_f1_score(soft_precision3, soft_recall3)

        soft_precision5 = soft_precision_score(
            self.y_test, y_hat, detection_range=5
        )
        soft_recall5 = soft_recall_score(
            self.y_test, y_hat, detection_range=5
        )
        soft_f1_5 = soft_f1_score(soft_precision5, soft_recall5)

        soft_precision10 = soft_precision_score(
            self.y_test, y_hat, detection_range=10
        )
        soft_recall10 = soft_recall_score(
            self.y_test, y_hat, detection_range=10
        )
        soft_f1_10 = soft_f1_score(soft_precision10, soft_recall10)

        soft_precision20 = soft_precision_score(
            self.y_test, y_hat, detection_range=20
        )
        soft_recall20 = soft_recall_score(
            self.y_test, y_hat, detection_range=20
        )
        soft_f1_20 = soft_f1_score(soft_precision20, soft_recall20)

        cct_score = ctt(self.y_test, y_hat)
        ttc_score = ttc(self.y_test, y_hat)

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,

            "soft_precision_1": soft_precision1,
            "soft_recall_1": soft_recall1,
            "soft_f1_1": soft_f1_1,

            "soft_precision_3": soft_precision3,
            "soft_recall_3": soft_recall3,
            "soft_f1_3": soft_f1_3,

            "soft_precision_5": soft_precision5,
            "soft_recall_5": soft_recall5,
            "soft_f1_5": soft_f1_5,

            "soft_precision_10": soft_precision10,
            "soft_recall_10": soft_recall10,
            "soft_f1_10": soft_f1_10,

            "soft_precision_20": soft_precision20,
            "soft_recall_20": soft_recall20,
            "soft_f1_20": soft_f1_20,

            "cct": cct_score,
            "ttc": ttc_score,
            "zoloss": zoloss,
            "value": zoloss,  # having zoloss twice because of the API
        }

    def get_objective(self):
        return dict(
            X_train=self.X_train, y_test=None, X_test=self.X_test
        )
