from benchopt import BaseObjective, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np


class Objective(BaseObjective):
    name = "Anomaly detection"

    def get_one_result(self):
        """Return one solution for which the objective can be computed,
        Used to get the shape of the result.
        Our algorithms will return an array of labels of shape (n_samples,)
        """
        return np.zeros(self.X.shape[0])

    def set_data(self, X, y, X_test):
        "Set the data to compute the objective."
        self.X, self.y = X, y
        self.X_test = X_test

    def evaluate_result(self, y_hat):
        "Evaluate the result provided by the solver."
        precision = self.get_precision(y_hat)
        recall = self.get_recall(y_hat)
        f1 = self.get_f1(precision, recall)
        zoloss = self.get_zoloss(y_hat)

        return dict(
            zoloss=zoloss,
            precision=precision,
            recall=recall,
            f1=f1,
            value=zoloss,  # having zoloss twice because of the API
        )

    def get_precision(self, y_hat):
        return np.sum(y_hat[self.y == 1] == 1) / np.sum(y_hat == 1)

    def get_recall(self, y_hat):
        return np.sum(y_hat[self.y == 1] == 1) / (
            np.sum(y_hat[self.y == 1] == 1) + np.sum(y_hat[self.y == 1] == 0)
        )

    def get_f1(self, precision, recall):
        return 2 * precision * recall / (precision + recall)

    def get_zoloss(self, y_hat):
        return np.sum(y_hat != self.y) / len(self.y)

    def get_objective(self):
        return dict(X=self.X, y=self.y)
