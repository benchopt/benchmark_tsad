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

    def set_data(self, X, y):
        "Set the data to compute the objective."
        self.X, self.y = X, y

    def evaluate_result(self, y_hat):
        "Evaluate the result provided by the solver."
        precision = np.sum(y_hat[self.y == 1] == 1) / np.sum(y_hat == 1)
        recall = np.sum(y_hat[self.y == 1] == 1) / (
            np.sum(y_hat[self.y == 1] == 1) + np.sum(y_hat[self.y == 1] == 0)
        )
        f1 = 2 * precision * recall / (precision + recall)
        zoloss = np.sum(y_hat != self.y) / len(self.y)
        return dict(value=zoloss, precision=precision, recall=recall, f1=f1)

    def get_objective(self):
        return dict(X=self.X, y=self.y)
