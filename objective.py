from benchopt import BaseObjective, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from sklearn.metrics import precision_score, recall_score, f1_score


class Objective(BaseObjective):
    name = "Anomaly detection"

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
        precision = self.get_precision(y_hat)
        recall = self.get_recall(y_hat)
        f1 = self.get_f1(y_hat)
        zoloss = self.get_zoloss(y_hat)

        return dict(
            value=zoloss,  # having zoloss twice because of the API
            zoloss=zoloss,
            precision=precision,
            recall=recall,
            f1=f1,
        )

    def get_precision(self, y_hat):
        return precision_score(self.y_test, y_hat, zero_division=0)

    def get_recall(self, y_hat):
        return recall_score(self.y_test, y_hat, zero_division=0)

    def get_f1(self, y_hat):
        return f1_score(self.y_test, y_hat, zero_division=0)

    def get_zoloss(self, y_hat):
        return np.sum(y_hat != self.y_test) / len(self.y_test)

    def get_objective(self):
        return dict(
            X_train=self.X_train, y_test=self.y_test, X_test=self.X_test
        )
