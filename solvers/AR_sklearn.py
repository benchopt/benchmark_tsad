from benchopt import BaseSolver
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from benchmark_utils import mean_overlaping_pred


class Solver(BaseSolver):
    name = "AR_sklearn"

    install_cmd = "conda"
    requirements = ["scikit-learn", "numpy"]

    parameters = {
        "window_size": [128, 256, 512],
        "horizon": [1],
        "percentile": [99.4],
    }

    def set_objective(self, X_train, y_test, X_test):
        self.X_train = X_train
        self.X_test, self.y_test = X_test, y_test
        self.n_features = X_train.shape[1]

        self.model = LinearRegression()
        self.scaler = StandardScaler()

        if self.X_train is not None:
            self.Xw_train = np.lib.stride_tricks.sliding_window_view(
                X_train,
                window_shape=self.window_size+self.horizon,
                axis=0
            ).reshape(-1, (self.window_size+self.horizon)*self.n_features)

        if self.X_test is not None:
            self.Xw_test = np.lib.stride_tricks.sliding_window_view(
                X_test,
                window_shape=self.window_size+self.horizon,
                axis=0
            ).reshape(-1, (self.window_size+self.horizon)*self.n_features)

    def run(self, _):
        X = self.Xw_train[:, :self.window_size*self.n_features]
        y = self.Xw_train[:, self.window_size*self.n_features:]

        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)

        X_test = self.Xw_test[:, :self.window_size*self.n_features]
        X_test_scaled = self.scaler.transform(X_test)
        xw_hat = self.model.predict(X_test_scaled)

        x_hat = np.zeros_like(self.X_test) - 1
        x_hat[self.window_size:self.window_size+self.horizon] = xw_hat[0].reshape(self.horizon, self.n_features)
        x_hat[self.window_size+self.horizon:] = mean_overlaping_pred(xw_hat.reshape(-1, self.horizon, self.n_features), stride=1)

        percentile_value = np.percentile(
            np.abs(self.X_test[self.window_size:] - x_hat[self.window_size:]),
            self.percentile
        )

        predictions = np.zeros_like(x_hat) - 1
        predictions[self.window_size:] = np.where(
            np.abs(
                self.X_test[self.window_size:] - x_hat[self.window_size:]
                ) > percentile_value, 1, 0
        )

        self.predictions = np.max(predictions, axis=1)

    def skip(self, X_train, X_test, y_test):
        if X_train.shape[0] < self.window_size + self.horizon:
            return True, "Not enough training samples"
        if X_test.shape[0] < self.window_size + self.horizon:
            return True, "Not enough testing samples"
        return False, None

    def get_result(self):
        return dict(y_hat=self.predictions)
