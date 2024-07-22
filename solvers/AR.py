# AR model
from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    import torch
    from torch import optim, nn
    import numpy as np
    from tqdm import tqdm


class AR_model(nn.Module):
    """
    Single linear layer for autoregressive model
    Taking in input a window of size window_size and
    outputting a window of size horizon
    input : (batch_size, window_size, n_features)
    output : (batch_size, horizon, n_features)
    """

    def __init__(self, window_size: int, n_features: int, horizon: int):
        super(AR_model, self).__init__()
        self.window_size = window_size
        self.n_features = n_features
        self.horizon = horizon
        self.linear = nn.Linear(window_size * n_features, horizon * n_features)

    def forward(self, x):
        x = x.reshape(x.size(0), -1)
        x = self.linear(x)
        x = x.reshape(x.size(0), -1, self.n_features)
        return x


class Solver(BaseSolver):
    name = "AR"

    install_cmd = "conda"
    requirements = ["pip:torch", "tqdm"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sampling_strategy = "run_once"

    parameters = {
        "batch_size": [128],
        "n_epochs": [50],
        "lr": [1e-5],
        "weight_decay": [1e-7],
        "window_size": [256],
        "horizon": [1],
        "percentile": [99.4],
    }

    def set_objective(self, X_train, y_test, X_test):
        self.X_train = X_train
        self.X_test, self.y_test = X_test, y_test
        self.n_features = X_train.shape[1]

        self.model = AR_model(
            self.window_size,
            self.n_features,
            self.horizon
        )
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            # weight_decay=self.weight_decay
        )
        self.criterion = nn.MSELoss()

        if self.X_train is not None:
            self.Xw_train = np.lib.stride_tricks.sliding_window_view(
                X_train,
                window_shape=self.window_size+self.horizon,
                axis=0
            ).transpose(0, 2, 1)

        if self.X_test is not None:
            self.Xw_test = np.lib.stride_tricks.sliding_window_view(
                X_test,
                window_shape=self.window_size+self.horizon,
                axis=0
            ).transpose(0, 2, 1)

    def mean_overlaping_pred(
        self, predictions, stride
    ):
        """
        Averages overlapping predictions for multivariate time series.

        Args:
        predictions: np.ndarray, shape (n_windows, H, n_features)
                    The predicted values for each window for each feature.
        stride: int
                The stride size.

        Returns:
        np.ndarray: Averaged predictions for each feature.
        """
        n_windows, H, n_features = predictions.shape
        total_length = (n_windows-1) * stride + H - 1

        # Array to store accumulated predictions for each feature
        accumulated = np.zeros((total_length, n_features))
        # store the count of predictions at each point for each feature
        counts = np.zeros((total_length, n_features))

        # Accumulate predictions and counts based on stride
        for i in range(n_windows):
            start = i * stride
            accumulated[start:start+H] += predictions[i]
            counts[start:start+H] += 1

        # Avoid division by zero
        counts[counts == 0] = 1

        # Average the accumulated predictions
        averaged_predictions = accumulated / counts

        return averaged_predictions


    def run(self, _):

        self.model.to(self.device)
        self.criterion.to(self.device)

        best_loss = float('inf')  # Initialize best_loss with infinity
        best_model = None         # Variable to store the best model

        ti = tqdm(range(self.n_epochs), desc="epoch", leave=True)

        for epoch in ti:
            self.model.train()
            epoch_loss = 0.0

            for i in range(0, len(self.Xw_train), self.batch_size):
                x = torch.tensor(
                    self.Xw_train[i:i+self.batch_size, :self.window_size, :],
                    dtype=torch.float32
                )
                y = torch.tensor(
                    self.Xw_train[i:i+self.batch_size, :self.horizon, :],
                    dtype=torch.float32
                )

                x = x.to(self.device)
                y = y.to(self.device)

                self.optimizer.zero_grad()
                y_pred = self.model(x)
                loss = self.criterion(y_pred, y)

                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

            epoch_loss /= (len(self.Xw_train) / self.batch_size)
            ti.set_description(f"Epoch {epoch}, Epoch Loss {epoch_loss: .5e}")

            # Checkpoint the model if the loss is lower
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model = self.model.state_dict()

        self.model.load_state_dict(best_model)

        self.model.eval()
        xw_hat = self.model(torch.tensor(
            self.Xw_test[:, :self.window_size, :],
            dtype=torch.float32
        ).to(self.device))

        if self.device == torch.device("cuda"):
            xw_hat = xw_hat.detach().cpu().numpy()

        # Reconstructing the prediction from the predicted windows
        # Creating the prediction array with -1 for the unknown values
        # Corresponding to the first window_size values
        x_hat = np.zeros_like(self.X_test)-1
        x_hat[self.window_size:self.window_size+self.horizon] = xw_hat[0]
        # x_hat[self.window_size+self.horizon:] = xw_hat[1:, -self.horizon]
        x_hat[self.window_size+self.horizon:] = self.mean_overlaping_pred(
            xw_hat, 1
        )

        percentile_value = np.percentile(
            np.abs(self.X_test[self.window_size:] - x_hat[self.window_size:]),
            self.percentile
        )

        predictions = np.zeros_like(x_hat)-1
        predictions[self.window_size:] = np.where(
            np.abs(self.X_test[self.window_size:] -
                   x_hat[self.window_size:]) > percentile_value, 1, 0
        )

        self.predictions = np.max(predictions, axis=1)

    def skip(self, X_train, X_test, y_test):
        if X_train.shape[0] < self.window_size + self.horizon:
            return True, "No enough training samples"
        if X_test.shape[0] < self.window_size + self.horizon:
            return True, "No enough testing samples"
        return False, None

    def get_result(self):
        return dict(y_hat=self.predictions)
