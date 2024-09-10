# AR model
from benchopt import BaseSolver, safe_import_context
from benchmark_utils import mean_overlaping_pred

with safe_import_context() as import_ctx:
    import torch
    from torch import optim
    import numpy as np
    from tqdm import tqdm


class Solver(BaseSolver):
    name = "AR"

    install_cmd = "conda"
    requirements = ["pip:torch", "tqdm"]

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

        class AR_model(torch.nn.Module):
            """
            Single linear layer for autoregressive model
            Taking in input a window of size window_size and
            outputting a window of size horizon
            input : (batch_size, window_size, n_features)
            output : (batch_size, horizon, n_features)
            """

            def __init__(self,
                         window_size: int,
                         n_features: int,
                         horizon: int):
                super(AR_model, self).__init__()
                self.window_size = window_size
                self.n_features = n_features
                self.horizon = horizon
                self.linear = torch.nn.Linear(
                    window_size * n_features, horizon * n_features
                    )

            def forward(self, x):
                x = x.reshape(x.size(0), -1)
                x = self.linear(x)
                x = x.reshape(x.size(0), -1, self.n_features)
                return x

        self.X_train = X_train  # (n_samples, n_features)
        self.X_test, self.y_test = X_test, y_test  # (n_samples, n_features)
        self.n_features = X_train.shape[1]

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
            )

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
        self.criterion = torch.nn.MSELoss()

        if self.X_train is not None:
            # (n_windows, window_size+horizon, n_features)
            self.Xw_train = np.lib.stride_tricks.sliding_window_view(
                X_train,
                window_shape=self.window_size+self.horizon,
                axis=0
            ).transpose(0, 2, 1)

        if self.X_test is not None:
            # (n_windows, window_size+horizon, n_features)
            self.Xw_test = np.lib.stride_tricks.sliding_window_view(
                X_test,
                window_shape=self.window_size+self.horizon,
                axis=0
            ).transpose(0, 2, 1)

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
                # (batch_size, window_size, n_features)
                x = torch.tensor(
                    self.Xw_train[i:i+self.batch_size, :self.window_size, :],
                    dtype=torch.float32, device=self.device
                )
                # (batch_size, horizon, n_features)
                y = torch.tensor(
                    self.Xw_train[i:i+self.batch_size, :self.horizon, :],
                    dtype=torch.float32, device=self.device
                )

                self.optimizer.zero_grad()
                y_pred = self.model(x)  # (batch_size, horizon, n_features)
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
        # (n_test_windows, horizon, n_features)
        xw_hat = self.model(torch.tensor(
            self.Xw_test[:, :self.window_size, :],
            dtype=torch.float32
        ).to(self.device))

        if xw_hat.is_cuda:
            xw_hat = xw_hat.detach().cpu().numpy()

        # Reconstructing the prediction from the predicted windows
        # Creating the prediction array with -1 for the unknown values
        # Corresponding to the first window_size values
        x_hat = np.zeros_like(self.X_test)-1  # (n_test_samples, n_features)
        x_hat[self.window_size:self.window_size+self.horizon] = xw_hat[0]

        x_hat[self.window_size+self.horizon:] = mean_overlaping_pred(
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
