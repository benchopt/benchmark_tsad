from torch import nn


class ARModel(nn.Module):
    """
    Class for the AR Solver
    Single linear layer for autoregressive model
    Taking in input a window of size window_size and
    outputting a window of size horizon
    input : (batch_size, window_size, n_features)
    output : (batch_size, horizon, n_features)
    """

    def __init__(self,
                 n_features: int,
                 window_size: int,
                 horizon=1,
                 ):
        super(ARModel, self).__init__()
        self.window_size = window_size
        self.n_features = n_features
        self.horizon = horizon
        self.linear = nn.Linear(window_size * n_features, horizon * n_features)

    def forward(self, x):
        x = x.reshape(x.size(0), -1)
        x = self.linear(x)
        x = x.reshape(x.size(0), -1, self.n_features)
        return x


class TransformerModel(nn.Module):
    """
    Class for the Vanilla-Transformer Solver
    Transformer model for time series forecasting
    input : (batch_size, sequence_length, n_features)
    output : (batch_size, horizon, n_features)
    """

    def __init__(self,
                 n_features: int,
                 sequence_length: int,
                 horizon=1,
                 num_layers=1,
                 num_heads=2,
                 dim_feedforward=512,
                 ):
        super(TransformerModel, self).__init__()
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.horizon = horizon

        # Ensure d_model is divisible by num_heads
        self.d_model = ((n_features - 1) // num_heads + 1) * num_heads

        self.input_projection = nn.Linear(n_features, self.d_model)

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer,
            num_layers=num_layers
        )
        self.fc_out = nn.Linear(
            self.d_model * sequence_length, horizon * n_features)

    def forward(self, src):
        # src shape: (batch_size, sequence_length, input_size)
        src = self.input_projection(src)  # Project to d_model
        src = src.transpose(0, 1)  # (sequence_length, batch_size, d_model)

        output = self.transformer_encoder(src)
        # (batch_size, sequence_length, d_model)
        output = output.transpose(0, 1)
        output = output.flatten(1)  # (batch_size, sequence_length * d_model)
        output = self.fc_out(output)
        # (batch_size, horizon, input_size)
        output = output.view(-1, self.horizon, self.input_size)

        return output


class AutoEncoderLSTM(nn.Module):
    """
    Class for the LSTM Solver
    LSTM Autoencoder model for time series forecasting
    input : (batch_size, sequence_length, n_features)
    output : (batch_size, sequence_length, n_features)
    """

    def __init__(self,
                 n_features: int,
                 sequence_length: int,
                 embedding_dim=64,
                 enc_layers=1,
                 dec_layers=1,
                 ):
        super(AutoEncoderLSTM, self).__init__()
        self.sequence_length, self.n_features = sequence_length, n_features
        self.embedding_dim, self.hidden_dim = embedding_dim, 2 * embedding_dim

        self.encoder = nn.LSTM(
            input_size=n_features,
            hidden_size=self.hidden_dim,
            num_layers=enc_layers,
            batch_first=True
        )

        self.decoder = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=n_features,
            num_layers=dec_layers,
            batch_first=True
        )

    def forward(self, x):

        x, (_, _) = self.encoder(x)
        x, (_, _) = self.decoder(x)

        return x
