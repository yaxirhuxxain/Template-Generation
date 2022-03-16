import pytorch_lightning as pl
from torch import nn


# Declaring the model
class recurrentModels(pl.LightningModule):

    def __init__(self,
                 vocab_size,
                 learner='RNN',
                 n_layers=1,
                 n_hidden=256,
                 n_embedding=300,
                 n_drop=0.5,
                 loss_function="cross_entropy",
                 learning_optmizer='Adam',
                 learn_rate=0.001,
                 ):

        super(recurrentModels, self).__init__()
        self.vocab_size = vocab_size
        self.n_hidden = n_hidden
        self.n_embedding = n_embedding
        self.n_layers = n_layers
        self.n_drop = n_drop
        self.loss_function = loss_function
        self.learner = learner
        self.learning_optmizer = learning_optmizer
        self.learn_rate = learn_rate

        self.embed = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=n_embedding)

        # define the recurrent model
        if self.learner == "RNN":
            self.rnn = nn.RNN(input_size=n_embedding,
                              hidden_size=n_hidden,
                              batch_first=True,
                              num_layers=self.n_layers,
                              dropout=self.n_drop)
        elif self.learner == "LSTM":
            self.lstm = nn.LSTM(input_size=n_embedding,
                                hidden_size=n_hidden,
                                batch_first=True,
                                num_layers=self.n_layers,
                                dropout=self.n_drop)
        elif self.learner == "GRU":
            self.gru = nn.GRU(input_size=n_embedding,
                              hidden_size=n_hidden,
                              batch_first=True,
                              num_layers=self.n_layers,
                              dropout=self.n_drop)
        else:
            raise ValueError(
                "Invalid learner choose from these (RNN,LSTM,GRU)")

        # define a dropout layer
        self.dropout = nn.Dropout(self.n_drop)

        # define the final, fully-connected output layer
        self.linear = nn.Linear(in_features=n_hidden, out_features=vocab_size)

        self.init_weights()
        self.save_hyperparameters()

    def init_weights(self):
        for name, param in self.named_parameters():
            nn.init.uniform_(param.data, -0.08, 0.08)

    def forward(self, x):

        x = self.dropout(self.embed(x))
        if self.learner == "RNN":
            _out, ht = self.rnn(x)
        elif self.learner == "LSTM":
            _out, (ht, ct) = self.lstm(x)
        elif self.learner == "GRU":
            _out, ht = self.gru(x)
        else:
            raise ValueError(
                "Invalid learner choose from these (RNN,LSTM,GRU)")

        return self.linear(ht[-1])
