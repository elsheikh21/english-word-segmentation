import torch
import torch.nn as nn

from torch.nn.utils.rnn import pack_padded_sequence


class BaselineModel(nn.Module):
    def __init__(self, hparams):
        super(BaselineModel, self).__init__()
        self.word_embedding = nn.Embedding(hparams.vocab_size, hparams.embedding_dim)
        if hparams.embeddings is not None:
            print("initializing embeddings from pretrained")
            self.word_embedding.weight.data.copy_(hparams.embeddings)

        self.lstm = nn.LSTM(hparams.embedding_dim, hparams.hidden_dim,
                            bidirectional=hparams.bidirectional,
                            num_layers=hparams.num_layers,
                            dropout=hparams.dropout if hparams.num_layers > 1 else 0)
        lstm_output_dim = hparams.hidden_dim if hparams.bidirectional is False else hparams.hidden_dim * 2

        self.dropout = nn.Dropout(hparams.dropout)
        self.classifier = nn.Linear(lstm_output_dim, hparams.num_classes)

    def forward(self, x, x_lengths):
        embeddings = self.word_embedding(x)
        embeddings = self.dropout(embeddings)
        packed_input = pack_padded_sequence(embeddings, x_lengths.cpu().numpy(),
                                            batch_first=True)
        o, _ = self.lstm(embeddings)
        o = self.dropout(o)
        output = self.classifier(o)
        return output