import torch.nn as nn

class VarDropout(nn.Module):
    '''
    Class for Variational Dropout
    '''
    def __init__(self, dropout=0.5):
        super(VarDropout, self).__init__()
        self.dropout = dropout

    def forward(self, x):
        # During evaluation, dropout is not applied
        if not self.training:
            return x

        batch_size = x.size(0)
        # Create a dropout mask with shape (batch_size, 1, feature_dim)
        mask = x.data.new(batch_size, 1, x.size(2)).bernoulli_(1 - self.dropout)
        # Scale the mask to preserve expected value
        mask = mask / (1 - self.dropout)
        # Expand mask to match input shape and apply it
        mask = mask.expand_as(x)
        return x * mask


class LM_LSTM(nn.Module):
    '''
    Language Model with LSTM and Variational Dropout
    '''
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, out_dropout=0.5, emb_dropout=0.5, n_layers=1):
        super(LM_LSTM, self).__init__()
        # Embedding layer with padding index
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        # Variational dropout for embeddings
        self.dropout1 = VarDropout(dropout=emb_dropout)
        # LSTM layer with the specified number of layers
        self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, bidirectional=False, batch_first=True)
        self.pad_token = pad_index
        # Variational dropout for LSTM output
        self.dropout2 = VarDropout(dropout=out_dropout)
        # Output linear layer
        self.output = nn.Linear(hidden_size, output_size)

        # Tie the weights between embedding and output layer if dimensions match
        if emb_size == hidden_size:
            self.output.weight = self.embedding.weight
        else:
            raise ValueError("Weight tying requires the hidden size to match the embedding size.")

    def forward(self, input_sequence):
        # Embedding lookup
        emb = self.embedding(input_sequence)
        # Apply dropout to embeddings
        drop1 = self.dropout1(emb)
        # LSTM forward pass
        lstm_out, _ = self.lstm(drop1)
        # Apply dropout to LSTM output
        drop2 = self.dropout2(lstm_out)
        # Pass through the output layer and adjust dimensions
        output = self.output(drop2).permute(0, 2, 1)
        return output
