import torch.nn as nn

class LM_LSTM(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size, pad_index = 0, out_dropout = 0.1,
                 emb_dropout = 0.1, n_layers = 1):
        super(LM_LSTM, self).__init__()
        self.embedding = nn.Embedding(output_size, emb_size, pad_index)
        # LSTM Layer instead of RNN
        self.rnn = nn.LSTM(emb_size, hidden_size, n_layers, bidirectional = False, batch_first = True)
        self.pad_token = pad_index
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, input_sequence):
        emb = self.embedding(input_sequence)
        # Unpack the output from LSTM
        rnn_out, (h_n, c_n) = self.rnn(emb)
        output = self.output(rnn_out).permute(0, 2, 1)
        return output
