import torch.nn as nn

# Base RNN
class LM_RNN(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, out_dropout=0.1,
                 emb_dropout=0.1, n_layers=1):
        super(LM_RNN, self).__init__()
        # Token ids to vectors, we will better see this in the next lab 
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        # Pytorch's RNN layer: https://pytorch.org/docs/stable/generated/torch.nn.RNN.html
        self.rnn = nn.RNN(emb_size, hidden_size, n_layers, bidirectional=False, batch_first=True)    
        self.pad_token = pad_index
        # Linear layer to project the hidden layer to our output space 
        self.output = nn.Linear(hidden_size, output_size)
        
    def forward(self, input_sequence):
        emb = self.embedding(input_sequence)
        rnn_out, _  = self.rnn(emb)
        output = self.output(rnn_out).permute(0,2,1)
        return output 

# First Modification - LSTM 
# class LM_LSTM(nn.Module):
#     def __init__(self, emb_size, hidden_size, output_size, pad_index = 0, out_dropout = 0.1,
#                  emb_dropout = 0.1, n_layers = 1):
#         super(LM_LSTM, self).__init__()
#         self.embedding = nn.Embedding(output_size, emb_size, pad_index)
#         # LSTM Layer instead of RNN
#         self.rnn = nn.LSTM(emb_size, hidden_size, n_layers, bidirectional = False, batch_first = True)
#         self.pad_token = pad_index
#         self.output = nn.Linear(hidden_size, output_size)

#     def forward(self, input_sequence):
#         emb = self.embedding(input_sequence)
#         # Unpack the output from LSTM
#         rnn_out, (h_n, c_n) = self.rnn(emb)
#         output = self.output(rnn_out).permute(0, 2, 1)
#         return output
    
# Second modification - Added Dropout Layers
class LM_LSTM(nn.Module):
    """
    A PyTorch class for a simple LSTM-based language model.
    """
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, out_dropout=0.3, emb_dropout=0.3, n_layers=1):
        super(LM_LSTM, self).__init__()
        
        # Embedding layer converts input tokens into vectors.
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        
        # Dropout applied to the embeddings to prevent overfitting.
        self.dropout1 = nn.Dropout(p=emb_dropout)
        
        # LSTM layer processes the embedded sequences.
        self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, bidirectional=False, batch_first=True)
        
        # The padding token index is stored here.
        self.pad_token = pad_index
        
        # Dropout applied to the output of the LSTM to prevent overfitting.
        self.dropout2 = nn.Dropout(p=out_dropout)
        
        # Final linear layer to project LSTM output to desired output size.
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, input_sequence):
        """
        Defines the forward pass through the model.
        """
        # Convert input tokens into embeddings.
        emb = self.embedding(input_sequence)
        
        # Apply dropout to the embeddings.
        drop1 = self.dropout1(emb)
        
        # Pass the embeddings through the LSTM layer.
        lstm_out, _ = self.lstm(drop1)
        
        # Apply dropout to the LSTM output.
        drop2 = self.dropout2(lstm_out)
        
        # Project the LSTM output to the final output size and permute for sequence tasks.
        output = self.output(drop2).permute(0, 2, 1)
        
        return output
