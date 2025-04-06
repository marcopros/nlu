import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class ModelIAS(nn.Module):
    def __init__(self, hid_size, out_slot, out_int, emb_size, vocab_len, n_layer = 1, pad_index = 0):
        super(ModelIAS, self).__init__()

        self.embedding = nn.Embdedding(vocab_len, emb_size, padding_idx = pad_index)

        self.utt_encoder = nn.LSTM(emb_size, hid_size, n_layer, bidirectional = False, batch_first = True)

        self.slot_out = nn.Linear(hid_size, out_slot)
        
        self.intent_out = nn.Linear(hid_size, out_int)

        self.dropout = nn.Dropout(0.1)

    def forward(self, utterance, seq_lenghts):
        utter_emb = self.embedding(utterance)
        packed_input = pack_padded_sequence(utter_emb, seq_lenghts.cpu().numpy(), batch_first = True)
        packed_output, (last_hidden, cell) = self.utt_encoder(packed_input)

        # Unpack the sequence
        utt_encoded, input_sizes = pad_packed_sequence(packed_output, batch_first = True)
        last_hidden = last_hidden[-1,:,:]

        # Compute slot logits
        slots = self.slot_out(utt_encoded)
        intent = self.intent_out(last_hidden)

        slots = slots.permute(0,2,1)
        return slots, intent