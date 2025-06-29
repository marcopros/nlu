import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class ModelIAS_Baseline(nn.Module):
    """Baseline IAS model - Unidirectional LSTM, no dropout"""
    def __init__(self, hid_size, out_slot, out_int, emb_size, vocab_len, n_layer = 1, pad_index = 0):
        super(ModelIAS_Baseline, self).__init__()

        self.embedding = nn.Embedding(vocab_len, emb_size, padding_idx = pad_index)
        self.utt_encoder = nn.LSTM(emb_size, hid_size, n_layer, bidirectional = False, batch_first = True)
        self.slot_out = nn.Linear(hid_size, out_slot)
        self.intent_out = nn.Linear(hid_size, out_int)

    def forward(self, utterance, seq_lengths):
        utter_emb = self.embedding(utterance)
        
        # Pack the input for efficient LSTM processing
        packed_input = pack_padded_sequence(utter_emb, seq_lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, (last_hidden, cell) = self.utt_encoder(packed_input)

        # Unpack output
        utt_encoded, _ = pad_packed_sequence(packed_output, batch_first=True)

        # Use last hidden state for intent classification
        last_hidden_cat = last_hidden[-1]

        # Slot logits: shape [batch, seq_len, out_slot]
        slots = self.slot_out(utt_encoded)

        # Intent logits: shape [batch, out_int]
        intent = self.intent_out(last_hidden_cat)

        # Rearrange for loss computation
        slots = slots.permute(0, 2, 1)

        return slots, intent

class ModelIAS_Bidir(nn.Module):
    """IAS model with Bidirectional LSTM, no dropout"""
    def __init__(self, hid_size, out_slot, out_int, emb_size, vocab_len, n_layer = 1, pad_index = 0):
        super(ModelIAS_Bidir, self).__init__()

        self.embedding = nn.Embedding(vocab_len, emb_size, padding_idx = pad_index)
        self.utt_encoder = nn.LSTM(emb_size, hid_size, n_layer, bidirectional = True, batch_first = True)
        self.slot_out = nn.Linear(hid_size * 2, out_slot)
        self.intent_out = nn.Linear(hid_size * 2, out_int)

    def forward(self, utterance, seq_lengths):
        utter_emb = self.embedding(utterance)
        
        # Pack the input for efficient LSTM processing
        packed_input = pack_padded_sequence(utter_emb, seq_lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, (last_hidden, cell) = self.utt_encoder(packed_input)

        # Unpack output
        utt_encoded, _ = pad_packed_sequence(packed_output, batch_first=True)

        # Concatenate final forward and backward hidden states for intent classification
        last_hidden_cat = torch.cat((last_hidden[-2], last_hidden[-1]), dim=1)

        # Slot logits: shape [batch, seq_len, out_slot]
        slots = self.slot_out(utt_encoded)

        # Intent logits: shape [batch, out_int]
        intent = self.intent_out(last_hidden_cat)

        # Rearrange for loss computation
        slots = slots.permute(0, 2, 1)

        return slots, intent

class ModelIAS(nn.Module):
    """Full IAS model - Bidirectional LSTM + Dropout"""
    def __init__(self, hid_size, out_slot, out_int, emb_size, vocab_len, n_layer = 1, pad_index = 0):
        super(ModelIAS, self).__init__()

        self.embedding = nn.Embedding(vocab_len, emb_size, padding_idx = pad_index)
        self.utt_encoder = nn.LSTM(emb_size, hid_size, n_layer, bidirectional = True, batch_first = True)
        self.slot_out = nn.Linear(hid_size * 2, out_slot)
        self.intent_out = nn.Linear(hid_size * 2, out_int)
        self.dropout = nn.Dropout(0.1)

    def forward(self, utterance, seq_lengths):
        utter_emb = self.embedding(utterance)
        utter_emb = self.dropout(utter_emb)
        
        # Pack the input for efficient LSTM processing
        packed_input = pack_padded_sequence(utter_emb, seq_lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, (last_hidden, cell) = self.utt_encoder(packed_input)

        # Unpack output
        utt_encoded, _ = pad_packed_sequence(packed_output, batch_first=True)
        utt_encoded = self.dropout(utt_encoded)

        # Concatenate final forward and backward hidden states for intent classification
        if self.utt_encoder.bidirectional:
            last_hidden_cat = torch.cat((last_hidden[-2], last_hidden[-1]), dim=1)
            last_hidden_cat = self.dropout(last_hidden_cat)
        else:
            last_hidden_cat = last_hidden[-1]

        # Slot logits: shape [batch, seq_len, out_slot]
        slots = self.slot_out(utt_encoded)

        # Intent logits: shape [batch, out_int]
        intent = self.intent_out(last_hidden_cat)

        # Rearrange for loss computation
        slots = slots.permute(0, 2, 1)

        return slots, intent