import os
import torch
import torch.utils.data as data
import json
from pprint import pprint
from collections import Counter
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer

device = 'cuda:0'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
PAD_TOKEN = 0

class ATISDataset(Dataset):
    def __init__(self, data, tokenizer, slot2id, intent2id, max_len=64):
        self.data = data
        self.tokenizer = tokenizer
        self.slot2id = slot2id
        self.intent2id = intent2id
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        tokens = item['utterance'].split()
        slots = item['slots'].split()
        intent = item['intent']
        encoding = self.tokenizer(tokens, is_split_into_words=True, truncation=True, max_length=self.max_len, return_tensors="pt", padding='max_length')
        word_ids = encoding.word_ids(batch_index=0)
        slot_labels = []
        prev_word = None
        for word_id in word_ids:
            if word_id is None:
                slot_labels.append(-100)
            elif word_id != prev_word:
                slot_labels.append(self.slot2id[slots[word_id]])
            else:
                slot_labels.append(-100)  # ignore sub-tokens for slot loss
            prev_word = word_id
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "token_type_ids": encoding["token_type_ids"].squeeze(0),
            "slot_labels": torch.tensor(slot_labels),
            "intent_label": torch.tensor(self.intent2id[intent])
        }


def load_data(path):
    '''
        input: path/to/data
        output: json
    '''
    dataset = []
    with open(path) as f:
        dataset = json.loads(f.read())
    return dataset


# NOT USED
# ----------------------------
# class Lang():
#     def __init__(self, words, intents, slots, cutoff=0):
#         self.word2id = self.w2id(words, cutoff=cutoff, unk=True)
#         self.slot2id = self.lab2id(slots)
#         self.intent2id = self.lab2id(intents, pad=False)
#         self.id2word = {v:k for k, v in self.word2id.items()}
#         self.id2slot = {v:k for k, v in self.slot2id.items()}
#         self.id2intent = {v:k for k, v in self.intent2id.items()}
        
#     def w2id(self, elements, cutoff=None, unk=True):
#         vocab = {'pad': PAD_TOKEN}
#         if unk:
#             vocab['unk'] = len(vocab)
#         count = Counter(elements)
#         for k, v in count.items():
#             if v > cutoff:
#                 vocab[k] = len(vocab)
#         return vocab
    
#     def lab2id(self, elements, pad=True):
#         vocab = {}
#         if pad:
#             vocab['pad'] = PAD_TOKEN
#         for elem in elements:
#                 vocab[elem] = len(vocab)
#         return vocab
    
# class IntentsAndSlots (data.Dataset):
#     # Mandatory methods are __init__, __len__ and __getitem__
#     def __init__(self, dataset, lang, unk='unk'):
#         self.utterances = []
#         self.intents = []
#         self.slots = []
#         self.unk = unk
        
#         for x in dataset:
#             self.utterances.append(x['utterance'])
#             self.slots.append(x['slots'])
#             self.intents.append(x['intent'])

#         self.utt_ids = self.mapping_seq(self.utterances, lang.word2id)
#         self.slot_ids = self.mapping_seq(self.slots, lang.slot2id)
#         self.intent_ids = self.mapping_lab(self.intents, lang.intent2id)

#     def __len__(self):
#         return len(self.utterances)

#     def __getitem__(self, idx):
#         utt = torch.Tensor(self.utt_ids[idx])
#         slots = torch.Tensor(self.slot_ids[idx])
#         intent = self.intent_ids[idx]
#         sample = {'utterance': utt, 'slots': slots, 'intent': intent}
#         return sample
    
#     # Auxiliary methods
    
#     def mapping_lab(self, data, mapper):
#         return [mapper[x] if x in mapper else mapper[self.unk] for x in data]
    
#     def mapping_seq(self, data, mapper): # Map sequences to number
#         res = []
#         for seq in data:
#             tmp_seq = []
#             for x in seq.split():
#                 if x in mapper:
#                     tmp_seq.append(mapper[x])
#                 else:
#                     tmp_seq.append(mapper[self.unk])
#             res.append(tmp_seq)
#         return res
    


# def collate_fn(data):
#     def merge(sequences):
#         '''
#         merge from batch * sent_len to batch * max_len 
#         '''
#         lengths = [len(seq) for seq in sequences]
#         max_len = 1 if max(lengths)==0 else max(lengths)
#         # Pad token is zero in our case
#         # So we create a matrix full of PAD_TOKEN (i.e. 0) with the shape 
#         # batch_size X maximum length of a sequence
#         padded_seqs = torch.LongTensor(len(sequences),max_len).fill_(PAD_TOKEN)
#         for i, seq in enumerate(sequences):
#             end = lengths[i]
#             padded_seqs[i, :end] = seq # We copy each sequence into the matrix
#         # print(padded_seqs)
#         padded_seqs = padded_seqs.detach()  # We remove these tensors from the computational graph
#         return padded_seqs, lengths
#     # Sort data by seq lengths
#     data.sort(key=lambda x: len(x['utterance']), reverse=True) 
#     new_item = {}
#     for key in data[0].keys():
#         new_item[key] = [d[key] for d in data]
        
#     # We just need one length for packed pad seq, since len(utt) == len(slots)
#     src_utt, _ = merge(new_item['utterance'])
#     y_slots, y_lengths = merge(new_item["slots"])
#     intent = torch.LongTensor(new_item["intent"])
    
#     src_utt = src_utt.to(device) # We load the Tensor on our selected device
#     y_slots = y_slots.to(device)
#     intent = intent.to(device)
#     y_lengths = torch.LongTensor(y_lengths).to(device)
    
#     new_item["utterances"] = src_utt
#     new_item["intents"] = intent
#     new_item["y_slots"] = y_slots
#     new_item["slots_len"] = y_lengths
#     return new_item

# def encode_utterance_and_labels(tokenizer, utterance, slot_labels, slot2id, max_length = 64):
#     tokens = utterance.split()
#     bert_inputs = tokenizer(tokens, is_split_into_words = True, truncation = True, max_length = max_length, return_tensors = 'pt')
#     word_ids = bert_inputs.word_ids()
#     slot_label_ids = []
#     previous_word_idx = None
#     for word_idx in word_ids:
#         if word_idx is None:
#             slot_label_ids.append(-100)  # ignore special tokens in loss
#         elif word_idx != previous_word_idx:
#             slot_label_ids.append(slot2id[slot_labels[word_idx]])
#         else:
#             # For sub-tokens, you can use the same label or a special "X" label
#             slot_label_ids.append(slot2id[slot_labels[word_idx]])
#         previous_word_idx = word_idx
#     bert_inputs['labels'] = torch.tensor([slot_label_ids])
#     return bert_inputs