import os
import torch
import torch.utils.data as data
import json
from collections import Counter
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer
from collections import Counter
from sklearn.model_selection import train_test_split

device = 'cuda:0'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
PAD_TOKEN = 0

class ATISDataset(Dataset):
    def __init__(self, data, tokenizer, slot2id, intent2id, max_len=64):
        # Initialize dataset with data, tokenizer, and label mappings
        self.data = data
        self.tokenizer = tokenizer
        self.slot2id = slot2id
        self.intent2id = intent2id
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Prepare input and labels for a single sample
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
                slot_labels.append(-100)  # Ignore special tokens
            elif word_id != prev_word:
                slot_labels.append(self.slot2id[slots[word_id]])  # Label for first sub-token
            else:
                slot_labels.append(-100)  # Ignore sub-tokens for slot loss
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
    Loads data from a JSON file.
    Args:
        path: path to data file
    Returns:
        dataset: loaded data as a list of dicts
    '''
    dataset = []
    with open(path) as f:
        dataset = json.loads(f.read())
    return dataset

def load_and_split_data(train_path, portion=0.10, random_state=42):
    """
    Loads training data and splits into train/dev sets with stratification.
    Rare intent samples (with only one occurrence) are always kept in train.
    Returns:
        train_raw, dev_raw
    """
    tmp_train_raw = load_data(train_path)
    intents = [x['intent'] for x in tmp_train_raw]
    count_y = Counter(intents)

    labels = []
    inputs = []
    mini_train = []

    # Separate rare intent samples (count == 1)
    for id_y, y in enumerate(intents):
        if count_y[y] > 1:
            inputs.append(tmp_train_raw[id_y])
            labels.append(y)
        else:
            mini_train.append(tmp_train_raw[id_y])

    # Stratified split for frequent intents
    X_train, X_dev, y_train, y_dev = train_test_split(
        inputs, labels, test_size=portion, random_state=random_state,
        shuffle=True, stratify=labels
    )
    # Add rare intent samples to training set
    X_train.extend(mini_train)
    train_raw = X_train
    dev_raw = X_dev
    return train_raw, dev_raw
