from transformers import BertTokenizerFast, BertConfig, AdamW
from model import JointBERT
from torch.utils.data import DataLoader, Dataset
from seqeval.metrics import f1_score as seqeval_f1
from sklearn.metrics import accuracy_score
import torch
import numpy as np
import os
from tqdm import tqdm
from functions import *
from utils import *
from sklearn.model_selection import train_test_split
from collections import Counter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert_model_name = "bert-large-uncased"

path = '/home/disi/nlu/NLU/part_B'
PAD_TOKEN = 0


if __name__ == "__main__":
    tmp_train_raw = load_data(os.path.join(path, 'dataset', 'train.json'))
    test_raw = load_data(os.path.join(path, 'dataset', 'test.json'))

    # Print dataset sizes
    print('Train samples:' , len(tmp_train_raw))
    print('Test samples: ', len(test_raw))

    portion = 0.10
    intents = [x['intent'] for x in tmp_train_raw]
    count_y = Counter(intents)

    labels = []
    inputs = []
    mini_train = []

    for id_y, y in enumerate(intents):
        if count_y[y] > 1:
            inputs.append(tmp_train_raw[id_y])
            labels.append(y)
        else:
            mini_train.append(tmp_train_raw[id_y])

    # Random stratify
    X_train, X_dev, y_train, y_dev = train_test_split(inputs, labels, test_size=portion, 
                                                      random_state=42,
                                                      shuffle=True,
                                                      stratify=labels)
    
    X_train.extend(mini_train)
    train_raw = X_train
    dev_raw = X_dev

    y_test = [x['intent'] for x in test_raw]

    words = sum([x['utterance'].split() for x in train_raw], [])
    corpus = train_raw + dev_raw + test_raw

    slots = set(sum([line['slots'].split() for line in corpus], []))
    intents = set([line['intent'] for line in corpus])

    tokenizer = BertTokenizerFast.from_pretrained(bert_model_name)
    slot2id = {slot: i for i, slot in enumerate(sorted(slots))}
    intent2id = {intent: i for i, intent in enumerate(sorted(intents))}
    id2slot = {i: slot for slot, i in slot2id.items()}
    id2intent = {i: intent for intent, i in intent2id.items()}

    train_dataset = ATISDataset(train_raw, tokenizer, slot2id, intent2id)
    dev_dataset = ATISDataset(dev_raw, tokenizer, slot2id, intent2id)
    test_dataset = ATISDataset(test_raw, tokenizer, slot2id, intent2id)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)

    n_epochs = 10
    runs = 3

    slot_f1s, intent_accs = [], []

    for run in tqdm(range(runs)):
        config = BertConfig.from_pretrained(bert_model_name)
        model = JointBERT.from_pretrained(bert_model_name, config=config, num_intents=len(intent2id), num_slots=len(slot2id)).to(device)
        optimizer = AdamW(model.parameters(), lr=2e-5)
        best_f1 = 0
        patience = 3

        for epoch in range(n_epochs):
            model.train()
            for batch in train_loader:
                optimizer.zero_grad()
                batch = {k: v.to(device) for k, v in batch.items()}
                loss, _, _ = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    token_type_ids=batch["token_type_ids"],
                    slot_labels=batch["slot_labels"],
                    intent_label=batch["intent_label"]
                )
                loss.backward()
                optimizer.step()

            # Validation
            model.eval()
            all_intents, all_intent_preds = [], []
            all_slots, all_slot_preds = [], []
            with torch.no_grad():
                for batch in dev_loader:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    _, intent_logits, slot_logits = model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        token_type_ids=batch["token_type_ids"]
                    )
                    intent_preds = intent_logits.argmax(dim=1).cpu().numpy()
                    all_intents.extend(batch["intent_label"].cpu().numpy())
                    all_intent_preds.extend(intent_preds)

                    for i, slot_label in enumerate(batch["slot_labels"]):
                        true = []
                        pred = []
                        for j, label_id in enumerate(slot_label.cpu().numpy()):
                            if label_id != -100:
                                true.append(id2slot[label_id])
                                pred.append(id2slot[slot_logits[i][j].argmax().item()])
                        all_slots.append(true)
                        all_slot_preds.append(pred)
            intent_acc = accuracy_score(all_intents, all_intent_preds)
            slot_f1 = seqeval_f1(all_slots, all_slot_preds)
            if slot_f1 > best_f1:
                best_f1 = slot_f1
                patience = 3
                # Save best model in a unique folder for each run
                run_dir = os.path.join(path, f"run{run+1}", f"bin{run+1}")
                os.makedirs(run_dir, exist_ok=True)
                PATH = os.path.join(run_dir, "weights.pt")
                torch.save(model.state_dict(), PATH)
            else:
                patience -= 1
            if patience == 0:
                break

        # Test
        model.eval()
        all_intents, all_intent_preds = [], []
        all_slots, all_slot_preds = [], []
        with torch.no_grad():
            for batch in test_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                _, intent_logits, slot_logits = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    token_type_ids=batch["token_type_ids"]
                )
                intent_preds = intent_logits.argmax(dim=1).cpu().numpy()
                all_intents.extend(batch["intent_label"].cpu().numpy())
                all_intent_preds.extend(intent_preds)
                for i, slot_label in enumerate(batch["slot_labels"]):
                    true = []
                    pred = []
                    for j, label_id in enumerate(slot_label.cpu().numpy()):
                        if label_id != -100:
                            true.append(id2slot[label_id])
                            pred.append(id2slot[slot_logits[i][j].argmax().item()])
                    all_slots.append(true)
                    all_slot_preds.append(pred)
        intent_accs.append(accuracy_score(all_intents, all_intent_preds))
        slot_f1s.append(seqeval_f1(all_slots, all_slot_preds))

    print('Slot F1', round(np.mean(slot_f1s), 3), '+-', round(np.std(slot_f1s), 3))
    print('Intent Acc', round(np.mean(intent_accs), 3), '+-', round(np.std(intent_accs), 3))