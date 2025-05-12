import torch
import torch.nn as nn
import os
from transformers import BertConfig
from model import JointBERT
from sklearn.metrics import accuracy_score
from seqeval.metrics import f1_score as seqeval_f1
from tqdm import tqdm
import numpy as np

def training_loop(
    runs,
    n_epochs,
    train_loader,
    dev_loader,
    test_loader,
    bert_model_name,
    intent2id,
    slot2id,
    id2slot,
    path,
    device
):
    slot_f1s, intent_accs = [], []
    for run in tqdm(range(runs)):
        config = BertConfig.from_pretrained(bert_model_name)
        model = JointBERT.from_pretrained(
            bert_model_name,
            config=config,
            num_intents=len(intent2id),
            num_slots=len(slot2id)
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
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
                run_dir = os.path.join(path, f"run{run+1}", f"bin{run+1}")
                os.makedirs(run_dir, exist_ok=True)
                PATH = os.path.join(run_dir, "weights.pt")
                torch.save(model.state_dict(), PATH)
            else:
                patience -= 1
            if patience == 0:
                break

        # Testing
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
    return slot_f1s, intent_accs
