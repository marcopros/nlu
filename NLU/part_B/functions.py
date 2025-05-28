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
    # Lists to store metrics for each run
    slot_f1s, intent_accs = [], []
    best_overall_f1 = 0
    best_model_state = None
    
    for run in tqdm(range(runs)):
        # Initialize BERT config and model for each run
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
        best_model_state_run = None

        # Training loop for each epoch
        for epoch in range(n_epochs):
            model.train()
            for batch in train_loader:
                optimizer.zero_grad()
                # Move all batch tensors to the correct device (CPU/GPU)
                batch = {k: v.to(device) for k, v in batch.items()}
                # Forward pass: compute total loss (intent + slot)
                loss, _, _ = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    token_type_ids=batch["token_type_ids"],
                    slot_labels=batch["slot_labels"],
                    intent_label=batch["intent_label"]
                )
                loss.backward()
                optimizer.step()

            # Validation after each epoch
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
                    # Intent predictions
                    intent_preds = intent_logits.argmax(dim=1).cpu().numpy()
                    all_intents.extend(batch["intent_label"].cpu().numpy())
                    all_intent_preds.extend(intent_preds)
                    # Slot predictions (only for valid sub-tokens)
                    for i, slot_label in enumerate(batch["slot_labels"]):
                        true = []
                        pred = []
                        for j, label_id in enumerate(slot_label.cpu().numpy()):
                            # Only consider positions where label_id != -100 (ignore index for sub-tokens and special tokens)
                            # This ensures that only the first sub-token of each word is evaluated for slot filling
                            if label_id != -100:
                                true.append(id2slot[label_id])
                                pred.append(id2slot[slot_logits[i][j].argmax().item()])
                        all_slots.append(true)
                        all_slot_preds.append(pred)
            intent_acc = accuracy_score(all_intents, all_intent_preds)
            slot_f1 = seqeval_f1(all_slots, all_slot_preds)

            # Early stopping based on best slot F1 (without saving during training)
            if slot_f1 > best_f1:
                best_f1 = slot_f1
                patience = 3
                # Save best model state for this run (in memory only)
                best_model_state_run = model.state_dict().copy()
            else:
                patience -= 1
            if patience == 0:
                break

        # Load the best model from this run for testing
        if best_model_state_run is not None:
            model.load_state_dict(best_model_state_run)

        # Testing phase: evaluate the best model on the test set
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
        # Store metrics for this run
        intent_accs.append(accuracy_score(all_intents, all_intent_preds))
        slot_f1s.append(seqeval_f1(all_slots, all_slot_preds))
        
        # Check if this is the best run overall
        current_f1 = seqeval_f1(all_slots, all_slot_preds)
        if current_f1 > best_overall_f1:
            best_overall_f1 = current_f1
            best_model_state = best_model_state_run.copy()
    
    # Save only the best model from all runs
    if best_model_state is not None:
        model_dir = os.path.join(path, "best_model")
        os.makedirs(model_dir, exist_ok=True)
        PATH = os.path.join(model_dir, "weights.pt")
        torch.save(best_model_state, PATH)
        print(f"✅ Best model saved to: {PATH} (F1: {best_overall_f1:.4f})")
    
    return slot_f1s, intent_accs

def evaluate_model(model, test_loader, id2slot, device):
    """
    Evaluate a trained model on the test set.
    
    Args:
        model: Trained JointBERT model
        test_loader: DataLoader for test data
        id2slot: Dictionary mapping slot IDs to slot names
        device: Device to run evaluation on
    
    Returns:
        intent_acc: Intent classification accuracy
        slot_f1: Slot filling F1 score
    """
    model.eval()
    all_intents, all_intent_preds = [], []
    all_slots, all_slot_preds = [], []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="📊 Evaluating"):
            batch = {k: v.to(device) for k, v in batch.items()}
            _, intent_logits, slot_logits = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                token_type_ids=batch["token_type_ids"]
            )
            
            # Intent predictions
            intent_preds = intent_logits.argmax(dim=1).cpu().numpy()
            all_intents.extend(batch["intent_label"].cpu().numpy())
            all_intent_preds.extend(intent_preds)
            
            # Slot predictions (only for valid sub-tokens)
            for i, slot_label in enumerate(batch["slot_labels"]):
                true = []
                pred = []
                for j, label_id in enumerate(slot_label.cpu().numpy()):
                    # Only consider positions where label_id != -100 (ignore index for sub-tokens and special tokens)
                    if label_id != -100:
                        true.append(id2slot[label_id])
                        pred.append(id2slot[slot_logits[i][j].argmax().item()])
                all_slots.append(true)
                all_slot_preds.append(pred)
    
    # Calculate metrics
    intent_acc = accuracy_score(all_intents, all_intent_preds)
    slot_f1 = seqeval_f1(all_slots, all_slot_preds)
    
    return intent_acc, slot_f1
