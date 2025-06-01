# Import necessary libraries
from transformers import BertTokenizerFast, BertConfig
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

# =================== EVALUATION MODE CONFIGURATION ===================
# Set to True to enable evaluation mode (load pre-trained weights and evaluate on test set)
EVALUATION_MODE = True  
# Path to the saved model weights (.pt file) for evaluation
EVALUATION_MODEL_PATH = "NLU/part_B/bin/bert-base/weights.pt"  # Change this to your model path
# ====================================================================

# Set device and model name
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert_model_name = "bert-base-uncased"  # or "bert-base-uncased" 

# Define and if necessary create the path to the dataset
path = 'NLU/part_B/dataset'  # Change this to your dataset path

if __name__ == "__main__":
    
    if EVALUATION_MODE:
        print("📊 EVALUATION MODE")
        print(f"📊 Loading model from: {EVALUATION_MODEL_PATH}")
        
        # Check if model file exists
        if not os.path.exists(EVALUATION_MODEL_PATH):
            print(f"❌ Error: Model file not found at {EVALUATION_MODEL_PATH}")
            print("💡 Make sure to:")
            print("   1. Train a model first (set EVALUATION_MODE = True)")
            print("   2. Update EVALUATION_MODEL_PATH to point to your saved weights")
            exit(1)
    else:
        print("🔧 TRAINING MODE")
    
    train_raw, dev_raw = load_and_split_data(
        os.path.join(path,'train.json'),
        portion=0.10, 
        random_state=42
    )
    test_raw = load_data(os.path.join(path,'test.json'))

    # Print dataset sizes
    print('Train samples:' , len(train_raw))
    print('Test samples: ', len(test_raw))
    
    # Combine all data splits to extract unique slots and intents
    corpus = train_raw + dev_raw + test_raw

    # Extract unique slots and intents
    slots = set(sum([line['slots'].split() for line in corpus], []))
    intents = set([line['intent'] for line in corpus])

    # Load BERT tokenizer and create slot and intent mappings (also reverse)
    tokenizer = BertTokenizerFast.from_pretrained(bert_model_name)
    slot2id = {slot: i for i, slot in enumerate(sorted(slots))}
    intent2id = {intent: i for i, intent in enumerate(sorted(intents))}
    id2slot = {i: slot for slot, i in slot2id.items()}
    id2intent = {i: intent for intent, i in intent2id.items()}

    # Create BERT-compatible dataset for train, dev and test
    train_dataset = ATISDataset(train_raw, tokenizer, slot2id, intent2id)
    dev_dataset = ATISDataset(dev_raw, tokenizer, slot2id, intent2id)
    test_dataset = ATISDataset(test_raw, tokenizer, slot2id, intent2id)

    # Create DataLoader for batching and shuffling
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True) 
    dev_loader = DataLoader(dev_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    # =================== EVALUATION MODE ===================
    if EVALUATION_MODE:
        print(f"🔧 Model: JointBERT ({bert_model_name})")
        print(f"📊 Vocab - Slots: {len(slot2id)}, Intents: {len(intent2id)}")
        
        # Load the saved model state first to avoid downloading BERT weights
        try:
            print("📊 Loading saved model weights...")
            model_state = torch.load(EVALUATION_MODEL_PATH, map_location=device)
            
            # Initialize BERT config (this doesn't download weights)
            config = BertConfig.from_pretrained(bert_model_name)
            
            # Initialize model architecture without pre-trained weights
            model = JointBERT(
                config=config,
                num_intents=len(intent2id),
                num_slots=len(slot2id)
            ).to(device)
            
            # Load our saved weights with strict=False to handle version differences
            missing_keys, unexpected_keys = model.load_state_dict(model_state, strict=False)
            
            # if missing_keys:
            #     print(f"⚠️ Missing keys (likely due to version differences): {missing_keys}")
            # if unexpected_keys:
            #     print(f"⚠️ Unexpected keys: {unexpected_keys}")
            
            print("✅ Model weights loaded successfully")
            
            # Set model to evaluation mode
            model.eval()
            
            # Evaluate on test set
            print("📊 Evaluating model on test set...")
            intent_acc, slot_f1 = evaluate_model(model, test_loader, id2slot, device)
            
            print("\n📊 ================ EVALUATION RESULTS ================")
            print(f"📊 Test Slot F1 Score: {slot_f1:.4f}")
            print(f"📊 Test Intent Accuracy: {intent_acc:.4f}")
            print("📊 ====================================================")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("💡 Please check the model path and ensure the file exists")
            exit(1)
            
        # Exit after evaluation
        exit(0)
    # =================== END EVALUATION MODE ===================

    n_epochs = 10
    runs = 3
    
    print(f"🔧 Training with JointBERT ({bert_model_name})...")
    print(f"🔧 Vocab - Slots: {len(slot2id)}, Intents: {len(intent2id)}")
    print(f"🔧 Training parameters - Epochs: {n_epochs}, Runs: {runs}")

    # Call the new training loop function
    slot_f1s, intent_accs = training_loop(
        runs=runs,
        n_epochs=n_epochs,
        train_loader=train_loader,
        dev_loader=dev_loader,
        test_loader=test_loader,
        bert_model_name=bert_model_name,
        intent2id=intent2id,
        slot2id=slot2id,
        id2slot=id2slot,
        path=path,
        device=device
    )

    # After all runs, print the average and standard deviation of the metrics
    print(f'\n🔧 === JOINT BERT TRAINING RESULTS ===')
    print('📊 Slot F1', round(np.mean(slot_f1s), 3), '+-', round(np.std(slot_f1s), 3))
    print('📊 Intent Acc', round(np.mean(intent_accs), 3), '+-', round(np.std(intent_accs), 3))