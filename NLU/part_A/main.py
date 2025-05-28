from functions import *
from utils import *
from model import *
from tqdm import tqdm
import os
from torch.utils.data import DataLoader
import numpy as np
import copy
import torch.optim as optim
from sklearn.model_selection import train_test_split
from collections import Counter

# Configuration parameters - modify this to choose model configuration
MODEL_CONFIG = "IAS_BIDIR_DROPOUT"  # Options: "IAS_BASELINE", "IAS_BIDIR", "IAS_BIDIR_DROPOUT"
# "IAS_BASELINE": Baseline IAS model (unidirectional LSTM, no dropout)
# "IAS_BIDIR": IAS + Bidirectional LSTM
# "IAS_BIDIR_DROPOUT": IAS + Bidirectional LSTM + Dropout layers

# =================== EVALUATION MODE CONFIGURATION ===================
# Set to True to enable evaluation mode (load pre-trained weights and evaluate on test set)
EVALUATION_MODE = True  
# Path to the saved model weights (.pt file) for evaluation
EVALUATION_MODEL_PATH = "NLU/part_A/bin/IAS_BIDIR_DROPOUT/weights_1.pt" 
# ====================================================================

# If necessary, modify the path with the absolute path of the dataset
path = 'NLU/part_A'  # Current directory since we're running from NLU/part_A/
PAD_TOKEN = 0

if __name__ == "__main__":
    
    if EVALUATION_MODE:
        print("📊 EVALUATION MODE")
        print(f"📊 Loading model from: {EVALUATION_MODEL_PATH}")
        
        # Check if model file exists
        if not os.path.exists(EVALUATION_MODEL_PATH):
            print(f"❌ Error: Model file not found at {EVALUATION_MODEL_PATH}")
            print("💡 Make sure to:")
            print("   1. Train a model first (set EVALUATION_MODE = False)")
            print("   2. Update EVALUATION_MODEL_PATH to point to your saved weights")
            exit(1)
    else:
        print("🔧 TRAINING MODE")
    
    # Load train and test data
    tmp_train_raw = load_data(os.path.join(path, 'dataset', 'train.json'))
    test_raw = load_data(os.path.join(path, 'dataset', 'test.json'))

    print('Train samples:' , len(tmp_train_raw))
    print('Test samples: ', len(test_raw))

    # Prepare stratified split for train/dev
    portion = 0.10
    intents = [x['intent'] for x in tmp_train_raw]
    count_y = Counter(intents)

    labels = []
    inputs = []
    mini_train = []

    # Separate rare classes for stratification
    for id_y, y in enumerate(intents):
        if count_y[y] > 1:
            inputs.append(tmp_train_raw[id_y])
            labels.append(y)
        else:
            mini_train.append(tmp_train_raw[id_y])

    # Stratified train/dev split
    X_train, X_dev, y_train, y_dev = train_test_split(inputs, labels, test_size=portion, 
                                                      random_state=42,
                                                      shuffle=True,
                                                      stratify=labels)
    X_train.extend(mini_train)
    train_raw = X_train
    dev_raw = X_dev

    y_test = [x['intent'] for x in test_raw]

    # Build vocabulary and mappings
    if EVALUATION_MODE:
        # In evaluation mode, load vocabulary from checkpoint to ensure consistency
        print("📋 Loading vocabulary from saved checkpoint...")
        checkpoint = torch.load(EVALUATION_MODEL_PATH, map_location='cpu')
        
        # Create a Lang object with the saved vocabularies
        lang = Lang([], [], [], 0)  # Empty initialization
        lang.word2id = checkpoint['w2id']
        lang.slot2id = checkpoint['slot2id'] 
        lang.intent2id = checkpoint['intent2id']
        
        # Build reverse mappings
        lang.id2word = {v: k for k, v in lang.word2id.items()}
        lang.id2slot = {v: k for k, v in lang.slot2id.items()}
        lang.id2intent = {v: k for k, v in lang.intent2id.items()}
        
        print(f"✅ Loaded vocabularies - Words: {len(lang.word2id)}, Slots: {len(lang.slot2id)}, Intents: {len(lang.intent2id)}")
    else:
        # In training mode, build vocabulary from data
        words = sum([x['utterance'].split() for x in train_raw], [])
        corpus = train_raw + dev_raw + test_raw
        slots = set(sum([line['slots'].split() for line in corpus], []))
        intents = set([line['intent'] for line in corpus])
        lang = Lang(words, intents, slots, 0)

    # Create datasets and dataloaders
    train_dataset = IntentsAndSlots(train_raw, lang)
    dev_dataset = IntentsAndSlots(dev_raw, lang)
    test_dataset = IntentsAndSlots(test_raw, lang)

    train_loader = DataLoader(train_dataset, batch_size=128, collate_fn=collate_fn,  shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=64, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=64, collate_fn=collate_fn)

    # Configure model and hyperparameters based on chosen configuration
    out_slot = len(lang.slot2id)
    out_int = len(lang.intent2id)
    vocab_len = len(lang.word2id)
    
    if MODEL_CONFIG == "IAS_BASELINE":
        print("Using: Baseline IAS (Unidirectional LSTM, no dropout)")
        hid_size = 200
        emb_size = 300
        lr = 0.0001
        model = ModelIAS_Baseline(hid_size, out_slot, out_int, emb_size, 
                                vocab_len, pad_index=PAD_TOKEN).to(device)
        
    elif MODEL_CONFIG == "IAS_BIDIR":
        print("Using: IAS + Bidirectional LSTM")
        hid_size = 200
        emb_size = 300
        lr = 0.0001
        model = ModelIAS_Bidir(hid_size, out_slot, out_int, emb_size, 
                             vocab_len, pad_index=PAD_TOKEN).to(device)
        
    elif MODEL_CONFIG == "IAS_BIDIR_DROPOUT":
        print("Using: IAS + Bidirectional LSTM + Dropout")
        hid_size = 200
        emb_size = 300
        lr = 0.0001
        model = ModelIAS(hid_size, out_slot, out_int, emb_size, 
                        vocab_len, pad_index=PAD_TOKEN).to(device)
        
    else:
        raise ValueError("MODEL_CONFIG must be 'IAS_BASELINE', 'IAS_BIDIR', or 'IAS_BIDIR_DROPOUT'")

    # Common hyperparameters
    clip = 5
    n_epochs = 200
    runs = 5
    
    # =================== EVALUATION MODE ===================
    if EVALUATION_MODE:
        print(f"🔧 Model configuration: {MODEL_CONFIG}")
        print(f"📊 Model: {type(model).__name__}, Hidden size: {hid_size}, Embedding size: {emb_size}")
        
        # Load the saved model state
        try:
            checkpoint = torch.load(EVALUATION_MODEL_PATH, map_location=device)
            
            # Verify model configuration matches
            if 'config' in checkpoint and checkpoint['config'] != MODEL_CONFIG:
                print(f"⚠️  Warning: Saved model config ({checkpoint['config']}) doesn't match current MODEL_CONFIG ({MODEL_CONFIG})")
                print("📝 Consider updating MODEL_CONFIG to match the saved model")
            
            # Load model weights
            model.load_state_dict(checkpoint['model'])
            print("✅ Model weights loaded successfully")
            
            # Verify vocabulary compatibility (optional check)
            if 'w2id' in checkpoint:
                saved_vocab_size = len(checkpoint['w2id'])
                current_vocab_size = len(lang.word2id)
                if saved_vocab_size != current_vocab_size:
                    print(f"⚠️  Warning: Vocabulary size mismatch (saved: {saved_vocab_size}, current: {current_vocab_size})")
            
            # Set model to evaluation mode
            model.eval()
            
            # Evaluate on test set
            print("📊 Evaluating model on test set...")
            criterion_slots = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
            criterion_intents = nn.CrossEntropyLoss()
            
            results_test, intent_test, test_loss = eval_loop(test_loader, criterion_slots, 
                                                           criterion_intents, model, lang)
            
            print("\n📊 ================ EVALUATION RESULTS ================")
            print(f"📊 Test Slot F1 Score: {results_test['total']['f']:.4f}")
            print(f"📊 Test Intent Accuracy: {intent_test['accuracy']:.4f}")
            print(f"📊 Test Loss: {np.mean(test_loss):.4f}")
            print("📊 ====================================================")
            
            # # Print detailed slot results
            # print("\n📊 Detailed Slot Results:")
            # for slot_name, metrics in results_test.items():
            #     if slot_name != 'total':
            #         print(f"📊   {slot_name}: F1={metrics['f']:.4f}, P={metrics['p']:.4f}, R={metrics['r']:.4f}")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("💡 Please check the model path and ensure the file exists")
            exit(1)
            
        # Exit after evaluation
        exit(0)
    # =================== END EVALUATION MODE ===================
    
    print(f"🔧 Training with {MODEL_CONFIG} configuration...")
    print(f"🔧 Model: {type(model).__name__}, Hidden size: {hid_size}, Embedding size: {emb_size}, LR: {lr}")
    
    # Run multiple training runs for robustness
    slot_f1s, intent_acc = [], []
    for run in tqdm(range(0, runs), desc=f"🔧 Training runs ({MODEL_CONFIG})"):
        # Reinitialize model for each run
        if MODEL_CONFIG == "IAS_BASELINE":
            model = ModelIAS_Baseline(hid_size, out_slot, out_int, emb_size, 
                                    vocab_len, pad_index=PAD_TOKEN).to(device)
        elif MODEL_CONFIG == "IAS_BIDIR":
            model = ModelIAS_Bidir(hid_size, out_slot, out_int, emb_size, 
                                 vocab_len, pad_index=PAD_TOKEN).to(device)
        elif MODEL_CONFIG == "IAS_BIDIR_DROPOUT":
            model = ModelIAS(hid_size, out_slot, out_int, emb_size, 
                            vocab_len, pad_index=PAD_TOKEN).to(device)
        
        model.apply(init_weights)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion_slots = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
        criterion_intents = nn.CrossEntropyLoss()
        
        patience = 3
        losses_train = []
        losses_dev = []
        sampled_epochs = []
        best_f1 = 0
        
        # Training loop with early stopping
        for epoch in range(1, n_epochs):
            loss = train_loop(train_loader, optimizer, criterion_slots, 
                            criterion_intents, model)
            if epoch % 5 == 0:
                sampled_epochs.append(epoch)
                losses_train.append(np.asarray(loss).mean())
                results_dev, intent_res, loss_dev = eval_loop(dev_loader, criterion_slots, 
                                                            criterion_intents, model, lang)
                losses_dev.append(np.asarray(loss_dev).mean())
                f1 = results_dev['total']['f']

                if f1 > best_f1:
                    best_f1 = f1
                    patience = 3  # Reset patience
                else:
                    patience -= 1
                if patience <= 0: # Early stopping
                    break

        # Evaluate on test set after training
        results_test, intent_test, _ = eval_loop(test_loader, criterion_slots, 
                                                criterion_intents, model, lang)
        intent_acc.append(intent_test['accuracy'])
        slot_f1s.append(results_test['total']['f'])
        
        print(f"🔧 Run {run+1}/{runs} - Slot F1: {results_test['total']['f']:.3f}, Intent Acc: {intent_test['accuracy']:.3f}")

    # Calculate statistics
    slot_f1s = np.asarray(slot_f1s)
    intent_acc = np.asarray(intent_acc)
    print(f'\n🔧 === {MODEL_CONFIG} TRAINING RESULTS ===')
    print(f'📊 Slot F1: {round(slot_f1s.mean(),3)} +- {round(slot_f1s.std(),3)}')
    print(f'📊 Intent Acc: {round(intent_acc.mean(), 3)} +- {round(intent_acc.std(), 3)}')

    # Save plots and report
    folder_name = create_report_folder()
    generate_plots(sampled_epochs, losses_train, losses_dev, os.path.join(folder_name,"plot.png"))

    # Ensure report/bin directory exists
    report_bin_path = os.path.join(folder_name, "bin")
    os.makedirs(report_bin_path, exist_ok=True)

    # Save model checkpoint
    PATH = os.path.join(report_bin_path, "weights_1.pt")
    saving_object = {
        "epoch": epoch, 
        "model": model.state_dict(), 
        "optimizer": optimizer.state_dict(), 
        "w2id": lang.word2id, 
        "slot2id": lang.slot2id, 
        "intent2id": lang.intent2id,
        "config": MODEL_CONFIG
    }
    torch.save(saving_object, PATH)

    # Generate textual report
    generate_report(
        epochs=sampled_epochs[-1], 
        number_epochs=n_epochs, 
        lr=lr, 
        hidden_size=hid_size, 
        emb_size=emb_size, 
        model=str(type(model)), 
        optimizer=str(type(optimizer)), 
        slot_f1=round(slot_f1s.mean(), 3), 
        intent_acc=round(intent_acc.mean(), 3), 
        slot_f1_std=round(slot_f1s.std(), 3), 
        intent_acc_std=round(intent_acc.std(), 3), 
        name=os.path.join(folder_name, "report.txt")
    )


