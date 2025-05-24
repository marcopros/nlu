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

# If necessary, modify the path with the absolute path of the dataset
path = '/home/disi/nlu/NLU/part_A'
PAD_TOKEN = 0

if __name__ == "__main__":
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

    # Model and training hyperparameters
    hid_size = 200
    emb_size = 300
    lr = 0.0001
    clip = 5

    out_slot = len(lang.slot2id)
    out_int = len(lang.intent2id)
    vocab_len = len(lang.word2id)

    n_epochs = 200
    runs = 5
    
    # Run multiple training runs for robustness
    slot_f1s, intent_acc = [], []
    for x in tqdm(range(0, runs)):
        # Model, optimizer, and loss setup
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
        for x in range(1,n_epochs):
            loss = train_loop(train_loader, optimizer, criterion_slots, 
                            criterion_intents, model)
            if x % 5 == 0:
                sampled_epochs.append(x)
                losses_train.append(np.asarray(loss).mean())
                results_dev, intent_res, loss_dev = eval_loop(dev_loader, criterion_slots, 
                                                            criterion_intents, model, lang)
                losses_dev.append(np.asarray(loss_dev).mean())
                f1 = results_dev['total']['f']

                if f1 > best_f1:
                    best_f1 = f1
                else:
                    patience -= 1
                if patience <= 0: # Early stopping
                    break

        # Evaluate on test set after training
        results_test, intent_test, _ = eval_loop(test_loader, criterion_slots, 
                                                criterion_intents, model, lang)
        intent_acc.append(intent_test['accuracy'])
        slot_f1s.append(results_test['total']['f'])
    slot_f1s = np.asarray(slot_f1s)
    intent_acc = np.asarray(intent_acc)
    print('Slot F1', round(slot_f1s.mean(),3), '+-', round(slot_f1s.std(),3))
    print('Intent Acc', round(intent_acc.mean(), 3), '+-', round(slot_f1s.std(), 3))

    # Save plots and report
    folder_name = create_report_folder()
    generate_plots(sampled_epochs, losses_train, losses_dev, os.path.join(folder_name,"plot.png"))

    # Ensure report/bin directory exists
    report_bin_path = os.path.join(folder_name, "bin")
    os.makedirs(report_bin_path, exist_ok=True)

    # Save model checkpoint
    PATH = os.path.join(report_bin_path, "weights_1.pt")
    saving_object = {
        "epoch": x, 
        "model": model.state_dict(), 
        "optimizer": optimizer.state_dict(), 
        "w2id": lang.word2id, 
        "slot2id": lang.slot2id, 
        "intent2id": lang.intent2id
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
        slot_f1=round(results_test['total']['f'], 3), 
        intent_acc=round(intent_test['accuracy'], 3), 
        slot_f1_std=round(np.std(losses_dev), 3), 
        intent_acc_std=round(np.std(losses_train), 3), 
        name=os.path.join(folder_name, "report.txt")
    )


