# Import libraries and functions
from functions import *
from utils import *
from torch.utils.data import DataLoader
from functools import partial
import torch.optim as optim
from tqdm import tqdm
import copy
import os
import torch.nn as nn
import math
import numpy as np
from model import *

# Configuration parameters - modify this to choose model configuration
MODEL_CONFIG = "LSTM_DROPOUT_ADAMW"  # Options: "RNN", "LSTM", "LSTM_DROPOUT", "LSTM_DROPOUT_ADAMW" <-- Change this to select the model configuration
# "RNN": Baseline RNN
# "LSTM": LSTM model  
# "LSTM_DROPOUT": LSTM + Dropout Layers
# "LSTM_DROPOUT_ADAMW": LSTM + Dropout Layers + AdamW optimizer

if __name__ == "__main__":
    # If necessary, modify the path with the absolute path of the dataset
    train_raw = read_file('LM/part_A/dataset/PennTreeBank/ptb.train.txt')
    dev_raw = read_file('LM/part_A/dataset/PennTreeBank/ptb.valid.txt')
    test_raw = read_file('LM/part_A/dataset/PennTreeBank/ptb.test.txt')

    # Load dataset
    lang = Lang(train_raw, ["<pad>", "<eos>"])
    vocab_len = len(lang.word2id)
    train_dataset = PennTreeBank(train_raw, lang)
    dev_dataset = PennTreeBank(dev_raw, lang)
    test_dataset = PennTreeBank(test_raw, lang)
    
    # Initialize data loaders
    train_loader = DataLoader(train_dataset, batch_size=10, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]),  shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=128, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]))
    test_loader = DataLoader(test_dataset, batch_size=128, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]))

    # Configure model and hyperparameters based on chosen configuration
    if MODEL_CONFIG == "RNN":
        print("Using: Baseline RNN")
        hid_size = 100
        emb_size = 100
        lr = 0.1
        model = LM_RNN(emb_size, hid_size, vocab_len, pad_index=lang.word2id["<pad>"]).to(DEVICE)
        optimizer = optim.SGD(model.parameters(), lr=lr)
        patience = 3
        
    elif MODEL_CONFIG == "LSTM":
        print("Using: LSTM")
        hid_size = 300
        emb_size = 300
        lr = 2
        model = LM_LSTM(emb_size, hid_size, vocab_len, pad_index=lang.word2id["<pad>"]).to(DEVICE)
        optimizer = optim.SGD(model.parameters(), lr=lr)
        patience = 3
        
    elif MODEL_CONFIG == "LSTM_DROPOUT":
        print("Using: LSTM + Dropout Layers")
        hid_size = 300
        emb_size = 300
        lr = 2
        model = LM_LSTM_DL(emb_size, hid_size, vocab_len, pad_index=lang.word2id["<pad>"]).to(DEVICE)
        optimizer = optim.SGD(model.parameters(), lr=lr)
        patience = 3
        
    elif MODEL_CONFIG == "LSTM_DROPOUT_ADAMW":
        print("Using: LSTM + Dropout Layers + AdamW")
        hid_size = 250
        emb_size = 300
        lr = 0.001
        model = LM_LSTM_DL(emb_size, hid_size, vocab_len, pad_index=lang.word2id["<pad>"]).to(DEVICE)
        optimizer = optim.AdamW(model.parameters(), lr=lr)
        patience = 3
        
    else:
        raise ValueError("MODEL_CONFIG must be 'RNN', 'LSTM', 'LSTM_DROPOUT', or 'LSTM_DROPOUT_ADAMW'")

    # Common hyperparameters
    clip = 5
    model.apply(init_weights)
    
    criterion_train = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"])
    criterion_eval = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"], reduction='sum')       

    n_epochs = 100
    losses_train = []
    losses_dev = []    
    perplexity_list = []
    sampled_epochs = []
    best_ppl = math.inf
    best_model = None
    pbar = tqdm(range(1,n_epochs))
    
    print(f"Training with {MODEL_CONFIG} configuration...")
    print(f"Model: {type(model).__name__}, Optimizer: {type(optimizer).__name__}, LR: {lr}")
    print(f"Hidden size: {hid_size}, Embedding size: {emb_size}")
    
    # Train loop   
    for epoch in pbar:
            loss = train_loop(train_loader, optimizer, criterion_train, model, clip)    
            if epoch % 1 == 0:  # validate every epoch
                sampled_epochs.append(epoch)
                losses_train.append(np.asarray(loss).mean())
                ppl_dev, loss_dev = eval_loop(dev_loader, criterion_eval, model)
                losses_dev.append(np.asarray(loss_dev).mean())
                perplexity_list.append(ppl_dev)
                pbar.set_description("PPL: %f" % ppl_dev)
                if  ppl_dev < best_ppl: 
                    best_ppl = ppl_dev
                    best_model = copy.deepcopy(model).to('cpu')
                    patience = 3
                else:
                    patience -= 1
                    
                if patience <= 0: # Early stopping 
                    break 

    best_model.to(DEVICE)
    # evaluate the best model on the test set
    final_ppl,  _ = eval_loop(test_loader, criterion_eval, best_model)    
            
    print('Test ppl: ', final_ppl)

    # Save the model and create plots
    folder = create_new_report_directory()
    plot_loss_curve(sampled_epochs, losses_train, losses_dev, os.path.join(folder, 'plot.png'))
    plot_perplexity_curve(sampled_epochs, perplexity_list, os.path.join(folder, 'ppl_plot.png'))
    torch.save(best_model.state_dict(), os.path.join(folder, "weights.pt"))
    generate_training_report(sampled_epochs[-1], n_epochs, lr, hid_size, emb_size, str(type(model)), str(type(optimizer)),final_ppl, os.path.join(folder,"report.txt"))




