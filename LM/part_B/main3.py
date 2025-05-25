from functions import *
from utils import *
from tqdm import tqdm
import copy
import torch.nn as nn
import torch.optim as optim
from functools import partial
from torch.utils.data import DataLoader
import math
import numpy as np
import torch.optim.lr_scheduler as lr_scheduler
from model import *

# Configuration parameters - modify these to choose model configuration
MODEL_CONFIG = "FULL"  # Options: "BASE", "VARDROP", "FULL"   <-- Change this to select the model configuration
# "BASE": LSTM + Weight Tying
# "VARDROP": LSTM + Weight Tying + Variational Dropout  
# "FULL": LSTM + Weight Tying + Variational Dropout + AvSGD

INTERVAL = 3

if __name__ == "__main__":
    # Load datasets
    train_raw = read_file('LM/part_B/dataset/PennTreeBank/ptb.train.txt')
    dev_raw = read_file('LM/part_B/dataset/PennTreeBank/ptb.valid.txt')
    test_raw = read_file('LM/part_B/dataset/PennTreeBank/ptb.test.txt')

    # Build vocabulary and datasets
    lang = Lang(train_raw, ["<pad>", "<eos>"])
    vocab_len = len(lang.word2id)
    train_dataset = PennTreeBank(train_raw, lang)
    dev_dataset = PennTreeBank(dev_raw, lang)
    test_dataset = PennTreeBank(test_raw, lang)
    
    # Data loaders for batching
    train_loader = DataLoader(train_dataset, batch_size=10, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]),  shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=32, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]))
    test_loader = DataLoader(test_dataset, batch_size=32, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]))

    # Hyperparameters
    hid_size = 400
    emb_size = 400 
    clip = 5 
    n_epochs = 100

    # Configure model and training based on chosen configuration
    if MODEL_CONFIG == "BASE":
        print("Using: LSTM + Weight Tying")
        model = LM_LSTM_WT(emb_size, hid_size, vocab_len, pad_index=lang.word2id["<pad>"]).to(DEVICE)
        lr = 0.010
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = None
        use_asgd = False
        patience = 5
        
    elif MODEL_CONFIG == "VARDROP":
        print("Using: LSTM + Weight Tying + Variational Dropout")
        model = LM_LSTM(emb_size, hid_size, vocab_len, pad_index=lang.word2id["<pad>"]).to(DEVICE)
        lr = 0.010
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = None
        use_asgd = False
        patience = 5
        
    elif MODEL_CONFIG == "FULL":
        print("Using: LSTM + Weight Tying + Variational Dropout + AvSGD")
        model = LM_LSTM(emb_size, hid_size, vocab_len, pad_index=lang.word2id["<pad>"]).to(DEVICE)
        lr = 2
        optimizer = optim.SGD(model.parameters(), lr=lr)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.75)
        use_asgd = True
        patience = 3
        
    else:
        raise ValueError("MODEL_CONFIG must be 'BASE', 'VARDROP', or 'FULL'")

    model.apply(init_weights)

    criterion_train = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"])
    criterion_eval = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"], reduction='sum')
    
    # Training variables
    losses_train = []
    losses_dev = []
    perplexity_list = []
    sampled_epochs = []
    best_ppl = math.inf
    best_loss = math.inf
    best_val_loss = []
    best_model = None
    pbar = tqdm(range(1, n_epochs))

    print(f"Training with {MODEL_CONFIG} configuration...")
    print(f"Model: {type(model).__name__}, Optimizer: {type(optimizer).__name__}, LR: {lr}")

    # Training loop
    for epoch in pbar:
        loss = train_loop(train_loader, optimizer, criterion_train, model, clip)    
        if epoch % 1 == 0:  # Validate every epoch
            sampled_epochs.append(epoch)
            losses_train.append(np.asarray(loss).mean())
            
            if use_asgd and 't0' in optimizer.param_groups[0]:  # If AvSGD is active
                # Temporarily use averaged weights for evaluation
                tmp = {}
                for prm in model.parameters():
                    tmp[prm] = prm.data.clone()
                    prm.data = optimizer.state[prm]['ax'].clone()
                ppl_dev, loss_dev = eval_loop(dev_loader, criterion_eval, model)
                # Restore original weights
                for prm in model.parameters():
                    prm.data = tmp[prm].clone()
            else:
                ppl_dev, loss_dev = eval_loop(dev_loader, criterion_eval, model)
                if loss_dev < best_loss:
                    best_loss = loss_dev
                # Trigger AvSGD if using FULL configuration and validation loss is non-monotonic
                if (use_asgd and 't0' not in optimizer.param_groups[0] and 
                    len(best_val_loss) > INTERVAL and loss_dev > min(best_val_loss[:-INTERVAL])):
                    print("Triggered, switching to ASGD")
                    optimizer = optim.ASGD(model.parameters(), lr=lr, t0=0, lambd=0.)
                if use_asgd:
                    best_val_loss.append(loss_dev)
            
            losses_dev.append(np.asarray(loss_dev).mean())
            perplexity_list.append(ppl_dev)
            pbar.set_description("PPL: %f" % ppl_dev)
            
            if ppl_dev < best_ppl: 
                best_ppl = ppl_dev
                best_model = copy.deepcopy(model).to('cpu')
                patience = 5 if not use_asgd else 3
            else: 
                patience -= 1
                
            if patience <= 0:  # Early stopping
                break
            
        if scheduler is not None:
            scheduler.step()
    
    best_model.to(DEVICE)
    # Final evaluation on the test set
    final_ppl, _ = eval_loop(test_loader, criterion_eval, best_model)  
    print('Test ppl: ', final_ppl)

    # Save results and plots
    folder = create_new_report_directory()
    plot_loss_curve(sampled_epochs, losses_train, losses_dev, os.path.join(folder, 'plot.png'))
    plot_perplexity_curve(sampled_epochs, perplexity_list, os.path.join(folder, 'ppl_plot.png'))
    torch.save(best_model.state_dict(), os.path.join(folder, "weights.pt"))
    generate_training_report(sampled_epochs[-1], n_epochs, lr, hid_size, emb_size, str(type(model)), str(type(optimizer)), final_ppl, os.path.join(folder, "report.txt"))