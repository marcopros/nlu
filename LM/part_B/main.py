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


INTERVAL = 3

if __name__ == "__main__":
    train_raw = read_file('LM/part_B/dataset/PennTreeBank/ptb.train.txt')
    dev_raw = read_file('LM/part_B/dataset/PennTreeBank/ptb.valid.txt')
    test_raw = read_file('LM/part_B/dataset/PennTreeBank/ptb.test.txt')

    lang = Lang(train_raw, ["<pad>", "<eos>"])
    vocab_len = len(lang.word2id)
    train_dataset = PennTreeBank(train_raw, lang)
    dev_dataset = PennTreeBank(dev_raw, lang)
    test_dataset = PennTreeBank(test_raw, lang)
    
    # Initialize data loaders
    train_loader = DataLoader(train_dataset, batch_size=10, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]),  shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=32, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]))
    test_loader = DataLoader(test_dataset, batch_size=32, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]))

    # Hyperparameters    
    hid_size = 400
    emb_size = 400 

    lr = 2
    clip = 5 

    # Initialize the model and weights
    model = LM_LSTM(emb_size, hid_size, vocab_len, pad_index=lang.word2id["<pad>"]).to(DEVICE)
    model.apply(init_weights)

    optimizer = optim.SGD(model.parameters(), lr=lr)
    # scheduler for the learning rate
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.75)
    criterion_train = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"])
    criterion_eval = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"], reduction='sum')
    
    n_epochs = 100
    patience = 5
    losses_train = []
    losses_dev = []
    perplexity_list = []
    sampled_epochs = []
    best_ppl = math.inf
    best_loss = math.inf
    best_val_loss = []
    best_model = None
    pbar = tqdm(range(1,n_epochs))


    # Train loop
    for epoch in pbar:
            loss = train_loop(train_loader, optimizer, criterion_train, model, clip)    
            if epoch % 1 == 0:  # validate every epoch
                sampled_epochs.append(epoch)
                losses_train.append(np.asarray(loss).mean())
                
                if 't0' in optimizer.param_groups[0]:       # Triggered
                    tmp = {}
                    for prm in model.parameters():
                        tmp[prm] = prm.data.clone()
                        prm.data = optimizer.state[prm]['ax'].clone()
                    
                    ppl_dev, loss_dev = eval_loop(dev_loader, criterion_eval, model)        # evaluate the model
                    
                    for prm in model.parameters():
                        prm.data = tmp[prm].clone()
                    
                else:                                       # ASGD not triggered
                    ppl_dev, loss_dev = eval_loop(dev_loader, criterion_eval, model)        # evaluate the model
                    
                    if loss_dev < best_loss:
                        best_loss = loss_dev
                    
                    # check if not usign AvSGD, than check for non-monotonicity
                    if 't0' not in optimizer.param_groups[0] and (len(best_val_loss) > INTERVAL and loss_dev > min(best_val_loss[:-INTERVAL])):
                        print("Triggered, switching to ASGD")
                        optimizer = optim.ASGD(model.parameters(), lr=lr, t0=0,  lambd=0.)
                
                    best_val_loss.append(loss_dev)
                    
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
                
            scheduler.step()
        
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

    


