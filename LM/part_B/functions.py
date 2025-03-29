
import os
import math
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def train_loop(data, optimizer, criterion, model, clip=5):
    model.train()
    loss_array = []
    number_of_tokens = []
    
    for sample in data:
        optimizer.zero_grad() 
        output = model(sample['source'])
        loss = criterion(output, sample['target'])
        loss_array.append(loss.item() * sample["number_tokens"])
        number_of_tokens.append(sample["number_tokens"])
        loss.backward() 
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)  
        optimizer.step() 
        
    return sum(loss_array)/sum(number_of_tokens)


def eval_loop(data, eval_criterion, model):
    model.eval()
    loss_to_return = []
    loss_array = []
    number_of_tokens = []
    with torch.no_grad(): 
        for sample in data:
            output = model(sample['source'])
            loss = eval_criterion(output, sample['target'])
            loss_array.append(loss.item())
            number_of_tokens.append(sample["number_tokens"])
            
    ppl = math.exp(sum(loss_array) / sum(number_of_tokens))
    loss_to_return = sum(loss_array) / sum(number_of_tokens)
    return ppl, loss_to_return


def init_weights(mat):
    for m in mat.modules():
        if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    for idx in range(4):
                        mul = param.shape[0]//4
                        torch.nn.init.xavier_uniform_(param[idx*mul:(idx+1)*mul])
                elif 'weight_hh' in name:
                    for idx in range(4):
                        mul = param.shape[0]//4
                        torch.nn.init.orthogonal_(param[idx*mul:(idx+1)*mul])
                elif 'bias' in name:
                    param.data.fill_(0)
        else:
            if type(m) in [nn.Linear]:
                torch.nn.init.uniform_(m.weight, -0.01, 0.01)
                if m.bias != None:
                    m.bias.data.fill_(0.01)

# Function to get the latest index of files in a directory based on a base name
def get_latest_index_in_directory(directory, prefix):
    # Get all files in the directory
    files = os.listdir(directory)
    # Filter files that start with the specified prefix
    indices = []
    for file in files:
        if file.startswith(prefix):
            try:
                index = int(str(file[len(prefix):]))  # Extract numeric part of the file name
                indices.append(index)
            except ValueError:
                pass
    # Return the maximum index or -1 if no files match
    return max(indices) if indices else -1

# Function to plot training and validation loss over epochs
def plot_loss_curve(epochs, train_loss, validation_loss, filename):
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss, label='Training Loss', marker='o')  
    plt.plot(epochs, validation_loss, label='Validation Loss', marker='s')  
    plt.title('Training and Validation Loss')  
    plt.xlabel('Epochs')  
    plt.ylabel('Loss')  
    plt.legend()  
    plt.grid(True)  
    plt.tight_layout()
    plt.savefig(filename)

# Function to plot validation perplexity over epochs
def plot_perplexity_curve(epochs, perplexity_values, filename):
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, perplexity_values, label='Validation Perplexity', marker='o')  
    plt.title('Validation Perplexity')  
    plt.xlabel('Epochs')  
    plt.ylabel('Perplexity')  
    plt.legend()  
    plt.grid(True)  
    plt.tight_layout()
    plt.savefig(filename)

# Function to generate a report with training details
def generate_training_report(epochs, total_epochs, learning_rate, hidden_layer_size, embedding_size, model_type, optimizer_type, final_perplexity, report_filename):
    with open(report_filename, 'w') as report_file:
        report_file.write(f'Epochs completed: {epochs} \n')
        report_file.write(f'Total epochs: {total_epochs} \n')
        report_file.write(f'Learning rate: {learning_rate} \n')
        report_file.write(f'Hidden layer size: {hidden_layer_size} \n')
        report_file.write(f'Embedding size: {embedding_size} \n')
        report_file.write(f'Model: {model_type} \n')
        report_file.write(f'Optimizer: {optimizer_type} \n')
        report_file.write(f'Final Perplexity: {final_perplexity} \n')

# Function to create a new folder for saving reports with the next index
def create_new_report_directory():
    base_directory = '/home/disi/nlu/LM/part_B/reports/test'
    last_index = get_latest_index_in_directory(os.path.dirname(base_directory), os.path.basename(base_directory))
    new_foldername = f'{base_directory}{last_index + 1:02d}'
    os.mkdir(new_foldername)
    return new_foldername
