import os
import math
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Training loop for a model
def train_loop(data, optimizer, criterion, model, clip=5):
    model.train()  # Set the model to training mode
    loss_array = []  # Store the loss values
    number_of_tokens = []  # Store the number of tokens per sample
    
    for sample in data:
        optimizer.zero_grad()  # Reset gradients
        output = model(sample['source'])  # Forward pass
        loss = criterion(output, sample['target'])  # Calculate loss
        loss_array.append(loss.item() * sample["number_tokens"])  # Accumulate loss weighted by tokens
        number_of_tokens.append(sample["number_tokens"])  # Accumulate token count
        loss.backward()  # Backpropagation
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)  # Gradient clipping to prevent exploding gradients
        optimizer.step()  # Update model parameters
        
    # Return average loss per token
    return sum(loss_array) / sum(number_of_tokens)

# Evaluation loop to compute perplexity and loss
def eval_loop(data, eval_criterion, model):
    model.eval()  # Set the model to evaluation mode
    loss_array = []  # Store the loss values
    number_of_tokens = []  # Store the number of tokens per sample
    with torch.no_grad():  # Disable gradient calculation for evaluation
        for sample in data:
            output = model(sample['source'])  # Forward pass
            loss = eval_criterion(output, sample['target'])  # Calculate loss
            loss_array.append(loss.item())  # Accumulate loss
            number_of_tokens.append(sample["number_tokens"])  # Accumulate token count
            
    # Calculate perplexity and average loss per token
    ppl = math.exp(sum(loss_array) / sum(number_of_tokens))
    loss_to_return = sum(loss_array) / sum(number_of_tokens)
    return ppl, loss_to_return

# Initialize model weights
def init_weights(mat):
    for m in mat.modules():
        # For recurrent layers (GRU, LSTM, RNN)
        if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
            for name, param in m.named_parameters():
                # Initialize input-hidden weights with Xavier initialization
                if 'weight_ih' in name:
                    for idx in range(4):
                        mul = param.shape[0] // 4
                        torch.nn.init.xavier_uniform_(param[idx * mul:(idx + 1) * mul])
                # Initialize hidden-hidden weights orthogonally
                elif 'weight_hh' in name:
                    for idx in range(4):
                        mul = param.shape[0] // 4
                        torch.nn.init.orthogonal_(param[idx * mul:(idx + 1) * mul])
                # Set biases to zero
                elif 'bias' in name:
                    param.data.fill_(0)
        # For linear layers
        elif type(m) == nn.Linear:
            torch.nn.init.uniform_(m.weight, -0.01, 0.01)  # Uniform initialization
            if m.bias is not None:
                m.bias.data.fill_(0.01)  # Set bias to a small positive value

# Get the latest index of files in a directory based on a prefix
def get_latest_index_in_directory(directory, prefix):
    files = os.listdir(directory)  # List all files in the directory
    indices = []
    for file in files:
        if file.startswith(prefix):  # Check if file name starts with the given prefix
            try:
                # Extract numeric part of the file name and add to indices
                index = int(str(file[len(prefix):]))  
                indices.append(index)
            except ValueError:
                pass
    return max(indices) if indices else -1  # Return the maximum index found or -1 if no files match

# Plot training and validation loss over epochs
def plot_loss_curve(epochs, train_loss, validation_loss, filename):
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss, label='Training Loss', marker='o')  # Plot training loss
    plt.plot(epochs, validation_loss, label='Validation Loss', marker='s')  # Plot validation loss
    plt.title('Training and Validation Loss')  
    plt.xlabel('Epochs')  
    plt.ylabel('Loss')  
    plt.legend()  
    plt.grid(True)  
    plt.tight_layout()  # Adjust layout
    plt.savefig(filename)  # Save the plot

# Plot validation perplexity over epochs
def plot_perplexity_curve(epochs, perplexity_values, filename):
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, perplexity_values, label='Validation Perplexity', marker='o')  # Plot perplexity
    plt.title('Validation Perplexity')  
    plt.xlabel('Epochs')  
    plt.ylabel('Perplexity')  
    plt.legend()  
    plt.grid(True)  
    plt.tight_layout()  # Adjust layout
    plt.savefig(filename)  # Save the plot

# Generate a training report with relevant details
def generate_training_report(epochs, total_epochs, learning_rate, hidden_layer_size, embedding_size, model_type, optimizer_type, final_perplexity, report_filename):
    with open(report_filename, 'w') as report_file:
        # Write key training information to the file
        report_file.write(f'Epochs completed: {epochs} \n')
        report_file.write(f'Total epochs: {total_epochs} \n')
        report_file.write(f'Learning rate: {learning_rate} \n')
        report_file.write(f'Hidden layer size: {hidden_layer_size} \n')
        report_file.write(f'Embedding size: {embedding_size} \n')
        report_file.write(f'Model: {model_type} \n')
        report_file.write(f'Optimizer: {optimizer_type} \n')
        report_file.write(f'Final Perplexity: {final_perplexity} \n')

# Create a new folder for saving reports with the next available index
def create_new_report_directory():
    base_directory = '/home/disi/nlu/LM/part_B/reports/test'
    os.makedirs(os.path.dirname(base_directory), exist_ok=True)  # Create directory if it doesn't exist

    # Get the latest index from the directory and increment it for the new folder
    last_index = get_latest_index_in_directory(os.path.dirname(base_directory), os.path.basename(base_directory))
    new_foldername = f'{base_directory}{last_index + 1:02d}'
    os.mkdir(new_foldername)  # Create a new folder
    return new_foldername