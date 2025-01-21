#!/home/shelbys/newcode/bin/python

import sys
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm

def g(x):
    return torch.where(x >= 0, x + 0.5, torch.sigmoid(x))

def log_g(x):
    return torch.where(x >= 0, torch.log(F.relu(x) + 0.5), -F.softplus(-x))

def parallel_scan_log(log_coeffs, log_values):
    f_name = 'parallel_scan_log'
    tellem.debug('========================Entering {0}========================'.format(f_name))
    #a_star = F.pad(torch.cumsum(log_coeffs, dim=1), (1, 0))
    a_star = F.pad(torch.cumsum(log_coeffs, dim=1), (1, 0))
    #log_h0_plus_b_star = torch.logcumsumexp(log_values - a_star.unsqueeze(-1), dim=1)
    a_star_unsqueeze = a_star.unsqueeze(-1)
    log_values_unsqueeze = log_values.unsqueeze(-1)
    tellem.debug('Shape of log values and unsqueeze {0}\n\t\t{1}'.format(log_values.shape,log_values_unsqueeze.shape))
    tellem.debug('Shape of A* & A*-unsqueeze {0}\n\t\t{1}'.format(a_star.shape, a_star_unsqueeze.shape))
    log_h0_plus_b_star = torch.logcumsumexp(log_values - a_star.unsqueeze(-1), dim=1)
    log_h = a_star.unsqueeze(-1) + log_h0_plus_b_star
    tellem.debug('Shape of hidden layer {0}'.format(log_h.shape))
    tellem.debug('========================Exiting {0}========================'.format(f_name))
    return torch.exp(log_h)

class MinGRU(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MinGRU, self).__init__()
        self.linear_z = nn.Linear(input_size, hidden_size)
        self.linear_h = nn.Linear(input_size, hidden_size)

    def forward(self, x, h_0):
        k = self.linear_z(x)
        log_z = -F.softplus(-k)
        log_coeffs = -F.softplus(k)
        log_h_0 = log_g(h_0)
        log_tilde_h = log_g(self.linear_h(x))
        #h = parallel_scan_log(log_coeffs, torch.cat([log_h_0, log_z + log_tilde_h], dim=1))
        h = parallel_scan_log(log_coeffs, torch.cat([log_h_0, log_z + log_tilde_h], dim=1))
        return h

class MinLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MinLSTM, self).__init__()
        self.linear_f = nn.Linear(input_size, hidden_size)
        self.linear_i = nn.Linear(input_size, hidden_size)
        self.linear_h = nn.Linear(input_size, hidden_size)

    def forward(self, x, h_0):
        f_name = 'MinLSTM.forward'
        tellem.debug('========================Entering {0}========================'.format(f_name))
        #try:
        tellem.debug('Shape of x {0}\n\tShape of h_0 {1}'.format(x.shape, h_0.shape))
        diff = F.softplus(-self.linear_f(x)) - F.softplus(-self.linear_i(x))
        log_f = -F.softplus(diff)
        log_i = -F.softplus(-diff)
        log_h_0 = log_g(h_0)
        log_tilde_h = log_g(self.linear_h(x))
        logs_2d = (log_i + log_tilde_h)[:,-1,:]
        tellem.debug('Shape of log_i {0}\n\t\tShape of log_tilde_h {1}'.format(log_i.shape, log_tilde_h.shape))
        tellem.debug('Shape of log_h_0 {0}\n\t\tShape of 2d log {1}'.format(log_h_0.shape,logs_2d.shape))
        forward_tensor = torch.cat([log_h_0, logs_2d], dim=1)
        #h = parallel_scan_log(log_f, torch.cat([log_h_0, log_i + log_tilde_h], dim=1))
        tellem.debug('Forward tensor shape {0}\n'.format(forward_tensor.shape))
        tellem.debug('Shape of log f {0}'.format(log_f.shape))
        h = parallel_scan_log(log_f, forward_tensor)
        tellem.debug('========================Exiting {0}========================'.format(f_name))
        return h
        #except Exception as exp:
        #    tellem.error('[{0}] {1} {2}'.format(f_name, exp, exp.__traceback__))

class LanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, rnn_type='minlstm'):
        super(LanguageModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn_type = rnn_type
        self.rnn_layers = nn.ModuleList()
        for _ in range(num_layers):
            if rnn_type == 'minlstm':
                self.rnn_layers.append(MinLSTM(embed_size, hidden_size))
            elif rnn_type == 'mingru':
                self.rnn_layers.append(MinGRU(embed_size, hidden_size))
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        f_name = 'LanguageModel.forward'
        x = self.embedding(x)
        tellem.debug('[{0}] X shape {1}'.format(f_name, x.shape))
        h = torch.zeros(x.size(0), x.size(2)).to(x.device)
        tellem.debug('[{0}] Shape of hidden layer {1}'.format(f_name, h.shape))
        #tellem.info('X {0}, hidden layer {1}'.format(x, h))
        for rnn in self.rnn_layers:
            x = rnn(x, h)
        return self.fc(x)

def load_shakespeare_data(file_path, seq_length=100):
    f_name = "load_shakespeare_data"
    tellem.debug('========================Entering {0}========================'.format(f_name))
    with open(file_path, 'r') as f:
        text = f.read()
    
    chars = sorted(list(set(text)))
    tellem.info('[{0}] Character count {1}\nSorted character list {2}'.format(f_name,len(chars),chars))
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}
    
    data = [char_to_idx[ch] for ch in text]
    num_sequences = len(data) // seq_length
   
    output = '[{0}] number of sequences {1} and sequence length {2}'.format(f_name, num_sequences, seq_length) 
    tellem.info(output)
    
    
    X = torch.tensor([data[i:i+seq_length] for i in range(0, num_sequences*seq_length, seq_length)])
    y = torch.tensor([data[i+1:i+seq_length+1] for i in range(0, num_sequences*seq_length, seq_length)])
    tellem.debug('[{0}] Shape of x tensor -> {1}, shape of y tensor -> {2}'.format(f_name, X.shape, y.shape))
    
    tellem.debug('========================Exiting {0}========================'.format(f_name))
    return X, y, char_to_idx, idx_to_char

# Training function
def train(model, train_loader, optimizer, criterion, device):
    f_name = "train"
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader, desc="Training"):
        x, y = batch
        x, y = x.to(device), y.to(device)
        tellem.debug('[{0}] X tensor shape -> {1}\n\tY tensor shape {2}'.format(f_name, x.shape, y.shape))
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output.view(-1, output.size(-1)), y.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

# Evaluation function
def evaluate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            x, y = batch
            x, y = x.to(device), y.to(device)
            output = model(x)
            loss = criterion(output.view(-1, output.size(-1)), y.view(-1))
            total_loss += loss.item()
    return total_loss / len(val_loader)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tellem.info("Device {0}".format(device))

    #vocab_size = 65  # Assuming ASCII characters
    vocab_size = 75  # Assuming ASCII characters
    embed_size = 384
    hidden_size = 384
    num_layers = 3
    batch_size = 64
    lr = 1e-3
    epochs = 50
    seq_length = 100
    training_set = '/home/shelbys/Datasets/shakespeare-dataset/text/all_less_chars.txt'
    # Load data
    #X, y, char_to_idx, idx_to_char = load_shakespeare_data("path_to_shakespeare_data.txt", seq_length)
    X, y, char_to_idx, idx_to_char = load_shakespeare_data(training_set, seq_length)
    dataset = TensorDataset(X, y)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    tellem.debug('Dataset length {0}\nTraining set size {1}\nTesting set size {2}'.format(len(dataset),train_size, val_size))
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    tellem.debug('train loader {0}\nval loader {1}'.format(train_loader, val_loader))
    model = LanguageModel(vocab_size, embed_size, hidden_size, num_layers, rnn_type='minlstm').to(device)
    #model = LanguageModel(vocab_size, embed_size, hidden_size, num_layers, rnn_type='mingru').to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    best_val_loss = float('inf')
    tellem.info('Optimizer {0}\nCriterion {1}\nBest Value loss {2}'.format(optimizer, criterion, best_val_loss))
   
    for epoch in range(epochs):
        train_loss = train(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate(model, val_loader, criterion, device)
        
        tellem.info(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pth")
        break
    
    tellem.info("Training completed.")
    '''
    '''
    

if __name__ == "__main__":
    tellem = logging.getLogger(__name__)
    logging.basicConfig(filename='logs/werernns.log', encoding='utf-8', level=logging.DEBUG)
    main()
