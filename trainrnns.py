#!/home/shelbys/newcode/bin/python

import sys
import torch
import numpy as np
import configparser
import torch.optim as optim
from   torch.utils.data import DataLoader, TensorDataset
from   tqdm             import tqdm
from   colorama         import Fore, Style
from   common           import *
from   modelutils       import *
from   base_lm          import * 

"""
def train_rnn():
    f_name = 'TestRNNs.train_rnn'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tellem.info("[{0}] Device {1}".format(f_name, device))

    #vocab_size = 65  # Assuming ASCII characters
    vocab_size = 75  # Assuming ASCII characters
    embed_size = 384
    hidden_size = 384
    num_layers = 3
    batch_size = 64
    lr = 1e-3
    epochs = 50
    seq_length = 100
    training_set = cfg.get('DATASETS', 'shakespeare')
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
    #model = LanguageModel(vocab_size, embed_size, hidden_size, num_layers, rnn_type='minlstm').to(device)
    model = LanguageModel(vocab_size, embed_size, hidden_size, num_layers, rnn_type='mingru').to(device)
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
def tensor_squeeze_test():
    embed_size = 384
    batch_size = 64
    seq_length = 100
    training_set = cfg.get('DATASETS', 'shakespeare')

    tA = torch.randint(low=10, high=99, size=(batch_size, seq_length, embed_size), device=device)
    tB = torch.randint(low=10, high=99, size=(batch_size, embed_size), device=device)
    tB_unsqueezed = tB.unsqueeze(1)

    tellem.debug('[{0}]\n\t Tensor A shape -> {1}\n\t Tensor B shape -> {2}'.format(f_name, tA.shape, tB.shape))

    tellem.debug('[{0}]\n\t Tensor B unsqueezed shape -> {1}'.format(f_name, tB_unsqueezed.shape))

    tC = torch.cat((tA, tB_unsqueezed), dim=1)

    tellem.debug('[{0}]\n\t Tensor C shape -> {1}'.format(f_name, tC.shape))

"""

def main():
    f_name = 'TestRNNs.main'

    'Set the type of RNN to model'
    rnn_type = 'minlstm'

    'Set the model save file'
    the_model = None
    model_path = cfg.get('MODELS', 'path')
    if rnn_type == 'minlstm':
        the_model = '{0}{1}'.format(model_path, cfg.get('MODELS', 'lstm'))
    elif rnn_type == 'mingru':
        the_model = '{0}{1}'.format(model_path, cfg.get('MODELS', 'gru2'))


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tellem.info("[{0}] Device {1}".format(f_name, device))

    """ BEGIN test code " ""
    embed_size = 384
    batch_size = 64
    seq_length = 100
    training_set = cfg.get('DATASETS', 'shakespeare')

    tA = torch.randint(low=10, high=99, size=(batch_size, seq_length, embed_size), device=device)
    tB = torch.randint(low=10, high=99, size=(batch_size, embed_size), device=device)
    tB_unsqueezed = tB.unsqueeze(1)

    tellem.debug('[{0}]\n\t Tensor A shape -> {1}\n\t Tensor B shape -> {2}'.format(f_name, tA.shape, tB.shape))

    tellem.debug('[{0}]\n\t Tensor B unsqueezed shape -> {1}'.format(f_name, tB_unsqueezed.shape))

    tC = torch_cat_with_check((tA, tB_unsqueezed), dim=1)

    tellem.debug('[{0}]\n\t Tensor C shape -> {1}'.format(f_name, tC.shape))
    " " " END test code """
    """
    """

    vocab_size   = 75  # Assuming ASCII characters
    embed_size   = 384
    hidden_size  = 384
    num_layers   = 3
    batch_size   = 64
    lr           = 1e-3
    epochs       = 50
    seq_length   = 100
    training_set = cfg.get('DATASETS', 'shakespeare')
    # Load data
    #X, y, char_to_idx, idx_to_char = load_shakespeare_data("path_to_shakespeare_data.txt", seq_length)
    X, y, char_to_idx, idx_to_char = load_shakespeare_data(training_set, seq_length)
    tellem.info('Char to IDX -->\n{0}\nIDX to Char -->\n{1}'.format(char_to_idx, idx_to_char))
    dataset                        = TensorDataset(X, y)
    train_size                     = int(0.8 * len(dataset))
    val_size                       = len(dataset) - train_size
    tellem.debug('Dataset length {0}\nTraining set size {1}\nTesting set size {2}'.format(len(dataset),train_size, val_size))
    train_dataset, val_dataset     = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader                   = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader                     = DataLoader(val_dataset, batch_size=batch_size)
    tellem.debug('train loader {0}\nval loader {1}'.format(train_loader, val_loader))
    model                          = LanguageModel(vocab_size, 
                                                   embed_size, 
                                                   hidden_size, 
                                                   num_layers, 
                                                   rnn_type=rnn_type).to(device)
    optimizer                      = optim.AdamW(model.parameters(), lr=lr)
    criterion                      = nn.CrossEntropyLoss()
    best_val_loss                  = float('inf')
    tellem.info('Optimizer {0}\nCriterion {1}\nBest Value loss {2}'.format(optimizer, criterion, best_val_loss))
    '''
    for epoch in range(epochs):
        train_loss = train(model, train_loader, optimizer, criterion, device)
        val_loss   = evaluate(model, val_loader, criterion, device)
        
        tellem.info(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pth")
            torch.save(model.state_dict(), the_model)
        break
    
    tellem.info("Training completed.")
    ''' 


if __name__ == "__main__":
    main()
