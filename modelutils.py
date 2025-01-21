import re
#import nltk
import inspect
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm   import tqdm
from common import tellem

#nltk.download('wordnet')
#from nltk.stem import WordNetLemmatizer


def g(x):
    return torch.where(x >= 0, x + 0.5, torch.sigmoid(x))

def log_g(x):
    return torch.where(x >= 0, torch.log(F.relu(x) + 0.5), -F.softplus(-x))

def replace_char_in_txt(chr_lst, txt):
    for the_chr in chr_lst:
        txt = txt.replace("{0}".format(the_chr), ' ')
    return txt
    #return re.sub(r'[.]', '', txt)

"""
def return_vocabulary(chr_lst, txt):
    vocab   = {}
    wnl     = WordNetLemmatizer()
    newtext = replace_char_in_txt([",","\'","(",")","\"","\\","\t","\n","\r","!","?","|","[","]","-","`",".",";"], txt)
    newtext = newtext.lower()
    newtext = newtext.split()
    ndx = 1
    for aword in newtext:
        aword = aword.strip()
        aword = wnl.lemmatize(aword)
        vocab_value = vocab.get(aword)
        if vocab_value == None:
            vocab.setdefault(aword,1)
        else:
            vocab[aword] = vocab_value + 1
    '''
        #print('[{1}] {0}\n'.format(aword, ndx))
    '''
    print_list(vocab)
    tellem.info('{0} words in the vocabulary.'.format(len(vocab)))
    #print('{0} words in the vocabulary.'.format(len(vocab)))
"""

def print_list(the_lst):
    ndx = 1
    for aword in the_lst:
        #print('[{0}] {1}'.format(ndx, aword))
        tellem.info('[{0}] {1}'.format(ndx, aword))
        ndx += 1


def parallel_scan_log(log_coeffs, log_values):
    f_name = inspect.stack()[0][3]
    tellem.debug('[{0}]\n\tLog coefficients shape {1}\n\tLog Values shape {2}'.format(f_name, log_coeffs.shape, log_values.shape))
    #a_star = F.pad(torch.cumsum(log_coeffs, dim=1), (1, 0))
    a_star = torch.cumsum(log_coeffs, dim=1)
    tellem.debug('[{0}]\n\tA Star shape {1}\n\tA star unsqueeze -1 shape {2}'.format(f_name, a_star.shape, a_star.unsqueeze(-1).shape))

    selected_log_values = log_values[:,0:100,:]
    #log_h0_plus_b_star = torch.logcumsumexp(log_values - a_star.unsqueeze(-1), dim=1)
    #split_log_values = log_values.select(1, 1)#, dim=1, index=100)
    tellem.debug('[{0}]\n\tLog values shape {1}\n\tSelected log values shape {2}'.format(f_name,log_values.shape, selected_log_values.shape))
    #raise Exception('Testing')
    log_h0_plus_b_star = torch.logcumsumexp(selected_log_values - a_star, dim=1)
    #log_h = a_star.unsqueeze(-1) + log_h0_plus_b_star
    log_h = a_star + log_h0_plus_b_star
    return torch.exp(log_h)
    #return torch.exp(a_star)

"""
def get_vocabulary(file_path):
    with open(file_path, 'r') as f:
        text = f.read()
    return return_vocabulary([",","\'","(",")","\"","\\","\t","\n","\r","!","?","|","[","]","-","`",".",";"],text)
"""

def load_shakespeare_data(file_path, seq_length=100):
    with open(file_path, 'r') as f:
        text = f.read()
    #vocab = return_vocabulary([",","\'","(",")","\"","\\","\t","\n","\r","!","?","|","[","]","-","`",".",";"],text)
    chars = sorted(list(set(text)))
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}

    data = [char_to_idx[ch] for ch in text]
    num_sequences = len(data) // seq_length

    X = torch.tensor([data[i:i+seq_length] for i in range(0, num_sequences*seq_length, seq_length)])
    y = torch.tensor([data[i+1:i+seq_length+1] for i in range(0, num_sequences*seq_length, seq_length)])

    return X, y, char_to_idx, idx_to_char

# Training function
def train(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader, desc="Training"):
        x, y = batch
        x, y = x.to(device), y.to(device)
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
