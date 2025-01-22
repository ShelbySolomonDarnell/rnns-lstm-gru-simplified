import sys
import torch
#import torchtext
import configparser
import datetime
from common         import *
from base_lm        import LanguageModel
from datetime       import date
#from torchtext.data import get_tokenizer
from modelutils import load_shakespeare_data


#tokenizer = get_tokenizer('basic_english')

def standstill():
    ' do nothing '
    print( "Here is where we stand still." )

def generate(prompt, max_seq_len, temperature, model, vocab, device, seed=None):
    f_name = inspect.stack()[0][3]
    if seed is not None:
        torch.manual_seed(seed)

    ' model will be in eval mode when passed '
    #model.eval()

    #tokens = tokenizer(prompt)
    tokens = [char for char in prompt]
    indices = [vocab[t] for t in tokens]
    batch_size = 1
    '''
    hidden isn't needed as the hidden tensor is created in the LanguageModel class
    '''
    #hidden = model.init_hidden(batch_size, device)
    with torch.no_grad():
        for i in range(max_seq_len):
            src = torch.LongTensor([indices]).to(device)
            #prediction, hidden = model(src, hidden) 'this is original code'
            prediction, hidden = model(src)
            probs = torch.softmax(prediction[:, -1] / temperature, dim=-1)  
            prediction = torch.multinomial(probs, num_samples=1).item()    
            
            while prediction == vocab['<unk>']:
                prediction = torch.multinomial(probs, num_samples=1).item()

            if prediction == vocab['<eos>']:
                break

            indices.append(prediction)

    itos = vocab.get_itos()
    tokens = [itos[i] for i in indices]
    return tokens

def main():
    f_name = 'TestRNNs.main'

    max_seq_len = 100
    temperature = 0.7
    prompts = [
      "Where art thou", 
      "Romeo and Juliet",
    ]
    vocab_size = 75  # Assuming ASCII characters
    embed_size = 384
    hidden_size = 384
    num_layers = 3

    the_dataset = cfg.get('DATASETS', 'shakespeare')
    X, y, char_to_idx, idx_to_char = load_shakespeare_data(the_dataset)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tellem.info("[{0}] Device {1}".format(f_name, device))

    model_path = cfg.get('MODELS', 'path')
    lstm_model = '{0}{1}'.format(model_path, cfg.get('MODELS', 'lstm'))
    gru_model = '{0}{1}'.format(model_path, cfg.get('MODELS', 'gru'))
    #lstm_model = '{0}{1}_{2}'.format(model_path, datetime.datetime.now(), cfg.get('MODELS', 'lstm'))
    #gru_model = '{0}{1}_{2}'.format(model_path, datetime.datetime.now(), cfg.get('MODELS', 'gru'))

    tellem.info('[{0}\n\t LSTM model path {1}\n\t GRU model path {2}]'.format(f_name, lstm_model, gru_model))

    model = LanguageModel(vocab_size, embed_size, hidden_size, num_layers, rnn_type='mingru').to(device)
    #model.load_state_dict(torch.load(lstm_model, weights_only=True))
    model.load_state_dict(torch.load(gru_model, weights_only=True))
    model.to(device)
    print (model.eval())
    for prompt in prompts:
        tokens = generate(prompt, max_seq_len, temperature, model, char_to_idx, device, seed=None)
        print(tokens)


if __name__ == "__main__":
    #standstill()
    main()
