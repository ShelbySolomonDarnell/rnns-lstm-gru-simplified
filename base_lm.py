import torch
import torch.nn as nn
from common  import tellem
from minlstm import MinLSTM
from mingru  import MinGRU



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
        x = self.embedding(x)
        h = torch.zeros(x.size(0), x.size(2)).to(x.device)
        for rnn in self.rnn_layers:
            x = rnn(x, h)
        return self.fc(x)

