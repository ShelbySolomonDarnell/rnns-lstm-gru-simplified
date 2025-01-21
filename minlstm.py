import torch.nn as nn
import torch.nn.functional as F
from common     import *
from modelutils import  *

class MinLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MinLSTM, self).__init__()
        self.linear_f = nn.Linear(input_size, hidden_size)
        self.linear_i = nn.Linear(input_size, hidden_size)
        self.linear_h = nn.Linear(input_size, hidden_size)

    def forward(self, x, h_0):
        diff = F.softplus(-self.linear_f(x)) - F.softplus(-self.linear_i(x))
        log_f = -F.softplus(diff)
        log_i = -F.softplus(-diff)
        log_h_0 = log_g(h_0)
        unsqueezed_log_h_0 = log_h_0.unsqueeze(1)
        log_tilde_h = log_g(self.linear_h(x))
        #h = parallel_scan_log(log_f, torch.cat([log_h_0, log_i + log_tilde_h], dim=1))
        h = parallel_scan_log(log_f, torch_cat_with_check([unsqueezed_log_h_0, log_i + log_tilde_h], dim=1))
        return h
