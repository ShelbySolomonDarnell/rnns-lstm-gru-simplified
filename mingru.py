import torch.nn as nn
import torch.nn.functional as F
from   modelutils import *
from   common     import *

class MinGRU(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MinGRU, self).__init__()
        self.linear_z = nn.Linear(input_size, hidden_size)
        self.linear_h = nn.Linear(input_size, hidden_size)

    def forward(self, x, h_0):
        f_name = 'MinGRU.forward'
        k = self.linear_z(x)
        log_z = -F.softplus(-k)
        tellem.debug('[{0}]\n\tShape of k {1}\n\tShape of log z {2}'.format(f_name, k.shape, log_z.shape))
        log_coeffs = -F.softplus(k)
        log_h_0 = log_g(h_0)
        unsqueezed_log_h_0 = log_h_0.unsqueeze(1)
        log_tilde_h = log_g(self.linear_h(x))
        tellem.debug('[{0}]\n\tShape of log_coeffs {1}\n\tShape of log h_0 {2}\n\tShape of log_tilde_h {3}'.format(f_name, log_coeffs.shape, log_h_0.shape, log_tilde_h.shape))
        log_z_tilde_h = [log_z+log_tilde_h]
        tellem.debug('[{0}]\n\tShape of concatenation of h_0, log z plus tilde_h {1}'.format(f_name, torch.cat([unsqueezed_log_h_0, log_z, log_tilde_h], dim=1).shape))
        tellem.debug('[{0}]\n\tShape of unsqueezed log_h_0 {1}'.format(f_name, unsqueezed_log_h_0.shape))
        h = parallel_scan_log(log_coeffs, torch_cat_with_check([unsqueezed_log_h_0, log_z + log_tilde_h], dim=1))
        return h
