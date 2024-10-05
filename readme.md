# Minimal RNN Language Model

This project implements the minimal RNN architectures (minGRU and minLSTM) proposed in the paper "Were RNNs All We Needed?" by Leo Feng et al. It applies these models to a character-level language modeling task using Shakespeare's text.

## Paper Reference

Feng, L., Tung, F., Ahmed, M.O., Bengio, Y., & Hajimirsadeghi, H. (2024). Were RNNs All We Needed? arXiv preprint arXiv:2410.01201v1.

## Key Equations

The core innovations of this paper lie in the simplified RNN architectures. Here are the key equations:

### minGRU

The minGRU update is given by:

$h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$

where:

$z_t = \sigma(W_z x_t)$
$\tilde{h}_t = W_h x_t$

### minLSTM

The minLSTM update is given by:

$h_t = f'_t \odot h_{t-1} + i'_t \odot \tilde{h}_t$

where:

$f_t = \sigma(W_f x_t)$
$i_t = \sigma(W_i x_t)$
$f'_t = \frac{f_t}{f_t + i_t}$
$i'_t = \frac{i_t}{f_t + i_t}$
$\tilde{h}_t = W_h x_t$

## Log-space Implementation

For numerical stability, the actual implementation uses log-space computations. The key function is:

$g(x) = \begin{cases} 
x + 0.5, & \text{if } x \geq 0 \\
\sigma(x), & \text{otherwise}
\end{cases}$

