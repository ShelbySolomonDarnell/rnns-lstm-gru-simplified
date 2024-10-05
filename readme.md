# RNN Language Model

This project implements the minimal RNN architectures (minGRU and minLSTM) proposed in the paper "Were RNNs All We Needed?" by Leo Feng et al. It applies these models to a character-level language modeling task using Shakespeare's text.

## Paper Reference

Feng, L., Tung, F., Ahmed, M.O., Bengio, Y., & Hajimirsadeghi, H. (2024). Were RNNs All We Needed? arXiv preprint arXiv:2410.01201v1.

## Key Equations and Concepts

### Parallel Scan Algorithm

The core of the efficiency in these models comes from the parallel scan algorithm. For a sequence of operations:

$v_t = a_t \odot v_{t-1} + b_t$

The parallel scan computes all $v_t$ efficiently in parallel.

### minGRU

The minGRU update in log-space:

$\log(z_t) = -\text{softplus}(-k_t)$
$\log(1 - z_t) = -\text{softplus}(k_t)$

where $k_t = W_z x_t$

### minLSTM

The minLSTM update in log-space:

$\log(f'_t) = -\text{softplus}(\text{softplus}(-p_t) - \text{softplus}(-k_t))$
$\log(i'_t) = -\text{softplus}(\text{softplus}(-k_t) - \text{softplus}(-p_t))$

where $k_t = W_i x_t$ and $p_t = W_f x_t$

### Activation Function

The continuous activation function $g(x)$ and its log-space version:

$g(x) = \begin{cases} x + 0.5, & \text{if } x \geq 0 \\ \sigma(x), & \text{otherwise} \end{cases}$

$\log(g(x)) = \begin{cases} \log(x + 0.5), & \text{if } x \geq 0 \\ -\text{softplus}(-x), & \text{otherwise} \end{cases}$

### Parallel Scan in Log-Space

The parallel scan algorithm in log-space is implemented as:

$a^* = \text{pad}(\text{cumsum}(\log(a_{1:T})), (1, 0))$
$\log(h_0 + b^*) = \text{logcumsumexp}(\log(b_{1:T}) - a^*)$
$\log(h) = a^* + \log(h_0 + b^*)$

## Model Architecture

The language model consists of:
1. An embedding layer: $x_t = \text{Embed}(w_t)$
2. Multiple layers of either minLSTM or minGRU: $h_t = \text{RNN}(x_t, h_{t-1})$
3. A final linear layer for prediction: $y_t = Wh_t + b$

## Training

The model is trained using AdamW optimizer and CrossEntropyLoss:

$\mathcal{L} = -\sum_t \log p(w_t | w_{<t})$

where $p(w_t | w_{<t})$ is the predicted probability of the correct word.

## Usage

1. Ensure you have PyTorch installed.
2. Replace "path_to_shakespeare_data.txt" with the path to your Shakespeare dataset.
3. Run the script:
