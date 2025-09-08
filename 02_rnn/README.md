# 02_rnn

A Recurrent Neural Network (RNN) implementation in Ruby that demonstrates sequence modeling and character-level language generation with manual backpropagation through time (BPTT).

## Overview

This folder contains a vanilla RNN implementation that learns to model character sequences. The network is trained on the text "hello world!" and learns to predict the next character given a sequence of previous characters.

## Files

- `02_rnn.rb` - RNN implementation with manual BPTT and character generation

## Neural Network Architecture

- **Input Layer**: One-hot encoded characters (vocab_size neurons)
- **Hidden Layer**: 32 neurons with tanh activation
- **Output Layer**: Softmax over vocabulary (vocab_size neurons)
- **Parameters**:
  - Wxh: hidden_size × vocab_size (input → hidden)
  - Whh: hidden_size × hidden_size (hidden → hidden)
  - Why: vocab_size × hidden_size (hidden → output)
  - bh: hidden_size bias terms
  - by: vocab_size bias terms

## Training Data

The network learns from the sequence "hello world!":
```
Training sequence: "hello world!"
Input sequence:    "hello world"
Target sequence:   "ello world!"
```

## Usage

```bash
ruby 02_rnn.rb
```

## Key Features

### Mathematical Operations
The implementation includes comprehensive vector/matrix operations:
- **One-hot encoding**: Convert character indices to sparse vectors
- **Softmax**: Probability distribution over vocabulary
- **Tanh activation**: Non-linear hidden state transformation
- **Matrix-vector multiplication**: Efficient linear transformations
- **Gradient clipping**: Prevents exploding gradients

### Backpropagation Through Time (BPTT)
The implementation manually calculates gradients through the entire sequence:

1. **Forward Pass**: Process entire sequence, storing all hidden states
2. **Backward Pass**: Propagate gradients backwards through time
3. **Gradient Clipping**: Limit gradient magnitudes to prevent instability
4. **Parameter Updates**: Update all weights using SGD

### Character Generation
Two generation modes are implemented:

1. **Next Character Prediction**: Given a prefix, predict the most likely next character
2. **Autoregressive Generation**: Generate sequences character by character

## Training Process

```ruby
# Forward pass through full sequence
hprev = zeros(hidden_size)
seq_len.times do |t|
  x_t = one_hot(inputs[t], vocab_size)
  h_t = tanh(vecadd(vecadd(matvec(wxh, x_t), matvec(whh, hprev)), bh))
  y_t = vecadd(matvec(why, h_t), by)
  p_t = softmax(y_t)
  loss += -Math.log(p_t[targets[t]] + 1e-12)
  hprev = h_t
end
```

## Learning Parameters

- **Hidden Size**: 32 neurons
- **Learning Rate**: 0.1
- **Epochs**: 400
- **Sequence Length**: 11 characters
- **Vocabulary Size**: 10 unique characters
- **Gradient Clipping**: ±5.0

## Weight Initialization

Uses Xavier/Glorot initialization:
- **Input weights**: `scale_x = sqrt(1/vocab_size)`
- **Hidden weights**: `scale_h = sqrt(1/hidden_size)`

## Key Concepts Demonstrated

### Sequence Modeling
- **Temporal Dependencies**: Hidden state carries information across time steps
- **Character-level Language Modeling**: Learning character transition probabilities
- **Autoregressive Generation**: Using previous predictions as future inputs

### Backpropagation Through Time
- **Unrolled Computation Graph**: Processing entire sequence in forward pass
- **Gradient Flow**: Propagating gradients backwards through time
- **Vanishing/Exploding Gradients**: Addressed with gradient clipping

### Vector Operations
- **One-hot Encoding**: Sparse representation of discrete inputs
- **Matrix Operations**: Efficient linear transformations
- **Activation Functions**: Tanh for hidden states, softmax for outputs

## Expected Output

After training, the network should be able to:
1. Predict "r" after seeing "hello wo"
2. Predict "l" after seeing "hello wor"
3. Generate reasonable character sequences

Example output:
```
Current: hello wo -> r??
Top5: r=0.456, d=0.234, l=0.123, o=0.098, !=0.089

Current: hello wor -> l??
Top5: l=0.567, d=0.234, r=0.123, o=0.076, !=0.000

Current: hello worl -> d??
Top5: d=0.678, l=0.234, r=0.088, o=0.000, !=0.000

Current: hello world -> !??
Top5: !=0.789, d=0.211, r=0.000, l=0.000, o=0.000
```

## Comparison with Previous Implementations

| Feature | 00_neural_nets | 01_mlp | 02_rnn |
|---------|----------------|--------|--------|
| Problem Type | Classification | Classification | Sequence Modeling |
| Architecture | Single layer | Multi-layer | Recurrent |
| Memory | None | None | Hidden state |
| Temporal Dependencies | No | No | Yes |
| Applications | Logic gates | XOR problem | Language modeling |

## Limitations

- **Vanishing Gradients**: Long sequences may suffer from gradient vanishing
- **Limited Context**: Hidden state size limits memory capacity
- **Simple Architecture**: No gating mechanisms like LSTM/GRU

This RNN implementation provides the foundation for understanding sequence modeling and serves as a stepping stone to more advanced architectures like LSTMs, GRUs, and Transformers.
