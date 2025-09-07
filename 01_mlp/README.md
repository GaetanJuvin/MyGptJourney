# 01_mlp

A Multi-Layer Perceptron (MLP) implementation in Ruby that demonstrates deep learning fundamentals with manual backpropagation.

## Overview

This folder contains a neural network with one hidden layer that learns to solve the XOR problem - a classic example that demonstrates why neural networks need hidden layers (single-layer perceptrons cannot solve XOR).

## Files

- `01_mlp.rb` - Multi-layer perceptron implementation with manual backpropagation

## Neural Network Architecture

- **Input Layer**: 2 neurons (for 2 inputs)
- **Hidden Layer**: 4 neurons with sigmoid activation
- **Output Layer**: 1 neuron with sigmoid activation
- **Parameters**: 
  - W1: 4×2 weight matrix (input → hidden)
  - B1: 4 bias terms (hidden layer)
  - W2: 1×4 weight matrix (hidden → output)
  - B2: 1 bias term (output layer)

## Training Data

The network learns to implement an XOR gate:
```
Input 1 | Input 2 | Expected Output
--------|---------|---------------
   0    |    0    |       0
   0    |    1    |       1
   1    |    0    |       1
   1    |    1    |       0
```

## Usage

```bash
ruby 01_mlp.rb
```

## Key Features

### Weight Initialization
- **Xavier/Glorot Initialization**: Weights are scaled by `1/sqrt(fan_in)` to prevent vanishing/exploding gradients
- **Hidden layer**: `scale_h = sqrt(1/input_size)`
- **Output layer**: `scale_o = sqrt(1/hidden_size)`

### Activation Functions
- **Sigmoid**: `1/(1 + e^(-x))` for non-linear transformation
- **Sigmoid Derivative**: `y * (1 - y)` for backpropagation

### Training Process
1. **Forward Pass**: Data flows through input → hidden → output
2. **Loss Calculation**: Mean squared error `0.5 * (target - output)²`
3. **Backpropagation**: Gradients calculated using chain rule
4. **Weight Updates**: Gradient descent with learning rate

## Learning Parameters

- **Learning Rate**: 0.8
- **Hidden Layer Size**: 4 neurons
- **Epochs**: 5,000
- **Print Frequency**: Every 500 epochs

## Backpropagation Implementation

The implementation manually calculates gradients for each layer:

1. **Output Layer Gradients**:
   ```ruby
   dy = (y[0] - t) * dsigmoid(y[0])
   dw2 = h.map { |hj| dy * hj }
   db2 = dy
   ```

2. **Hidden Layer Gradients**:
   ```ruby
   dh = @w2[0].map { |w2j| w2j * dy }
   dz1 = dh.map.with_index { |val, j| val * dsigmoid(h[j]) }
   dw1 = dz1.map { |dz| [dz * x[0], dz * x[1]] }
   db1 = dz1
   ```

## Expected Output

After training, the network should output:
```
[0, 0] => 0 (raw: 0.0xxx)
[0, 1] => 1 (raw: 0.9xxx)
[1, 0] => 1 (raw: 0.9xxx)
[1, 1] => 0 (raw: 0.0xxx)
```

## Key Concepts Demonstrated

- **Multi-layer Architecture**: Why hidden layers are necessary
- **XOR Problem**: Classic example requiring non-linear separation
- **Manual Backpropagation**: Understanding gradient flow through layers
- **Weight Initialization**: Proper scaling for training stability
- **Batch Training**: Processing multiple examples per epoch
- **Loss Monitoring**: Tracking training progress

## Comparison with 00_neural_nets

| Feature | 00_neural_nets | 01_mlp |
|---------|----------------|--------|
| Layers | Single | Multi-layer (2 hidden) |
| Problem | AND gate | XOR gate |
| Activation | Sigmoid | Sigmoid |
| Initialization | Random | Xavier/Glorot |
| Architecture | Linear separable | Non-linear separable |

This implementation bridges the gap between basic neural networks and more complex architectures, demonstrating the power of hidden layers in solving non-linearly separable problems.
