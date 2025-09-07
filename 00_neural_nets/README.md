# 00_neural_nets

A simple neural network implementation in Ruby that demonstrates the fundamentals of machine learning with backpropagation.

## Overview

This folder contains a basic neural network implementation that learns to solve the AND gate problem. The network uses:
- **Sigmoid activation function**
- **Manual backpropagation**
- **Gradient descent optimization**

## Files

- `00_neural_nets.rb` - Main neural network implementation

## Neural Network Architecture

- **Input Layer**: 2 neurons (for 2 inputs)
- **Hidden Layer**: None (single-layer perceptron)
- **Output Layer**: 1 neuron with sigmoid activation
- **Parameters**: 2 weights (w1, w2) and 1 bias (b)

## Training Data

The network learns to implement an AND gate:
```
Input 1 | Input 2 | Expected Output
--------|---------|---------------
   0    |    0    |       0
   0    |    1    |       0
   1    |    0    |       0
   1    |    1    |       1
```

## Usage

```bash
ruby 00_neural_nets.rb
```

## How It Works

1. **Initialization**: Random weights and bias are initialized
2. **Forward Pass**: Inputs are processed through the network
3. **Error Calculation**: Difference between predicted and actual output
4. **Backpropagation**: Weights and bias are updated using gradient descent
5. **Training**: Process repeats for 10,000 epochs
6. **Testing**: Final predictions are displayed

## Key Concepts Demonstrated

- **Activation Function**: Sigmoid function for non-linear transformation
- **Loss Function**: Mean squared error (implicit in error calculation)
- **Optimization**: Gradient descent with learning rate
- **Backpropagation**: Chain rule for weight updates

## Output

After training, the network should output:
```
0, 0 => 0
0, 1 => 0
1, 0 => 0
1, 1 => 1
```

## Learning Parameters

- **Learning Rate**: 0.1
- **Epochs**: 10,000
- **Initialization**: Random values between 0 and 1

This is a foundational example that demonstrates the core principles of neural networks before moving to more complex architectures like transformers.
