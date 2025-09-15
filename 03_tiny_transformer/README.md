# 03_tiny_transformer

A complete Transformer implementation in pure Ruby with manual backpropagation, demonstrating the core architecture that revolutionized natural language processing.

## Overview

This folder contains a simplified Transformer model that implements the essential components of the "Attention Is All You Need" architecture. The implementation includes manual gradient computation, Adam optimizer, and training on conversational dialog data.

## Files

- `03_tiny_transformer.rb` - Complete Transformer implementation with training pipeline

## Neural Network Architecture

### Core Components
- **Token Embeddings**: Learnable representations for each character
- **Positional Embeddings**: Learnable position encodings
- **Self-Attention**: Single attention head with Query, Key, Value matrices
- **Feed-Forward**: Linear transformation layer
- **Causal Masking**: Prevents attending to future tokens

### Architecture Details
- **Input Layer**: Token + positional embeddings
- **Attention Layer**: Single-head self-attention with causal mask
- **Output Layer**: Linear projection to vocabulary
- **Parameters**:
  - E: vocab_size × d_model (token embeddings)
  - P: max_len × d_model (positional embeddings)
  - W_Q, W_K, W_V, W_O: d_model × d_model (attention weights)
  - W_out: d_model × vocab_size (output projection)
  - b_out: vocab_size (output bias)

## Training Data

The model learns from structured conversational data:
```
System: <system>You are helpful.</system>
User: <user>hi</user>
Assistant: <assistant>hello!</assistant>
End: <eot>

System: <system>You are helpful.</system>
User: <user>what is 2+2?</user>
Assistant: <assistant>4</assistant>
End: <eot>

System: <system>You are helpful.</system>
User: <user>color of sky?</user>
Assistant: <assistant>blue</assistant>
End: <eot>
```

## Usage

```bash
ruby 03_tiny_transformer.rb
```

## Key Features

### Self-Attention Mechanism
```ruby
def forward_block(x)
  q = matmul(x, @W_Q)  # Query
  k = matmul(x, @W_K)  # Key
  v = matmul(x, @W_V)  # Value
  
  # Scaled dot-product attention
  scores = matmul(q, transpose(k)) / sqrt(d_model)
  causal_mask!(scores)  # Prevent future attention
  a = softmax_rows(scores)
  o = matmul(a, v)
  matmul(o, @W_O)
end
```

### Manual Backpropagation
The implementation manually computes gradients for all operations:
- **Cross-entropy loss** gradients
- **Softmax** Jacobian-vector products
- **Matrix multiplications** gradient flow
- **Embedding** gradient updates

### Adam Optimizer
Custom Adam implementation with:
- **Momentum**: First-order moment estimates
- **Adaptive Learning**: Second-order moment estimates
- **Bias Correction**: Corrects for initialization bias
- **Per-parameter State**: Maintains separate state for each parameter

## Learning Parameters

- **Model Dimension**: 64
- **Vocabulary Size**: Varies by training data
- **Max Sequence Length**: 512
- **Learning Rate**: 0.03
- **Epochs**: 3,000
- **Print Frequency**: Every 200 epochs
- **Optimizer**: Adam (β₁=0.9, β₂=0.999, ε=1e-8)

## Training Process

1. **Forward Pass**: 
   - Token + positional embeddings
   - Self-attention with causal mask
   - Output projection and softmax

2. **Loss Calculation**: Cross-entropy loss on next-token prediction

3. **Backpropagation**: Manual gradient computation through all operations

4. **Parameter Updates**: Adam optimizer updates all parameters

## Key Concepts Demonstrated

### Attention Mechanism
- **Query, Key, Value**: Core attention components
- **Scaled Dot-Product**: Attention score computation
- **Causal Masking**: Preventing future information leakage
- **Multi-head Capability**: Single head implementation (extensible)

### Positional Encoding
- **Learnable Embeddings**: Position-specific representations
- **Token + Position**: Combined input representation
- **Sequence Awareness**: Model understands token positions

### Gradient Flow
- **Manual Differentiation**: Understanding gradient computation
- **Chain Rule**: Backpropagation through complex operations
- **Gradient Clipping**: Preventing exploding gradients

## Expected Output

After training, the model should generate coherent responses:
```
Input: <system>You are helpful.</system><user>hi</user><assistant>
Output: hello!</assistant><eot>

Input: <system>You are helpful.</system><user>what is 2+2?</user><assistant>
Output: 4</assistant><eot>
```

## Comparison with Previous Implementations

| Feature | 00_neural_nets | 01_mlp | 02_rnn | 03_tiny_transformer |
|---------|----------------|--------|--------|-------------------|
| Architecture | Single layer | Multi-layer | Recurrent | Attention-based |
| Memory | None | None | Hidden state | Attention weights |
| Parallelization | Sequential | Sequential | Sequential | Parallel |
| Long Dependencies | No | No | Limited | Full sequence |
| Training Stability | Good | Good | Gradient issues | Stable |

## Advanced Features

### Mathematical Operations
- **Matrix Operations**: Efficient linear algebra implementations
- **Softmax with Numerical Stability**: Prevents overflow/underflow
- **Gradient Clipping**: Maintains training stability
- **Xavier Initialization**: Proper weight scaling

### Training Infrastructure
- **Modular Design**: Separate model, optimizer, and trainer classes
- **Checkpointing Ready**: Easy to add model saving/loading
- **Extensible**: Can add more attention heads, layers, etc.

## Limitations

- **Single Attention Head**: Limited representational capacity
- **No Layer Normalization**: May affect training stability
- **No Residual Connections**: Missing key Transformer components
- **Manual Implementation**: Slower than optimized libraries

## Next Steps

This implementation provides the foundation for:
- **Multi-head Attention**: Adding multiple attention heads
- **Layer Normalization**: Improving training stability
- **Residual Connections**: Better gradient flow
- **Multiple Layers**: Deeper Transformer architectures
- **Advanced Optimizers**: Learning rate scheduling, warmup

This tiny Transformer demonstrates the core concepts that make Transformers so powerful for sequence modeling and serves as an educational stepping stone to understanding modern language models.
