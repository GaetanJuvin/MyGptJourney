# 04_optimize_tiny_transformer

An optimized Transformer implementation using Torch.rb (PyTorch bindings for Ruby) with automatic differentiation, advanced training features, and interactive chat functionality.

## Overview

This folder contains a production-ready Transformer implementation that leverages Torch.rb for efficient computation and automatic differentiation. It includes advanced training features like learning rate scheduling, early stopping, gradient clipping, and checkpointing, along with an interactive chat interface.

## Files

- `04_optimize_tiny_transformer.rb` - Optimized Transformer with Torch.rb and chat interface
- `Gemfile` - Ruby dependencies (torch-rb)

## Key Improvements over 03_tiny_transformer

| Feature | 03_tiny_transformer | 04_optimize_tiny_transformer |
|---------|-------------------|----------------------------|
| **Differentiation** | Manual backpropagation | Automatic differentiation (Torch.rb) |
| **Optimizer** | Custom Adam | Torch::Optim::Adam |
| **Training** | Basic SGD | Advanced features (scheduling, early stopping) |
| **Performance** | Pure Ruby (slow) | Torch.rb (optimized) |
| **Features** | Training only | Training + Interactive chat |
| **Stability** | Manual gradient clipping | Built-in stability features |

## Neural Network Architecture

### Core Components
- **Token Embeddings**: Learnable character representations
- **Positional Embeddings**: Learnable position encodings  
- **Self-Attention**: Single attention head with causal masking
- **Layer Normalization**: Stabilizes training
- **Residual Connections**: Improves gradient flow
- **Feed-Forward**: Linear projection to vocabulary

### Architecture Details
- **Model Dimension**: 64
- **Vocabulary Size**: Dynamic (based on training data)
- **Max Sequence Length**: 512
- **Attention Heads**: 1 (single-head)
- **Blocks**: 1 (single transformer block)

## Training Features

### Advanced Optimizer
```ruby
@optimizer = Torch::Optim::Adam.new(@model.parameters, lr: lr, weight_decay: weight_decay)
```

### Learning Rate Scheduling
- **Warmup**: Linear warmup for first 200 steps
- **Plateau Reduction**: Reduces LR when validation loss plateaus
- **Early Stopping**: Stops training if no improvement for 150 epochs

### Training Stability
- **Gradient Clipping**: Prevents exploding gradients (`clip_norm: 1.0`)
- **Weight Decay**: L2 regularization (configurable)
- **Validation Split**: 10% of data for validation monitoring

## Training Data

Structured conversational dialogs repeated 300 times:
```
<system>You are helpful.</system>
<user>hi</user>
<assistant>hello!</assistant>
<eot>

<system>You are helpful.</system>
<user>what is 2+2?</user>
<assistant>4</assistant>
<eot>

<system>You are helpful.</system>
<user>color of sky?</user>
<assistant>blue</assistant>
<eot>
```

## Usage

### Setup
```bash
cd 04_optimize_tiny_transformer
bundle install
```

### Training
```bash
bundle exec ruby 04_optimize_tiny_transformer.rb
```

### Interactive Chat
After training completes, the script automatically starts an interactive chat:
```
Chat ready. Type 'exit' to quit.
You> hi
Assistant> hello!
You> what is 2+2?
Assistant> 4
You> exit
```

## Learning Parameters

- **Learning Rate**: 0.0005 (with scheduling)
- **Epochs**: 400
- **Steps per Epoch**: 200
- **Block Size**: 256 (sequence length for training)
- **Batch Size**: 1 (sliding window training)
- **Weight Decay**: 0.0 (configurable)
- **Gradient Clipping**: 1.0
- **Warmup Steps**: 200
- **Plateau Patience**: 50 epochs
- **Early Stop Patience**: 150 epochs

## Key Features

### Automatic Differentiation
```ruby
class TinyTransformerTorch < Torch::NN::Module
  def forward(token_indices)
    embedded_sequence = embed_sequence(token_indices)
    # ... attention computation ...
    logits = residual_normalized.matmul(@W_out) + @b_out
    logits  # Torch.rb automatically tracks gradients
  end
end
```

### Checkpointing System
```ruby
module Checkpointing
  def self.save!(model, chars, path = CKPT_PATH)
    meta = {
      chars:   chars,
      d_model: model.d_model,
      max_len: model.max_len
    }
    File.binwrite(path, Marshal.dump({ meta: meta }))
  end
end
```

### Interactive Chat Interface
```ruby
module Chat
  def self.start(model, system_prompt, end_token, char_to_ix, ix_to_char, max_length = 200)
    # Interactive chat loop with error handling
  end
end
```

## Training Process

1. **Data Preparation**: Build vocabulary and tokenize training text
2. **Model Initialization**: Create TinyTransformerTorch with Xavier initialization
3. **Training Loop**:
   - Sliding window sampling
   - Forward pass with automatic differentiation
   - Loss computation (cross-entropy)
   - Backward pass (automatic gradients)
   - Gradient clipping
   - Learning rate scheduling
   - Parameter updates
4. **Validation**: Monitor validation loss and accuracy
5. **Early Stopping**: Stop if no improvement
6. **Checkpointing**: Save model metadata
7. **Chat Interface**: Start interactive conversation

## Expected Training Output

```
Training with Torch
Epoch 1, train_loss=2.3456 val_loss=2.1234 val_acc=0.234 (n=8159) lr=5.00e-04
Epoch 2, train_loss=2.1234 val_loss=2.0123 val_acc=0.345 (n=8159) lr=5.00e-04
...
Epoch 400, train_loss=0.0113 val_loss=0.0120 val_acc=0.995 (n=8159) lr=1.25e-04
[ckpt] saved to tiny_transformer.ckpt
Chat ready. Type 'exit' to quit.
```

## Performance Improvements

### Speed
- **Torch.rb Backend**: Leverages optimized PyTorch operations
- **Automatic Differentiation**: No manual gradient computation
- **GPU Support**: Automatic GPU acceleration if available

### Stability
- **Gradient Clipping**: Prevents training instability
- **Learning Rate Scheduling**: Adaptive learning rate management
- **Early Stopping**: Prevents overfitting
- **Weight Decay**: Regularization for better generalization

### Usability
- **Checkpointing**: Resume training from saved models
- **Interactive Chat**: Real-time conversation testing
- **Error Handling**: Robust error handling and recovery
- **Modular Design**: Clean separation of concerns

## Comparison with Previous Implementations

| Feature | 00_neural_nets | 01_mlp | 02_rnn | 03_tiny_transformer | 04_optimize_tiny_transformer |
|---------|----------------|--------|--------|-------------------|----------------------------|
| **Library** | Pure Ruby | Pure Ruby | Pure Ruby | Pure Ruby | Torch.rb |
| **Speed** | Fast | Fast | Slow | Slow | Fast |
| **Features** | Basic | Basic | Sequence | Attention | Production-ready |
| **Training** | Simple | Simple | BPTT | Manual AD | Advanced AD |
| **Interface** | None | None | Generation | Generation | Interactive Chat |

## Dependencies

- **torch-rb**: Ruby bindings for PyTorch
- **Ruby**: Version 2.7+ recommended

## File Structure

```
04_optimize_tiny_transformer/
├── 04_optimize_tiny_transformer.rb  # Main implementation
├── Gemfile                          # Dependencies
├── README.md                        # This file
└── tiny_transformer.ckpt           # Saved model (after training)
```

## Advanced Configuration

### Custom Training Parameters
```ruby
trainer = TorchWindowTrainer.new(
  model, tokens,
  lr: 0.0005,                    # Learning rate
  epochs: 400,                   # Training epochs
  steps_per_epoch: 200,          # Steps per epoch
  block_size: 256,               # Sequence length
  print_every: 1,                # Logging frequency
  clip_norm: 1.0,                # Gradient clipping
  weight_decay: 0.01,            # L2 regularization
  warmup_steps: 200,             # LR warmup steps
  plateau_patience: 50,          # Plateau detection
  early_stop_patience: 150       # Early stopping
)
```

### Model Architecture
```ruby
model = TinyTransformerTorch.new(
  vocab_size: vocab_size,        # Vocabulary size
  d_model: 64,                   # Model dimension
  max_len: 512,                  # Maximum sequence length
  seed: 7                        # Random seed
)
```

This optimized implementation represents the culmination of the neural network journey, combining the educational value of understanding Transformers with the practical benefits of modern deep learning frameworks and production-ready features.
