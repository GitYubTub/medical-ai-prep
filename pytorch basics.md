## Section 4: PyTorch Basics

### PyTorch
- Deep learning framework
- Dynamic computation graphs
- GPU acceleration

### Core Objects
- Tensor: PyTorch version of NumPy array
- torch.tensor()
- .to(device) → CPU / GPU

### Autograd
- Automatically computes gradients
- Used for backpropagation
- Enabled by default

### Model Building Blocks
- torch.nn.Module
- Layers:
  - nn.Linear
  - nn.Conv2d
  - nn.ReLU
- Loss functions:
  - CrossEntropyLoss
  - BCEWithLogitsLoss
- Optimizers:
  - Adam
  - SGD

### Training Loop (Simplified)
```python
for epoch:
    forward pass
    compute loss
    backward pass
    optimizer step
```
### Key Takeaway
PyTorch turns math and gradients into learnable systems.
