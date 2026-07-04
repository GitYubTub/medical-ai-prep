## Getting Started with PyTorch

### The Building Blocks of Neural Networks
- A neuron is just a linear equation that can be adjusted with weights and biases
  - The neuron adjusts the weights and biases with calculus(gradient descent)
- Inference is the neuron making predictions on data it has not seen before
  
### The ML Pipeline
<img width="569" height="140" alt="image" src="https://github.com/user-attachments/assets/6bf1dd2b-c3b0-40dc-b268-3f06bea26ffd" />

- Data ingestion: get the raw data
- Data prep: Clean, transform, and organize the data into something the model can use
- Model building: Choosing the model architecture
- Training: Use data to train the model
- Evaluation: How well the model does on unseen data

### Building a simple neural network
- ```python
  import torch
  import torch.nn as nn
  import torch.optim as optim
  ```
  - core functions: torch
  - components for building: nn
  - tools for training: optim
- Tensors: contain and store data in a way the model understands
- Batch is the entire list, and each inner bracket is a singular sample in that batch
  - a sample can contain multiple values, otherwise known as features
- dtype tells torch what kind of data is in each tensor
- Similar to TensorFlow, PyTorch uses Sequential to pass the data through the layers in order
<img width="468" height="83" alt="image" src="https://github.com/user-attachments/assets/354f22ee-c3e3-4a72-a029-871e43c54072" />

- Linear means the type of model, and the first 1 represents the input, the second 1 represents the output
<img width="542" height="93" alt="image" src="https://github.com/user-attachments/assets/76eba143-8f08-48b2-b68b-8f1e8d9b1557" />

- Loss function (MSELoss() is Mean Squared Error loss used for linear models): how wrong or how right the predictions are
- Optimizer (SDG is Stochastic Gradient Descent, which is the gradient descent model used): improve the weights and biases based on the error of the loss function with a learning rate included
<img width="507" height="258" alt="image" src="https://github.com/user-attachments/assets/bf423ed5-c910-4b4c-8a58-159026cbbc2c" />

- Runs 500 epochs or runs the training loop 500 times
- Optimizer.zero_grad: Clears the values from the previous training round; the adjustments data won't accumulate
  - Q: What does it mean that the adjustment data won't accumulate? In my mind, I'm thinking it means the adjustments to the model will be cleared after each run so that the model won't be adjusted with the same adjustments over and over again (accumulate)
  - A: (AI) When it says the adjustment data won't accumulate, it means that before each training step, the gradients (which represent adjustments to the model's parameters) are cleared.
- outputs = model(inputs): predicts the inputs using the model
- loss = loss_function(outputs, real data): calculates the loss based on the predictions compared to the actual data
- loss.backwards(): calculate adjustments using back propagation
- optimizer.step(): makes all of the adjustments
<img width="740" height="101" alt="image" src="https://github.com/user-attachments/assets/e30e518a-da1d-4be8-a3a9-e7126df08b9b" />

- torch.no_grad() tells PyTorch it's no longer training
- Tries to let the model predict data it has never seen before after its training

### Activation functions
- Q: How do more neurons capture more complexity?
  - A: (AI) When you have multiple neurons, each with its own weight and bias, they activate (or "turn on") at different points, creating multiple bends. 
- Q: Why is ReLU able to bend the curve even at non-zero points on the line?
  - A: (help from AI) Since the input for ReLU is the output of a neuron before activation, for z = w*x + b, if z is negative, the ReLU of that z would output zero, creating a bend. The weights and biases can be changed to make z negative using the equation x = -b/w, which is the equation if z were zero.
- Note: In PyTorch, we only write code for the layers that compute
  - Q: What if there were multiple hidden layers?
  - A: I originally thought that we don't write code for hidden layers, but instead we don't write code for the input layer

### Tensors
- .shape gives batch size(number of samples) first, then number of features per sample
  - In your model, if it takes only one input, you would get an error if there is more than one feature per sample
- .float() converts any tensors into 32bit float data types
  - now can mix floats and integers for outputs of floats
- can convert arrays from numpy to tensors using ".from_numpy(numpy array)"
- Quick data
  - zeros = torch.zeros(3, 3): 3x3 tensor of zeros
  - ones = torch.ones (2, 4): 2x4 tensor of ones
  - random = torch.rand (5, 5): 5x5 tensor with random values
- You always need a batch size for input data
   - can use ".unsqueeze()" to add dimensions
   - ".squeeze()" takes away dimensions
- Indexing and slicing in TensorFlow work the same way as in Python
  - .item() can be used to convert elements in tensors to a Python number
    - It can only be used on singular elements
  

  
