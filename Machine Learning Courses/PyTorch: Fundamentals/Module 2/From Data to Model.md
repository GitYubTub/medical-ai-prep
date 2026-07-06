## From Data to Model

### Overview of the ML Pipeline with PyTorch - Part 1: Data
- With huge data networks, the data needs to be loaded in batches, which are smaller and more manageable
- Neural networks train better when inputs are small numbers centered around 0
- Transforms: transforms your data into a format easy for machine learning models to learn and recognize
  - Code: <img width="427" height="105" alt="image" src="https://github.com/user-attachments/assets/c9c254bc-dd86-4ff8-8151-c04c1beca2a6" />
  - Compose: do the following things in order
  - ToTensor(): converts data into PyTorch tensors and scales them to fall between 0 and 1
  - Normalize((mean,), (std,)): centers the data in the tensor around 0 and scales the data using standard deviation
    
- Dataset:
  - Q: How does the dataset not preload everything? What is telling it to load one thing at a time?
  - A: because "dataset SomeDataset('./data', train=True, download=True, transform=transform)" doesn't load anything. It's a line of code that tells PyTorch where the data is stored and what data to access when asked
  - Q: How does the dataset know which is training data and which is testing data
  - A: You have to manually separate them into the two datasets, which will be learned in more detail in module 3
  <img width="814" height="60" alt="image" src="https://github.com/user-attachments/assets/c4a8220a-bae3-4bf0-9041-81f558de37f0" />
  
  - './data' indicates where the data is stored on your computer
  - train=True lets you decide to either load the train or testing data
  - download=True downloads the data if it's not there already
 
- Dataloader:
<img width="682" height="42" alt="image" src="https://github.com/user-attachments/assets/ecc8e5d3-32cb-491b-9b79-d52ca100f14b" />

  - Loads the data in batches with the batch_size
  - shuffle=True lets you train your dataset better by shuffling the training data 

### Overview of the ML Pipeline with PyTorch - Part 2: Models
- Making a model using nn.Module (similar to nn.Sequential)
  - _init_ defines layers
  - forward describes how the data flows through the layers
- There is no need to write forward out when trying to run forward "output = model(data)"
- "super()._init_()" creates a tracking system that tracks all the learnable parameters of the model (weights and biases)
- model.eval() sets your model into evaluation mode
<img width="700" height="250" alt="image" src="https://github.com/user-attachments/assets/94c466ba-3a77-47f2-a6b8-d580cd6418e0" />

### Loss
- "loss.backward()" back propagation determines which weights and biases contributed to the loss and need to be changed
- "optimizer.step()" updates the weights based on the diagnostic on the backward step
- Crossentropy loss: loss_function used for classification problems
  - punishes more confident wrong answers
- Mean Squared Error: a loss function that uses the squared difference between the real and predicted value to determin error
  - Squard, so it's always positive
  - punishes worse mistakes more than the smaller ones
  - best when predicting continuous values
- Loss Functions
<img width="747" height="243" alt="image" src="https://github.com/user-attachments/assets/5e624902-9fdd-4574-870b-57a0d3925787" />

    

  
