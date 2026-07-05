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



