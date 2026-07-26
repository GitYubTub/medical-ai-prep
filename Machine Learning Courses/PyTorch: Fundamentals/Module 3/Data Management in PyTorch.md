## Data Management in PyTorch

### Data Access
- Download function:
  <img width="871" height="191" alt="image" src="https://github.com/user-attachments/assets/e93131a9-4c5f-4dd5-8514-0b643777c74b" />

  - dowloads the imageset and their labels from the URLs
  - creates a new file named "flower_data" in the current directory and that it's ok if the file named the samething already exists in another place
  - Gets images and their labels separately and need to be connected
- Pytorch dataset class:
  
  <img width="470" height="209" alt="image" src="https://github.com/user-attachments/assets/d5043f55-31c3-4b72-a63a-95111a090c47" />
  
  - def __init__(self): where to find the images and labels
    <img width="833" height="207" alt="image" src="https://github.com/user-attachments/assets/2fe92a32-835a-4bc1-973f-0d6b257ed116" />
    - 
  - def __len__(self): how many total samples
  - def __getitem__(self, idx): how to get image and label number "idx"
