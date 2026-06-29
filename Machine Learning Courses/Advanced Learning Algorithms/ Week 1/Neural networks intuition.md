## Neural networks intuition

### Demand Prediction
- Understood that a logistic regression algorithm can represent each neuron
- Had questions about how each layer was a vector and how each layer connected to the others
  - All neurons from each layer are given access to all the neurons from the previous layer
  - The neurons in the previous layer act as features for each of the neurons in the current layer
  - It's then the neuron network's job to think of which features of the previous layer are most connected and relevant to each of the neurons in the current layer
    
- How will the neurons in each layer know which features (neurons) in the previous layer are the most relevant and which to ignore?
- So if each neuron can be seen as a logistic regression algorithm, we can assume that there are already existing data that each neuron is trained upon. Right? (Yes, stated later in the video)
  
- For now, think of each neuron as a logistic regression model with multiple features, linear regression above it's e 
  - $$f_{\vec{w},b}(\vec{x}) = g(\vec{w} \cdot \vec{x} + b) = \frac{1}{1+e^{-(\vec{w} \cdot \vec{x} + b)}}$$
    
- The point and ability of a neural network is that it can learn and engineer its features so that each layer creates better prediction features than the previous layer
  - The Nueron network, I'm assuming, adjusts the weight of each feature based on its relevance to the neurons in the current layer
  - It's stated in the video that a neural network can also decide the features of each layer after the input layer to give the best prediction
  - How can the neuron network decide which features to combine from the previous layer to output better features?
    - In the video used t-shirts as an example. How would the neuron network be able to decide that a better feature was to combine price and shipping cost to create affordability?
   
### Example: Recognizing Images
- Each neuron first looks for lines at different orientations from the image and uses the contrast in the image to detect these lines
- The second hidden layer then uses the features from the first hidden layer to zoom out and look for contrast in a bigger pixel window (human features on plain skin)
- The third hidden layer then uses the features from the second hidden layer to zoom out more and look for a whole face
- The neuron is able to do this using data and can completely train itself without a human to tell it what to look for
<img width="996" height="385" alt="image" src="https://github.com/user-attachments/assets/29bc97e6-bb8b-4397-a5b4-475427b495f1" />
