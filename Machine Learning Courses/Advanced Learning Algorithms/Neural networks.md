## Neural networks intuition

### Demand Prediction
- Understood that a logistic regression algorithm can represent each neuron
- Had questions about how each layer was a vector and how each layer connected to the others
  - All neurons from each layer are given access to all the neurons from the previous layer
  - The neurons in the previous layer act as features for each of the neurons in the current layer
  - It's then the neuron network's job to think of which features of the previous layer are most connected and relevant to each of the neurons in the current layer
- For now, think of each neuron as a logistic regression model with multiple features, linear regression above it's e 
  - $$f_{\vec{w},b}(\vec{x}) = g(\vec{w} \cdot \vec{x} + b) = \frac{1}{1+e^{-(\vec{w} \cdot \vec{x} + b)}}$$
- The point and ability of a neural network is that it can learn and engineer its features so that each layer creates better prediction features than the previous layer
  - The Nueron network, I'm assuming, adjusts the weight of each feature based on its relevance to the neurons in the current layer
  - It's stated in the video that a neural network can also decide the features of each layer after the input layer to give the best prediction
  - How can the neuron network decide which features to combine from the previous layer to output better features?
    - In the video used t-shirts as an example. How would the neuron network be able to decide a better feature was to combine price and shipping cost to create affordability?
