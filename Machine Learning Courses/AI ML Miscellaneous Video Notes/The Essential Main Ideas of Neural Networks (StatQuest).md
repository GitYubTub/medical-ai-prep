## The Essential Main Ideas of Neural Networks
- https://www.youtube.com/watch?v=CqOfi41LfDw
- https://www.youtube.com/watch?v=IN2XmBhILt4
- https://www.youtube.com/watch?v=68BZ5f7P94E

This will hopefully help you better visualize what a neural network is actually doing

## 3Blue1Brown
https://www.youtube.com/watch?v=aircAruvnKk
- A simple neural network starts with an input, its "activation," which is stored within a neuron.
- In a classification neural network, the last layer usually represents the percentage in decimals of the possibility of that output being the right classification.
- The task of identifying numbers can be broken down into a separate task for each layer, then built up from each layer to the text
  - For example, the first layer represents the pixel input, and the next signals a line segment that the pixels have lit up, and the next can be a whole bunch of segments together that make up a number
  - Layer 1: input; layer 2: little edges; layer 3: larger edges ot loops that build up numbers; layer 4: identification of the numbers
- Each neuron in one layer has a connection to each of the neurons in the next layer, and each of the connections has a weight attached that tells the network how much the neurons in the next layer matter based on the input of the previous layer
- The weights and bias change after each full epoch and is changed to better values that 
