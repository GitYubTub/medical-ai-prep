## Neural network model

### Neural network layer
- plug values of the first vector of features into each neuron, which are just sigmoids or logistic regression models
- all the outputs of a layer are then put into a vector and plugged into the next layer
- Each layer can be represented by superscript i (<sup>[i]</sup>) where i is their layer number
<img width="507" height="245" alt="image" src="https://github.com/user-attachments/assets/2636646b-c25c-43ee-9faa-7d99c5105252" />

### More complex neural networks
- The sigmoid function of a neuron in one layer will use the output from the last layer as input, which, when written out, will have the superscript of the previous layer
- The neurons in each layer will have subscripts of numbers starting with the top neuron at one and ending at however many neurons there are
EX: Activation value of layer l, unit(neuron) j: $\text{a}_{j}^{[l]} = g(\vec{w} _{j}^{[l]} \cdot \vec{a} ^{[l-1]} + b _{j}^{[l]})$
