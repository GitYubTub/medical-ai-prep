## Neural network implementation in Python

### Forward prop in a single layer
The previous lesson implemented a neural network using Tensorflow; this lesson walked through how to implement using numpy
- The weight and bias of each neuron are already known because of forward propagation
- define: Input(x), Weights(w), Bias(b), Linear regression function(z), Activation(a)
  - to signify superscripts and subscripts, use format ex: w2_1 $$w^2 _1$$
<img width="906" height="392" alt="image" src="https://github.com/user-attachments/assets/ff3a344c-f86e-4d63-984e-dad9876509c4" />
- I somehow didn't realize that the lab had both back prob and for prob (I did go back to look at it again)
- These lessons are just to show the behind-the-scenes code using numpy

### General implementation of forward propagation
- Define W as a numpy matrix
- Define b as a numpy array
- Define input as a numpy array
<img width="350" height="286" alt="image" src="https://github.com/user-attachments/assets/26a89476-e073-417d-8296-9fffa19b78ae" />

- Write the Dense function
- parameters are input, weight, and bias
  - Q: was a bit confused about the relation between the number of repeats and the number of features
  - A: Because we want every neuron in each layer to have experienced the input 
- Since arrays are rigid and have a predetermined number of elements inside, create a 3-element array of zeros
- for loop to run the sigmoid function 3 times and plug in the input to each of the neurons
- define each of the columns using W[:,j] as j represents the jth column (0,1,2) 
- define the sigmoid function and compute it using $$np.dot$$
- set the 3-element array of zeros equal to the output
- return
<img width="349" height="220" alt="image" src="https://github.com/user-attachments/assets/fbca6631-e05c-481e-81ec-c30f603fbfb9" />

Note: g is not shown but is the $$1/(1-e^z)$$ part

- Sequential function
  - the only parameter is the input
  - define all the activation variables using dense
    - Change the input, weight, and bias to the appropriate variables of each layer
    - In the video, the Weights and biases aren't shown, but pretty sure every single one needs to be defined
  - return the final activation
<img width="292" height="196" alt="image" src="https://github.com/user-attachments/assets/69328d8a-6105-40ce-8361-fea90a09ade1" />

