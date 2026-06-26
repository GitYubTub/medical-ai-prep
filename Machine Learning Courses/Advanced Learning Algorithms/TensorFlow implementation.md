## TensorFlow implementation

### Inference in Code
<img width="513" height="76" alt="image" src="https://github.com/user-attachments/assets/9a1a3cc2-4670-45f6-8905-9c483a9d417c" />

- Didn't quite understand what Dense was and how layer_1 was a function
  - (Asked AI) Dense: It's just the type of layer that I'm currently using in the course
    - The Dense layer means that every individual sigmoid function (neuron) in each layer takes every single output from the previous layer
    - it's created as a function object that contains two arguments: (units = # of neurons, activation = type of function used)
  - x is defined as a numpy array that contains the features for the input layer
  - The array x is then sent into layer one by plugging x as the input to layer 1
- Through forward propagation, the output from the output layer is then used to predict something by checking whether yhat is bigger than or less than 0.5
- 
