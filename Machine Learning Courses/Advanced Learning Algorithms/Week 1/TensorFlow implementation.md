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

### Data in TensorFlow
- Note: There are some inconsistencies between how data is ​represented in NumPy and in TensorFlow
  - The difference between the two Python libraries is in how they store data
    - Numpy stores data in one-dimensional arrays (Edit: correct, but the main point here is that Tensorflow ONLY stores data in matrices)
    - Tensorflow stores data in matrices
  - Confused about how the two are different since numpy can also store data in matrices by using double brackets ([[]]), and it's the same format that TensorFlow uses
- To make matrices in numpy, use double brackets ([[]]), and to make multiple-row matrices, make each row its own list with brackets on either side
<img width="500" height="275" alt="image" src="https://github.com/user-attachments/assets/ac27881e-6d58-414e-b3e3-2fc81e2ad46c" />

- To change the output format back to numpy, use $variable.numpy()$
- The only thing that changes for the output is the beginning, from tf.Tensor to array, and TensorFlow includes the dimensions of the matrix
- I don't quite understand why this is such a big deal
- (Asked AI) Tensor: is TensorFlow's word for Matrix

### Building a neural network
- First, build the neuron layers by using Tensorflow's $$Dense$$ 
- To make the model, put the layers you've made as an array into the $$Sequential$$ function
  - Tells Tensorflow to create a neural network by sequentially stringing together the layers you've created
<img width="521" height="115" alt="image" src="https://github.com/user-attachments/assets/4c649a0f-a491-4aa6-b1aa-d11d65885b85" />

  - Note: conventionally, you build the model using the $$Sequential$$ function first and then build the neuron layers by using Tensorflow's $$Dense$$ directly into the model
 <img width="472" height="110" alt="image" src="https://github.com/user-attachments/assets/18a48482-fccd-4620-84b7-bbab1a2a26a2" />

    
- Then, to train the network, use $$model.compile(...)$$ and $$model.fit(x,y)$$ with x and y being the example data
  - .compile will be talked more about in later videos, as what is said in the video
- To output use model.predict(x_new), which will output the value of the output layer
  - Though not specifically stated in the video, x_new, based on my knowledge, the testing matrix for the neural network model

