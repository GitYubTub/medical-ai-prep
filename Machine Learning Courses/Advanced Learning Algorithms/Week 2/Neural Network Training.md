## Neural Network Training

### TensorFlow implementation
<img width="528" height="364" alt="image" src="https://github.com/user-attachments/assets/9183baa0-6ed2-4d67-9935-c2edbde5e298" />

1. Import TensorFlow and its Functions
2. Use the TenserFlow function, Sequential, to sequentially string together the Dense layers
3. Compile the model and tell it to use the cross-entropy loss function (had a question about what compile does)
   - AI response: choosing the method to measure how well the model is doing and how it should improve.
5. Call the fit function, which fits the model to the loss function and all using the edataset X, Y

### Training Details
- loss function: measures how far off the network's prediction is from the answer
  - Q: In a classification example, is the loss calculated using the activation output values? A: Yes
- Cost function: measures how well the model is doing over several training examples using the loss function of each example
<img width="237" height="97" alt="image" src="https://github.com/user-attachments/assets/d12df12c-d29e-4520-9c94-c162931bdff9" />

Binary cross-entropy loss function
1. Create the network
2. Set up the loss function (Compile Function)
   - Q: Does the compiling of the loss function happen in this step, i.e., behind the scenes of the Compile function? A: In the fit function
3. Minimize the cost function using gradient descent (Fit Function)
   
