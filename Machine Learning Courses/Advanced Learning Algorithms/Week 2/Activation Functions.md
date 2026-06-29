## Activation Functions

### Alternatives to the sigmoid activation
1. Sigmoid Function
   - Output Range: Between $0$ and $1$.
   - Used when the output layer in a binary classification problem (where you need to predict a probability between 0 and 1)
   - limitation only between 0 and 1, so limited for hidden layers
     - Q: Why is this a limitation to show the degree of something? Why can't 1 signify the highest degree and 0 signify the lowest degree?
     - A: The 1 caps the max at a certain value, but an input might give an output that exceeds the max value
       
2. ReLU (Rectified Linear Unit)
   - Output Range: $0$ to $\infty$
   - Able to increase the output's value without a cap
   - great for hidden layers
     
3. Linear Activation Function (no activation function)
   - Output Range: $-\infty$ to $\infty$
   - Best when predicting a continuous numerical value that can be positive or negative
