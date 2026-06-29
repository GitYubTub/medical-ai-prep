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

### Choosing activation functions
- Output layer
  - Sigmoid: Binary Classification (0 or 1)
  - Linear Regression: Any real number (Positive or Negative)
  - ReLU: Non-negative numbers only
<img width="937" height="273" alt="image" src="https://github.com/user-attachments/assets/ed67423d-2a09-4665-a486-7e104634d52c" />

- Hidden layer
  - ReLU is better than Sigmoid because ReLU only requires a simple maximum operation, but Sigmoid requires expensive operations which takes exptra processing time
  - ReLU is only flat in one place as opposed to Sigmoid's one. Allows for better, faster gradient desent
