## Train the model with gradient descent
- Gradient descent is a systematic method to find the values of parameters (like w and b) that minimize the cost function J
- It applies not only to linear regression but also to more complex models with multiple parameters, including neural networks

### Outline:
- start with some w,b (set w = 0, b = 0)
- Keep changing w, b to reduce J(w,b)
- Until we settle at or near a minimum
  - Note: there can be more than one minimum
  - Square error cost function always ends up in a bowl or hammock shape

### How Gradient Descent Works
- Start with initial guesses for parameters, often zero
- Iteratively update parameters by taking small steps in the direction that reduces the cost function most steeply
- This process continues until the cost function reaches a minimum or near-minimum value

### Gradient descent algorithm
- w = w - $\alpha$ $\frac{d}{dw}$ J(w,b)
  - $\alpha$: Learning rate (size of the gradient descent procedure)
  - $\frac{d}{dw}$ J(w,b): Direction of the gradient descent procedure
- b = b - $\alpha$ $\frac{d}{db}$ J(w,b)
- Repeat until convergence (w and b don't change with each additional gradient descent procedure)
- Simultaneously update w and b:
  - tmp_w = w - $\alpha$ $\frac{d}{dw}$ J(w,b)
  - tmp_b = b - $\alpha$ $\frac{d}{db}$ J(w,b)
  - w = tmp_w
  - b = tmp_b
- w as example:
  - 
