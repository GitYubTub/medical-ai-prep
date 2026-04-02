## Classification with logistic regression

### Classification
- Classification predicts categories rather than continuous values, unlike linear regression
- Binary classification involves two classes, often labeled as 0 or 1, false or true, negative or positive
- Linear regression
  - Linear regression outputs continuous values, which is not ideal for classification tasks
  - Using a threshold (e.g., 0.5) to convert linear regression outputs to classes can lead to poor decision boundaries, especially when new data points are added
    
### Logistic Regression 
- Logistic regression is used for binary classification, such as determining if a tumor is malignant (1) or benign (0)
- Unlike linear regression, logistic regression fits an S-shaped curve (Sigmoid function) to predict probabilities between 0 and 1
- **Sigmoid Function:**
  - The Sigmoid function, g(z) = 1 / (1 + e^(-z)), maps any real-valued number into the range (0,1)
  - It outputs values close to 0 for large negative inputs, close to 1 for large positive inputs, and 0.5 when the input is zero
  - $$z = \vec{w} \cdot \vec{x} + b$$ -> $$g(z) = \frac{1}{1+e^{-z}}$$
  - $$f_{\vec{w},b}(\vec{x}) = g(\vec{w} \cdot \vec{x} + b) = \frac{1}{1+e^{-(\vec{w} \cdot \vec{x} + b)}}$$
  - Also can been written as $$P(y = 1|x;\vec{w},b)$$
- The output probability can be interpreted as the likelihood of the positive class; for example, a 0.7 output means a 70% chance the tumor is malignant
- Since y is binary, the probability of y=0 is 1 minus the probability of y=1
- The model parameters w and b influence the probability calculation.
- Predictions are made by setting a threshold (commonly 0.5); if f(x) ≥ 0.5, predict y = 1, otherwise y = 0
  - If $$f_{\vec{w},b}(\vec{x}) \ge 0.5$$
    - Yes: $$\hat{y} = 1$$
    - No: $$\hat{y} = 0$$
  - When is $$f_{\vec{w},b}(\vec{x}) \ge 0.5$$?
    - g(z) $\ge$ 0.5
    - z $\ge$ 0
    - $$\vec{w} \cdot \vec{x} + b \ge 0$$: $$\hat{y} = 1$$
    - $$\vec{w} \cdot \vec{x} + b < 0$$: $$\hat{y} = 0$$
- **Decision Boundary**
  - The decision boundary is defined by the equation w·x + b = 0, where the model is neutral between classes
  - For two features, this boundary is a line (e.g., x1 + x2 = 3) separating predicted classes
- **Complex Decision Boundaries**
  - Using polynomial features (e.g., x1², x2²) allows logistic regression to create nonlinear decision boundaries like circles or ellipses
  - Higher-order polynomial terms enable even more complex boundaries, allowing the model to fit intricate data patterns


  
