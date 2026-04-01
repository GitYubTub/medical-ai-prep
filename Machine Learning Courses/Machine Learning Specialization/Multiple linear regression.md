## Multiple linear regression
- Instead of a single feature (e.g., house size), multiple features like number of bedrooms, floors, and age of the house are used to predict the target variable (e.g., house price)
- Features are denoted as X_1, X_2, ..., X_n, where n is the number of features, and each training example has a vector of these features

### Model Representation
- The model predicts the output as a weighted sum of all features plus a bias term: $$f_{w,b}(X) = w_1X_1 + w_2X_2 + ... + w_nX_n + b$$
  - Note to self: weighted represents the importance of that feature
  - $\vec{w}$ = $$[w_1, w_2, w_3 ... w_n]$$
  - b is a number
  - $\vec{x}$ = $$[x_1, x_2, x_3 ... x_n]$$
 
### Vectorization
Without Vectorization
- $$f_{\vec{w},b}(\vec{x}) = \Sigma_{j=1}^{n} w_jx_j + b$$
  - In Python:
  ```python
  f = 0
  for j in range(n):
    f = f + w[j] * x[j]
  f = f + b
Vectorization
- $$f_{\vec{w},b}(\vec{x}) = \vec{w} \cdot \vec{x} + b = w_1X_1 + w_2X_2 + ... + w_nX_n + b$$
  - In Python:
  ```python
  f = np.dot(w, x) + b
- Vectorized code uses operations like the dot product to compute results in a single line
- It runs much faster by utilizing parallel hardware such as CPUs and GPUs, making it practical for large datasets

### Gradient descent
- $$w_j = w_j - \alpha \frac{d}{dw_j} J(\vec{w},b)$$
  - $$w_n = w_n - \alpha \frac{1}{m} \Sigma_{i=1}^{m} (f_{\vec{w},b}(\vec{x}^{(i)}) - y^{(i)})x_n^{(i)} $$
- $$b = b - \alpha \frac{d}{db_j} J(\vec{w},b)$$
  - $$b = b - \alpha \frac{1}{m} \Sigma_{i=1}^{m} (f_{\vec{w},b}(\vec{x}^{(i)}) - y^{(i)})

### Normal Equation Method
- An alternative to gradient descent for linear regression is the normal equation, which solves for w and b directly using linear algebra without iterations
- This method is not generalizable to other algorithms like logistic regression or neural networks and can be slow for large feature sets, but some libraries may use it internally


  
    

