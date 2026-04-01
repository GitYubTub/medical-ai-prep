## Multiple linear regression
- Instead of a single feature (e.g., house size), multiple features like number of bedrooms, floors, and age of the house are used to predict the target variable (e.g., house price)
- Features are denoted as X_1, X_2, ..., X_n, where n is the number of features, and each training example has a vector of these features

### Model Representation
- The model predicts the output as a weighted sum of all features plus a bias term: $$f_{w,b}(X) = w_1X_1 + w_2X_2 + ... + w_nX_n + b$$
  - Note to self: weighted represents the importance of that feature
  - $\vec{w}$ = $$[w_1, w_2, w_3 ... w_n]$$
  - b is a number
  - $\vec{x}$ = $$[x_1, x_2, x_3 ... x_n]$$
- $$f_{\vec{w},b}(\vec{x}) = \vec{w} \cdot \vec{x} + b = w_1X_1 + w_2X_2 + ... + w_nX_n + b$$
