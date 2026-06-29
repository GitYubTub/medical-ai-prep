## Multiclass Classification

### Multiclass
- Multiclass classification involves predicting one label from multiple possible categories
- Algorithms can learn decision boundaries that separate the feature space into multiple class regions

### Softmax
- Logistic regression predicts the probability of two classes (0 or 1) using a sigmoid function on a linear combination of inputs
- Softmax regression extends this to multiple classes by computing scores (z values) for each class and converting them into probabilities using the softmax function <img width="606" height="490" alt="image" src="https://github.com/user-attachments/assets/f9415a72-b5ef-45f3-a698-2eb5aa74c70a" />
- The loss for a training example with true class j is the negative log of the predicted probability a_j
- This loss encourages the model to assign high probability to the correct class
- The overall cost is the average loss over all training examples
- When there are only two classes, softmax regression reduces to logistic regression
