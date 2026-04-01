## Gradient descent in practice
<img width="373" height="211" alt="image" src="https://github.com/user-attachments/assets/74062dbf-975a-4be8-a356-8ad760e90813" />

### Feature scaling
- Features with large value ranges (e.g., house size in square feet) tend to have smaller associated parameter values ($\vec{w}$)
- Features with small value ranges (e.g., number of bedrooms) tend to have larger associated parameter values ($\vec{w}$)
- Rescaling features to have comparable ranges (e.g., both from 0 to 1) transforms the cost function contours into more circular shapes
- This allows gradient descent to converge faster and more efficiently

### Feature Scaling Methods
- Scaling by maximum value: Divide each feature value by its maximum to bring the range between 0 and 1
- Mean normalization: Subtract the mean of the feature and divide by the range (max - min) to center values around zero, typically between -1 and 1
- Z-Score Normalization
  - Calculate the mean and standard deviation of each feature
  - Normalize by subtracting the mean and dividing by the standard deviation, resulting in values that reflect how many standard deviations they are from the mean
 
- Aim for feature values roughly between -1 and 1, but some variation is acceptable
- Features with very large or very small ranges benefit most from scaling
- Feature scaling generally helps gradient descent converge faster and is recommended when in doubt

### Checking gradient descent for convergence
- If gradient descent is working well, the cost J should go down after every iteration
- If the cost J ever goes up, it might mean the learning rate (Alpha) is too high or there is a bug
- When the cost J stops decreasing much and the curve flattens, gradient descent has likely converged
- Automatic convergence test:  use a small threshold (epsilon) to automatically decide when the decrease in cost is small enough to stop
