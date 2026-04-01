## Gradient descent in practice

### Feature scaling
- Features with large value ranges (e.g., house size in square feet) tend to have smaller associated parameter values ($\vec{w}$)
- Features with small value ranges (e.g., number of bedrooms) tend to have larger associated parameter values ($\vec{w}$)
- Rescaling features to have comparable ranges (e.g., both from 0 to 1) transforms the cost function contours into more circular shapes
- This allows gradient descent to converge faster and more efficiently
