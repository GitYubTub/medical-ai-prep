## Section 3: Machine Learning Concepts

### Machine Learning
- Machine learning is a method where models learn patterns from data rather than being explicitly programmed.
- In medical imaging:
  - Input: image
  - Output: diagnosis, class, or mask

### Core Terms
- Features: pixel values or learned patterns
- Labels: ground truth (disease / no disease)
- Model: function mapping input to output

### Training Workflow
1. Split data:
   - Training set
   - Validation set
   - Test set
2. Forward pass to prediction
3. Loss calculation
4. Backpropagation
5. Update weights

### Common Problems
- Overfitting: memorizes training data
- Underfitting: too simple to learn patterns

### Metrics
- Accuracy (not always enough)
- Precision / Recall
- Sensitivity / Specificity
- Dice score (segmentation)

### Key Takeaway
In medicine, false negatives matter more than accuracy.
