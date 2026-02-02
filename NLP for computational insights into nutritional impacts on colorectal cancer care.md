# Article 2: NLP for Computational Insights into Nutritional Impacts on Colorectal Cancer Care  

## 1. Data Collection and Structure

The study uses a population-level colorectal cancer (CRC) dataset with approximately 1,000 participants.  
Each participant record contains two main types of data:
- Structured data
  - Demographic variables (age, BMI)
  - Nutritional variables (macronutrient intake such as carbohydrates, fats, proteins). Only uses this + Lifestyle.
- Unstructured data
  - Free-text dietary descriptions
  - Lifestyle-related survey responses

## 2. NLP Preprocessing of Unstructured Dietary Text
The unstructured dietary text is processed using a standard NLP preprocessing pipeline:
- Convert all text to lowercase
- Perform stop-word removal to eliminate non-informative words
- Remove punctuation and noise
- Standardize text formatting
The authors also use word-frequency–based visualization (word clouds) to:
- Explore dominant dietary patterns
- Validate that preprocessing preserves meaningful nutritional information

## 3. Feature Extraction Using Large Language Models (LLMs)
The LLM used in the article uses a decoder-only backbone, as the article states that the LLM generates samples word by word using an autoregressive approach.
- A large language model processes each participant’s dietary text
- The LLM captures higher-level semantic information, such as:
  - Dietary quality
  - Eating patterns
  - Restrictions or food categories
These outputs are converted into structured, numerical features that can be used by downstream machine-learning models.

## 4. Feature Fusion
The study combines:
- LLM-derived dietary features
- Structured numeric lifestyle and nutrition variables
These are concatenated into a single feature vector per participant, allowing the model to learn interactions between:
- Quantitative values (protein intake)
- Qualitative dietary descriptions (“high-fiber diet,” “processed foods”)

## 5. Modeling and Optimization Framework
The combined feature vectors are used within a predictive framework that includes:
- Adaptive Tunicate Swarm Optimization (ATSO)  
  - Used to tune model parameters
  - Helps optimize performance in high-dimensional feature space
- LLM-enhanced classifier
  - Leverages both structured data and semantic text features
ATSO is particularly useful given the small dataset size and heterogeneous features.

## 6. Handling Class Imbalance
CRC-positive cases are significantly fewer than CRC-negative cases.  
To address this, the paper applies:
- Oversampling techniques to balance the training data
- Comparisons against SMOTE-based baselines

## 7. Baseline Model Comparisons
The proposed framework is compared against several baseline approaches:
- KNN + SMOTE
- Generalized Linear Models (GLM)
- Neural Networks (NN)

## 8. Evaluation Metrics
The study emphasizes medical-appropriate evaluation metrics, including:
- Confusion matrix
- Sensitivity (recall) – critical for detecting CRC cases
- Specificity – ability to correctly identify non-CRC cases
- F1-score – balances precision and recall under class imbalance

## 9. Replication Plan (Next Steps)

To replicate the methodology step by step, the following plan will be used:

1. Recreate the **NLP preprocessing pipeline** on dietary text
2. Implement a simplified version of **diet-text → structured feature extraction**
3. Fuse text-derived features with numeric lifestyle variables
4. Compare **no resampling vs oversampling vs SMOTE**
5. Use **k-fold cross-validation** to evaluate robustness
6. Report results using:
   - Confusion matrices
   - Sensitivity
   - Specificity
   - F1-score

## Key Machine Learning Concepts Involved
- Structured vs unstructured data  
- Feature engineering  
- Class imbalance  
- Oversampling and SMOTE  
- Overfitting  
- Cross-validation  
- Sensitivity, specificity, precision, recall, F1-score  


## Goal

The ultimate goal is to understand **how NLP-derived dietary features and structured clinical data can be combined** to improve CRC risk prediction, while carefully managing imbalance, overfitting, and robustness.

