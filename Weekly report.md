# 2-Week Progress Report

**Student:** Yubo Jin  
**Project/Repo:** CRC Risk Modeling with ATSO  
**Reporting Period:** February 1 – February 14, 2026

---

## Overview

Over the past two weeks, I focused on implementing and understanding a full machine learning pipeline for CRC risk prediction using Random Forests and the ATSO optimization algorithm. Work included data preprocessing, model evaluation, hyperparameter optimization, and preparing tools for reproducible experiments.

---

## Tasks Completed

### **1. Data Preparation**
- Loaded and explored dataset, identified target `CRC_Risk` and features.
- Implemented `load_data(path)`:
  - Removed `Participant_ID` from features
  - One-hot encoded categorical variables
  - Combined numeric and categorical features into final matrix `X`
- Implemented `upsample_minority(X, y, rng)` to balance class distribution.

### **2. Model Evaluation**
- Created `compute_metrics(y_true, y_prob, threshold)`:
  - Accuracy, F1, ROC-AUC, sensitivity, specificity
- Designed custom weighted score for model optimization:

### **3. Hyperparameter Optimization (ATSO)**
- Implemented `decode_solution(vec)` to map candidate vector to RF hyperparameters.
- Built `objective(vec, X, y, cv, seed)` to evaluate candidates via 5-fold stratified CV:
- Upsampling training folds
- Training RF
- Computing metrics
- Implemented `atso_optimize(...)`:
- Population-based optimization
- Iterative updates with mutation and best/peer guidance
- Tracking global best solution
- Print CV scores per iteration

### **4. Experiment Pipeline**
- Created `run_experiment(...)`:
- Split train/test sets
- Run ATSO optimizer on training set
- Train final RF on best hyperparameters
- Evaluate on untouched test set
- Output final metrics and hyperparameters

### **5. Command-Line Interface**
- Added `parse_args()` + `if __name__ == "__main__"` block
- Supported arguments:
- Dataset path
- Population size
- Iterations (default = 30)

---

## Challenges / Notes
- Handling minority class imbalance required careful use of upsampling only on training folds to prevent data leakage.
- Mapping optimization vectors to valid hyperparameters needed scaling and rounding.
- Computing combined weighted metrics required integrating multiple evaluation functions.

---

## Next Steps
- Run ATSO optimization on the full training set and evaluate on final test set.
- Explore feature importance and interpretability for Random Forest predictions.
- Experiment with alternative weighting schemes in the scoring function.
- Document all code and ensure reproducibility in the GitHub repository.

---

**Summary:**  
In the last two weeks, I successfully built a complete ML pipeline from data loading to hyperparameter optimization, implemented evaluation metrics, and prepared CLI execution for reproducible experiments. Work is on track for next steps in model evaluation and interpretation.
