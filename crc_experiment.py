import argparse  # Lets us pass settings from the terminal, like --iterations 20
import numpy as np  # Numerical library used for arrays, random numbers, and math
import pandas as pd  # Data library used to read and process CSV files

from sentence_transformers import SentenceTransformer
from sklearn.ensemble import RandomForestClassifier  # The model we train
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score  # Evaluation metrics
from sklearn.model_selection import StratifiedKFold, train_test_split  # Data split helpers
from sklearn.preprocessing import OneHotEncoder  # Converts text categories into numeric columns


RANDOM_SEED = 42  # Fixed seed for reproducibility (same random behavior each run)


def load_data(path: str):

    text_col = "Clinical_Notes"
    if text_col in X_df.columns:
        model = SentenceTransformer("all-MiniLM-L6-v2")
        text_vals = X_df[text_col].fillna("").astype(str).tolist()
        X_text = np.asarray(model.encode(text_vals, show_progress_bar=False), dtype=float)
        X_df = X_df.drop(columns=[text_col])
    else:
        X_text = None

    if X_text is not None:
        X = np.hstack([X, X_text])


    # Read the dataset file from disk
    df = pd.read_csv(path)

    # This is the target column we want to predict (0/1)
    target_col = "CRC_Risk"
    # Convert target to integers and then to a NumPy array
    y = df[target_col].astype(int).values

    # Start list of columns we do NOT want as input features
    drop_cols = [target_col]
    # Participant_ID is just an identifier, so drop it if present
    if "Participant_ID" in df.columns:
        drop_cols.append("Participant_ID")

    # Keep only feature columns (all columns except target/id)
    X_df = df.drop(columns=drop_cols)

    # Find numeric columns (age, bmi, etc.)
    num_cols = X_df.select_dtypes(include=["number"]).columns.tolist()
    # Everything else is treated as categorical text
    cat_cols = [c for c in X_df.columns if c not in num_cols]

    # If we have categorical columns, encode them
    if cat_cols:
        try:
            # Newer scikit-learn uses sparse_output
            encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            # Older scikit-learn uses sparse
            encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
        # Transform text categories into one-hot numeric columns
        X_cat = encoder.fit_transform(X_df[cat_cols])
        # Convert numeric columns to NumPy array (or empty if none)
        X_num = X_df[num_cols].to_numpy(dtype=float) if num_cols else np.empty((len(df), 0))
        # Join numeric + encoded categorical side-by-side
        X = np.hstack([X_num, X_cat])
    else:
        # If there are no categorical columns, just convert directly
        X = X_df.to_numpy(dtype=float)

    # Return features and target
    return X, y


def upsample_minority(X, y, rng):
    # Ensure y is NumPy array for indexing
    y = np.asarray(y)
    # Get class labels and their counts
    classes, counts = np.unique(y, return_counts=True)
    # If not binary or already balanced, no resampling needed
    if len(classes) != 2 or counts[0] == counts[1]:
        return X, y

    # Find which class is minority/majority
    min_class = classes[np.argmin(counts)]
    maj_class = classes[np.argmax(counts)]

    # Get row indices for each class
    min_idx = np.where(y == min_class)[0]
    maj_idx = np.where(y == maj_class)[0]

    # Randomly duplicate minority rows until both classes have equal size
    extra = rng.choice(min_idx, size=len(maj_idx) - len(min_idx), replace=True)
    # Combine majority + original minority + duplicated minority
    idx = np.concatenate([maj_idx, min_idx, extra])
    # Shuffle final order so classes are mixed
    rng.shuffle(idx)
    # Return resampled arrays
    return X[idx], y[idx]


def compute_metrics(y_true, y_prob, threshold):
    # Convert probabilities to class labels using selected threshold
    y_pred = (y_prob >= threshold).astype(int)
    # Get confusion matrix cells: true negative, false positive, false negative, true positive
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Standard metrics
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc = roc_auc_score(y_true, y_prob)

    # Sensitivity = recall for positive class
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    # Specificity = recall for negative class
    specificity = tn / (tn + fp) if (tn + fp) else 0.0

    # Return all metrics together
    return {
        "accuracy": acc,
        "f1": f1,
        "roc_auc": roc,
        "sensitivity": sensitivity,
        "specificity": specificity,
    }


def decode_solution(vec):
    # Keep optimizer vector values in valid range [0, 1]
    vec = np.clip(vec, 0.0, 1.0)

    # Map vector element 0 to n_estimators range
    n_estimators = int(round(100 + vec[0] * 500))
    # Map vector element 1 to max_depth range
    max_depth = int(round(3 + vec[1] * 17))
    # Map vector element 2 to min_samples_leaf range
    min_samples_leaf = int(round(1 + vec[2] * 24))
    # Map vector element 3 to min_samples_split range
    min_samples_split = int(round(2 + vec[3] * 28))
    # Map vector element 4 to max_features fraction range
    max_features = 0.3 + vec[4] * 0.7
    # Map vector element 5 to class 1 weight range
    pos_weight = 1.0 + vec[5] * 5.0
    # Map vector element 6 to decision threshold range
    threshold = 0.2 + vec[6] * 0.6

    # Return a dictionary of usable model settings
    return {
        "n_estimators": max(50, n_estimators),
        "max_depth": max(2, max_depth),
        "min_samples_leaf": max(1, min_samples_leaf),
        "min_samples_split": max(2, min_samples_split),
        "max_features": float(np.clip(max_features, 0.1, 1.0)),
        "class_weight": {0: 1.0, 1: float(max(1.0, pos_weight))},
        "threshold": float(np.clip(threshold, 0.05, 0.95)),
    }


def objective(vec, X, y, cv, seed):
    # Convert vector into model hyperparameters
    params = decode_solution(vec)
    # Remove threshold from model params (threshold is for prediction, not fitting)
    threshold = params.pop("threshold")

    # Store one score per fold
    fold_scores = []

    # Loop through each cross-validation split
    for fold_id, (tr_idx, va_idx) in enumerate(cv.split(X, y)):
        # Create per-fold RNG so results are deterministic but fold-specific
        rng = np.random.default_rng(seed + fold_id)

        # Select training and validation data for this fold
        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_va, y_va = X[va_idx], y[va_idx]

        # Balance the training fold only (avoid leakage into validation fold)
        X_tr, y_tr = upsample_minority(X_tr, y_tr, rng)

        # Build RandomForest model with decoded hyperparameters
        model = RandomForestClassifier(
            **params,
            random_state=seed,
            n_jobs=-1,
        )
        # Fit model on this fold's training data
        model.fit(X_tr, y_tr)

        # Predict probabilities for class 1 on validation fold
        y_prob = model.predict_proba(X_va)[:, 1]
        # Compute validation metrics at selected threshold
        m = compute_metrics(y_va, y_prob, threshold)

        # Weighted objective score used by optimizer
        score = 0.60 * m["f1"] + 0.20 * m["sensitivity"] + 0.20 * m["specificity"]
        # Save this fold score
        fold_scores.append(score)

    # Return average score across folds
    return float(np.mean(fold_scores))


def atso_optimize(X, y, pop_size=20, iterations=40, seed=RANDOM_SEED):
    # Main random generator for optimization
    rng = np.random.default_rng(seed)
    # Number of decision variables in each solution vector
    dim = 7
    # Initialize population with random values in [0, 1]
    pop = rng.uniform(0.0, 1.0, size=(pop_size, dim))

    # Stratified K-fold preserves class ratio in each fold
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    # Evaluate initial population fitness
    fitness = np.array([objective(ind, X, y, cv, seed) for ind in pop])

    # Identify best initial solution
    best_idx = int(np.argmax(fitness))
    best = pop[best_idx].copy()
    best_score = float(fitness[best_idx])

    # Outer loop: number of ATSO iterations (this controls search length)
    for t in range(iterations):
        # Progress goes from 0 to 1 as iterations advance
        progress = t / max(1, iterations - 1)

        # Adaptive factor: high early (explore), low later (exploit)
        c1 = 2.0 * (1.0 - progress)
        # Mutation amount decreases over time
        mutation_sigma = 0.20 * (1.0 - progress) + 0.02

        # Update each individual in the population
        for i in range(pop_size):
            # Random vectors for stochastic movement
            r1 = rng.random(dim)
            r2 = rng.random(dim)

            # Movement coefficients
            a = 2.0 * c1 * r1 - c1
            b = 2.0 * r2

            # Pick a random peer from population
            peer = pop[rng.integers(0, pop_size)]

            # Candidate update: sometimes move relative to global best, sometimes peer
            if rng.random() < 0.5:
                candidate = best + a * np.abs(b * best - pop[i])
            else:
                candidate = peer + a * np.abs(b * peer - pop[i])

            # With some probability, add Gaussian mutation for diversity
            if rng.random() < 0.35:
                candidate = candidate + rng.normal(0.0, mutation_sigma, size=dim)

            # Keep candidate values within valid bounds
            candidate = np.clip(candidate, 0.0, 1.0)
            # Evaluate candidate quality
            cand_fit = objective(candidate, X, y, cv, seed)

            # Greedy acceptance: keep candidate only if it improves this individual
            if cand_fit > fitness[i]:
                pop[i] = candidate
                fitness[i] = cand_fit

                # Update global best if this candidate is best so far
                if cand_fit > best_score:
                    best_score = float(cand_fit)
                    best = candidate.copy()

        # Print progress each iteration
        print(f"Iteration {t + 1:02d}/{iterations} - Best CV score: {best_score:.4f}")

    # Return best solution found and its score
    return best, best_score


def run_experiment(data_path, test_size, pop_size, iterations, seed):
    # Load and preprocess dataset
    X, y = load_data(data_path)

    # Create train/test split with class balance preserved
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )

    # Run optimizer only on training data
    best_vec, best_cv_score = atso_optimize(
        X_train,
        y_train,
        pop_size=pop_size,
        iterations=iterations,
        seed=seed,
    )

    # Decode best vector into hyperparameters
    best_params = decode_solution(best_vec)
    # Pull out threshold separately (not a RandomForest constructor arg)
    threshold = best_params.pop("threshold")

    # Balance training set before final model fit
    rng = np.random.default_rng(seed)
    X_train_bal, y_train_bal = upsample_minority(X_train, y_train, rng)

    # Train final model using best parameters found by optimizer
    final_model = RandomForestClassifier(
        **best_params,
        random_state=seed,
        n_jobs=-1,
    )
    final_model.fit(X_train_bal, y_train_bal)

    # Predict probabilities on untouched test set
    y_prob = final_model.predict_proba(X_test)[:, 1]
    # Compute final test metrics
    metrics = compute_metrics(y_test, y_prob, threshold)

    # Print tuned hyperparameters and CV objective score
    print("\n=== ATSO-like Best Hyperparameters ===")
    print(decode_solution(best_vec))
    print(f"Best CV objective score: {best_cv_score:.4f}")

    # Print final test-set performance
    print("\n=== Test Results ===")
    print(f"Accuracy:    {metrics['accuracy']:.4f}")
    print(f"F1-Score:    {metrics['f1']:.4f}")
    print(f"ROC-AUC:     {metrics['roc_auc']:.4f}")
    print(f"Sensitivity: {metrics['sensitivity']:.4f}")
    print(f"Specificity: {metrics['specificity']:.4f}")


def parse_args():
    # Build argument parser for command-line usage
    parser = argparse.ArgumentParser(
        description="ATSO-like optimization for CRC risk prediction with Random Forest."
    )
    # CSV file path
    parser.add_argument("--dataset", default="crc_dataset1.csv", help="Path to CSV dataset")
    # Fraction used for test split
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split ratio")
    # Number of candidate solutions per iteration
    parser.add_argument("--pop-size", type=int, default=20, help="ATSO population size")
    # Number of optimization rounds; larger means longer search
    parser.add_argument("--iterations", type=int, default=100, help="ATSO iterations")
    # Seed for reproducible randomness
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed")
    # Return parsed args object
    return parser.parse_args()


if __name__ == "__main__":
    # Parse terminal arguments
    args = parse_args()
    # Run full experiment with provided arguments
    run_experiment(
        data_path=args.dataset,
        test_size=args.test_size,
        pop_size=args.pop_size,
        iterations=args.iterations,
        seed=args.seed,
    )
