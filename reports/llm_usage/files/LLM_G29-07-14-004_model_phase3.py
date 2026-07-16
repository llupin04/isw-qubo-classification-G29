import time
import json
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression


def train(
        classifier: str,
        reducedTrain_csv: str,
        target_column: str,
        model_path: str,
        metrics_json: str,
        seed: int = 42,
):
    """
    Reads a reduced training dataset, trains a specified binary classifier,
    saves the model via joblib, and outputs training statistics to a JSON file.
    """
    # ---------------------------------------------------------
    # 1. Load Dataset & Track Time
    # ---------------------------------------------------------
    t_load_start = time.time()

    df = pd.read_csv(reducedTrain_csv)

    dataset_input_time = time.time() - t_load_start

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in the dataset.")

    y = df[target_column]
    X = df.drop(columns=[target_column])

    # ---------------------------------------------------------
    # 2. Extract Dataset Statistics
    # ---------------------------------------------------------
    n_samples = len(df)
    n_features = X.shape[1]

    # Assuming binary target 0/1. Summing gives the count of 1s.
    target_1_count = y.sum()
    target_1_percentage = (target_1_count / n_samples) * 100.0

    # ---------------------------------------------------------
    # 3. Model Selection
    # ---------------------------------------------------------
    clf_name_clean = classifier.lower().strip()

    if clf_name_clean == "random_forest":
        model = RandomForestClassifier(random_state=seed, n_jobs=-1)
    elif clf_name_clean == "logistic_regression":
        model = LogisticRegression(random_state=seed, max_iter=1000)
    elif clf_name_clean == "gradient_boosting":
        model = GradientBoostingClassifier(random_state=seed)
    else:
        raise ValueError(
            f"Unsupported classifier '{classifier}'. "
            "Choose from: 'random_forest', 'logistic_regression', 'gradient_boosting'."
        )

    # ---------------------------------------------------------
    # 4. Train the Model
    # ---------------------------------------------------------
    t_train_start = time.time()

    model.fit(X, y)

    training_time = time.time() - t_train_start

    # ---------------------------------------------------------
    # 5. Save the Trained Model
    # ---------------------------------------------------------
    # Ensure directory structure exists if necessary
    import os
    os.makedirs(os.path.dirname(model_path) or '.', exist_ok=True)
    joblib.dump(model, model_path)

    # ---------------------------------------------------------
    # 6. Generate and Save JSON Report
    # ---------------------------------------------------------
    report = {
        "classifier": clf_name_clean,
        "seed": seed,
        "training_dataset": reducedTrain_csv,
        "target_column": target_column,
        "model_path": model_path,
        "n_samples": n_samples,
        "n_features": n_features,
        "target_1_percentage": round(float(target_1_percentage), 2),
        "dataset_input_time": round(float(dataset_input_time), 2),
        "training_time": round(float(training_time), 2)
    }

    os.makedirs(os.path.dirname(metrics_json) or '.', exist_ok=True)
    with open(metrics_json, "w") as f:
        json.dump(report, f, indent=4)

    return report