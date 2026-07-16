import argparse
import time
import json
import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix
)


# =====================================================================
# Phase 3: Classifier Training
# =====================================================================
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
    t_load_start = time.time()

    # Load Dataset
    df = pd.read_csv(reducedTrain_csv)
    dataset_input_time = time.time() - t_load_start

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in the dataset.")

    y = df[target_column]
    X = df.drop(columns=[target_column])

    # Extract Dataset Statistics
    n_samples = len(df)
    n_features = X.shape[1]

    target_1_count = int(y.sum())
    target_1_percentage = (target_1_count / n_samples) * 100.0

    # Model Selection
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

    # Train the Model
    t_train_start = time.time()
    model.fit(X, y)
    training_time = time.time() - t_train_start

    # Ensure output directories exist
    os.makedirs(os.path.dirname(model_path) or '.', exist_ok=True)
    os.makedirs(os.path.dirname(metrics_json) or '.', exist_ok=True)

    # Save the Trained Model
    joblib.dump(model, model_path)

    # Generate and Save JSON Report
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

    with open(metrics_json, "w") as f:
        json.dump(report, f, indent=4)

    return report


# =====================================================================
# Phase 4: Test Set Prediction & Evaluation
# =====================================================================
def predict(
        reduced_Test_csv: str,
        target_column: str,
        model_path: str,
        predictions_csv: str,
        classif_stats_json: str,
):
    """
    Reads a test dataset, loads a trained classifier, performs inference,
    and calculates classification performance metrics.
    """
    # 1 & 2. Load the Model and Test Data
    model = joblib.load(model_path)
    df = pd.read_csv(reduced_Test_csv)

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in the test dataset.")

    y_true = df[target_column]
    X_test = df.drop(columns=[target_column])

    # 3. Generate Predictions
    y_pred = model.predict(X_test)

    # Get probabilities for the positive class (class 1)
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        # Fallback if a model doesn't support predict_proba
        y_prob = y_pred.astype(float)

    # 4. Save Per-Record Predictions CSV
    predictions_df = pd.DataFrame({
        "row_index": df.index,
        "true_target": y_true,
        "predicted_class": y_pred,
        "positive_probability": y_prob
    })

    os.makedirs(os.path.dirname(predictions_csv) or '.', exist_ok=True)
    predictions_df.to_csv(predictions_csv, index=False)

    # 5. Compute Classification Metrics
    n_samples = len(y_true)
    target_1_count = int(y_true.sum())
    target_1_percentage = (target_1_count / n_samples) * 100.0

    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1], zero_division=0)

    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        roc_auc = 0.0  # Handle edge case where only one class is present in y_true

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    # Extract model name for the report
    clf_name = type(model).__name__

    # Generate JSON Report
    stats_report = {
        "classifier": clf_name,
        "n_samples": n_samples,
        "target_1_count": target_1_count,
        "target_1_percentage": round(float(target_1_percentage), 2),
        "accuracy": float(acc),
        "class_0": {
            "precision": float(precision[0]),
            "recall": float(recall[0]),
            "f1": float(f1[0]),
            "support": int(support[0])
        },
        "class_1": {
            "precision": float(precision[1]),
            "recall": float(recall[1]),
            "f1": float(f1[1]),
            "support": int(support[1])
        },
        "roc_auc": float(roc_auc),
        "confusion_matrix": {
            "labels": [0, 1],
            "matrix": cm.tolist()
        }
    }

    os.makedirs(os.path.dirname(classif_stats_json) or '.', exist_ok=True)
    with open(classif_stats_json, "w") as f:
        json.dump(stats_report, f, indent=4)

    return stats_report


# =====================================================================
# Command Line Interface
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="Model training and inference manager.")
    subparsers = parser.add_subparsers(dest="command", required=True, help="Available sub-commands")

    # Subparser for 'train'
    train_parser = subparsers.add_parser("train", help="Train a classification model")
    train_parser.add_argument("--classifier", required=True, help="Classifier to use (e.g., random_forest).")
    train_parser.add_argument("--in-reduced", required=True, help="Path to the reduced training CSV dataset.")
    train_parser.add_argument("--target", required=True, help="Name of the target column.")
    train_parser.add_argument("--out-model", required=True, help="Path for the output trained model (.joblib).")
    train_parser.add_argument("--out-metrics", required=True, help="Path for the output metrics JSON file.")
    train_parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")

    # Subparser for 'predict'
    predict_parser = subparsers.add_parser("predict", help="Run inference and evaluate a trained model")
    predict_parser.add_argument("--in-reduced-test", required=True, help="Path to the reduced test CSV dataset.")
    predict_parser.add_argument("--target", required=True, help="Name of the target column.")
    predict_parser.add_argument("--in-model", required=True, help="Path to the trained model (.joblib) to load.")
    predict_parser.add_argument("--out-predictions", required=True, help="Path for the output predictions CSV file.")
    predict_parser.add_argument("--out-stats", required=True,
                                help="Path for the output classification stats JSON file.")

    args = parser.parse_args()

    if args.command == "train":
        print(f"[*] Starting model training: {args.classifier}")
        report = train(
            classifier=args.classifier,
            reducedTrain_csv=args.in_reduced,
            target_column=args.target,
            model_path=args.out_model,
            metrics_json=args.out_metrics,
            seed=args.seed
        )
        print("[+] Training completed successfully!")
        print(f"    - Output Model:   {args.out_model}")
        print(f"    - Output Metrics: {args.out_metrics}")

    elif args.command == "predict":
        print(f"[*] Starting model inference and evaluation")
        print(f"    - Model: {args.in_model}")
        print(f"    - Test Set: {args.in_reduced_test}")

        try:
            stats = predict(
                reduced_Test_csv=args.in_reduced_test,
                target_column=args.target,
                model_path=args.in_model,
                predictions_csv=args.out_predictions,
                classif_stats_json=args.out_stats
            )
            print("[+] Inference completed successfully!")
            print(f"    - Predictions CSV: {args.out_predictions}")
            print(f"    - Stats JSON:      {args.out_stats}")

            print("\n--- Evaluation Summary ---")
            print(f"Accuracy: {stats['accuracy']:.4f} | ROC-AUC: {stats['roc_auc']:.4f}")
            print(f"F1 (Class 0): {stats['class_0']['f1']:.4f} | F1 (Class 1): {stats['class_1']['f1']:.4f}")

        except Exception as e:
            print(f"[-] An error occurred during prediction: {e}")


if __name__ == "__main__":
    main()