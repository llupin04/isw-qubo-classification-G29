import argparse
import time
import json
import os
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


def main():
    # Main parser
    parser = argparse.ArgumentParser(description="Model training and inference manager.")

    # Subparsers for commands like 'train', 'predict', etc.
    subparsers = parser.add_subparsers(dest="command", required=True, help="Available sub-commands")

    # 'train' command parser
    train_parser = subparsers.add_parser("train", help="Train a classification model")
    train_parser.add_argument("--classifier", required=True, help="Classifier to use (e.g., random_forest).")
    train_parser.add_argument("--in-reduced", required=True, help="Path to the reduced training CSV dataset.")
    train_parser.add_argument("--target", required=True, help="Name of the target column.")
    train_parser.add_argument("--out-model", required=True, help="Path for the output trained model (.joblib).")
    train_parser.add_argument("--out-metrics", required=True, help="Path for the output metrics JSON file.")
    train_parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42).")

    args = parser.parse_args()

    if args.command == "train":
        print(f"[*] Starting model training: {args.classifier}")
        print(f"    - Dataset: {args.in_reduced}")

        try:
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

            print("\n--- Training Summary ---")
            print(
                f"Algorithm: {report['classifier']} | Features: {report['n_features']} | Samples: {report['n_samples']}")
            print(f"Class '1' Balance: {report['target_1_percentage']}%")
            print(f"Train Time: {report['training_time']}s")

        except Exception as e:
            print(f"[-] An error occurred during training: {e}")


if __name__ == "__main__":
    main()