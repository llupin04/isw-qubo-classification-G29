import argparse
import time
import json
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def fit_normalize(
        input_csv: str,
        target_column: str,
        normalized_csv: str,
        outInitalRes_json: str,
        minPercValid: float = 0.05,
):
    """
    Reads a numeric CSV dataset, drops sparse features, normalizes the remaining
    features, and exports the result alongside a processing statistics report.
    """
    t_load_start = time.time()

    # Read the dataset
    df = pd.read_csv(input_csv)

    load_time = time.time() - t_load_start
    t_process_start = time.time()

    total_rows = len(df)

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in the dataset.")

    y = df[target_column]
    X = df.drop(columns=[target_column])

    original_feature_cols = X.columns.tolist()
    num_input_features = len(original_feature_cols)

    # Filter "Empty or Nearly Empty" Columns
    is_valid = X.notna() & (X != 0)
    valid_percentages = is_valid.sum() / total_rows

    kept_features = valid_percentages[valid_percentages >= minPercValid].index.tolist()
    dropped_features = valid_percentages[valid_percentages < minPercValid].index.tolist()

    X_filtered = X[kept_features].copy()

    # Fill NaNs with the mean of their respective columns
    X_filtered = X_filtered.fillna(X_filtered.mean())

    # Identify and drop zero-variance (constant) columns
    stds = X_filtered.std()
    zero_variance_cols = stds[stds == 0].index.tolist()

    if zero_variance_cols:
        X_filtered = X_filtered.drop(columns=zero_variance_cols)
        # Update our tracking lists for the JSON report
        kept_features = [f for f in kept_features if f not in zero_variance_cols]
        dropped_features.extend(zero_variance_cols)

    # Normalize Remaining Feature Columns
    scaler = StandardScaler()
    X_scaled_array = scaler.fit_transform(X_filtered)
    X_scaled = pd.DataFrame(X_scaled_array, columns=kept_features, index=X_filtered.index)

    # Recombine
    final_df = pd.concat([X_scaled, y], axis=1)

    # Ensure output directories exist before saving
    os.makedirs(os.path.dirname(normalized_csv) or '.', exist_ok=True)
    os.makedirs(os.path.dirname(outInitalRes_json) or '.', exist_ok=True)

    # Write to CSV
    final_df.to_csv(normalized_csv, index=False)

    process_time = time.time() - t_process_start

    # Generate and Save JSON Report
    report = {
        "n_input_features": num_input_features,
        "n_kept_features": len(kept_features),
        "dataset_size": total_rows,
        "dataset_input_time": round(load_time, 4),
        "dataset_preprocessing_time": round(process_time, 4),
        "dropped_feature_names": dropped_features,
        "num_dropped_features": len(dropped_features)
    }

    with open(outInitalRes_json, "w") as f:
        json.dump(report, f, indent=4)

    return report


def main():
    parser = argparse.ArgumentParser(description="Preprocess loan dataset for binary classification.")

    # Define expected CLI arguments
    parser.add_argument("--input", required=True, help="Path to the input CSV dataset file.")
    parser.add_argument("--target", required=True, help="Name of the target column.")
    parser.add_argument("--out-data", required=True, help="Path for the output normalized CSV file.")
    parser.add_argument("--out-json", required=True, help="Path for the output statistics JSON file.")
    parser.add_argument("--min-perc-valid", type=float, default=0.05,
                        help="Minimum percentage of valid (non-zero/non-empty) data for a column to be kept (default: 0.05).")

    args = parser.parse_args()

    try:
        report = fit_normalize(
            input_csv=args.input,
            target_column=args.target,
            normalized_csv=args.out_data,
            outInitalRes_json=args.out_json,
            minPercValid=args.min_perc_valid
        )

        print(f"outputs/{args.out_data}")
        print(f"outputs/{args.out_json}")
        json.dumps(report, indent=4)

    except Exception as e:
        print(f"[-] An error occurred during preprocessing: {e}")


if __name__ == "__main__":
    main()

"""
>> cd ./scr/qubo_project
python src/qubo_project/preprocessing.py --input data/input_dataset.csv --target target --out-data outputs/normalized.csv --out-json outputs/preprocessing_result.json --min-perc-valid 0.06                                                   
"""