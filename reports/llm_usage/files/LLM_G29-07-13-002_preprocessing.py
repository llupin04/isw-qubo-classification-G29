import time
import json
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
    # ---------------------------------------------------------
    # 1. Load Dataset & Track Loading Time
    # ---------------------------------------------------------
    t_load_start = time.time()

    # We read the entire dataset into memory.
    # (See the Scalability Considerations below for datasets exceeding RAM limits)
    df = pd.read_csv(input_csv)

    load_time = time.time() - t_load_start
    t_process_start = time.time()

    total_rows = len(df)

    # ---------------------------------------------------------
    # 2. Separate Target from Features
    # ---------------------------------------------------------
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in the dataset.")

    y = df[target_column]
    X = df.drop(columns=[target_column])

    original_feature_cols = X.columns.tolist()
    num_input_features = len(original_feature_cols)

    # ---------------------------------------------------------
    # 3. Filter "Empty or Nearly Empty" Columns
    # ---------------------------------------------------------
    # A value is valid if it is not NaN and not equal to 0.
    is_valid = X.notna() & (X != 0)

    # Calculate the percentage of valid data per column
    valid_percentages = is_valid.sum() / total_rows

    # Identify which columns to keep and which to drop
    kept_features = valid_percentages[valid_percentages >= minPercValid].index.tolist()
    dropped_features = valid_percentages[valid_percentages < minPercValid].index.tolist()

    X_filtered = X[kept_features]

    # ---------------------------------------------------------
    # 4. Normalize Remaining Feature Columns
    # ---------------------------------------------------------
    # We use StandardScaler for z-score standardization (mean=0, std=1)
    scaler = StandardScaler()

    # Fit and transform the data. This returns a NumPy array.
    X_scaled_array = scaler.fit_transform(X_filtered)

    # Convert back to a DataFrame to preserve column headers
    X_scaled = pd.DataFrame(X_scaled_array, columns=kept_features, index=X_filtered.index)

    # ---------------------------------------------------------
    # 5. Recombine and Write Cleaned Dataset
    # ---------------------------------------------------------
    # Concatenate the scaled features with the untouched target column
    final_df = pd.concat([X_scaled, y], axis=1)

    # Write to CSV, preserving the headers in the first row
    final_df.to_csv(normalized_csv, index=False)

    process_time = time.time() - t_process_start

    # ---------------------------------------------------------
    # 6. Generate and Save JSON Report
    # ---------------------------------------------------------
    report = {
        "dataset_rows": total_rows,
        "num_input_features": num_input_features,
        "num_kept_features": len(kept_features),
        "num_dropped_features": len(dropped_features),
        "load_time_seconds": round(load_time, 4),
        "process_time_seconds": round(process_time, 4),
        "dropped_features": dropped_features
    }

    with open(outInitalRes_json, "w") as f:
        json.dump(report, f, indent=4)

    return report