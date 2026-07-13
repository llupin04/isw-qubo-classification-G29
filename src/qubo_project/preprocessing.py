import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def ensure_parent_folder(file_path: str) -> None:
    """Create the output folder if it does not exist."""
    folder = Path(file_path).parent
    if str(folder) != ".":
        folder.mkdir(parents=True, exist_ok=True)


def fit_normalize(
    input_csv: str,
    target_column: str,
    normalized_csv: str,
    outInitalRes_json: str = None,
    minPercValid: float = 0.05,
    **kwargs,
):
    """Load the dataset, remove sparse columns, normalize features and save the result."""
    if outInitalRes_json is None:
        outInitalRes_json = kwargs.get("outInitialRes_json")
    if outInitalRes_json is None:
        raise TypeError("Missing output JSON path")

    input_start = time.time()
    data = pd.read_csv(input_csv)
    input_time = round(time.time() - input_start, 4)

    if target_column not in data.columns:
        raise ValueError(f"Target column '{target_column}' not found")
    if data.empty:
        raise ValueError("Input dataset is empty")

    processing_start = time.time()

    target = pd.to_numeric(data[target_column], errors="coerce")
    if target.isna().any():
        raise ValueError("Target column must contain only numeric values")

    features = data.drop(columns=[target_column]).apply(pd.to_numeric, errors="coerce")
    original_feature_count = features.shape[1]

    kept_columns = []
    dropped_columns = []
    row_count = len(data)

    for column in features.columns:
        valid_values = features[column].notna() & (features[column] != 0)
        valid_percentage = valid_values.sum() / row_count

        if valid_percentage >= minPercValid:
            kept_columns.append(column)
        else:
            dropped_columns.append(column)

    features = features[kept_columns]

    if kept_columns:
        features = features.fillna(features.median(numeric_only=True)).fillna(0.0)
        scaled_features = StandardScaler().fit_transform(features)
        normalized_features = pd.DataFrame(scaled_features, columns=kept_columns)
        normalized_features = normalized_features.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    else:
        normalized_features = pd.DataFrame(index=data.index)

    output_data = normalized_features.copy()
    output_data[target_column] = target.astype(int).values

    ensure_parent_folder(normalized_csv)
    ensure_parent_folder(outInitalRes_json)
    output_data.to_csv(normalized_csv, index=False)

    stats = {
        "n_input_features": int(original_feature_count),
        "n_kept_features": int(len(kept_columns)),
        "dataset_size": int(row_count),
        "dataset_input_time": input_time,
        "dataset_processing_time": round(time.time() - processing_start, 4),
        "dropped_feature_names": dropped_columns,
    }

    with open(outInitalRes_json, "w", encoding="utf-8") as file:
        json.dump(stats, file, indent=4)

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess and normalize a numeric CSV dataset.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--out-data", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--min-perc-valid", type=float, default=0.05)
    args = parser.parse_args()

    fit_normalize(
        input_csv=args.input,
        target_column=args.target,
        normalized_csv=args.out_data,
        outInitalRes_json=args.out_json,
        minPercValid=args.min_perc_valid,
    )


if __name__ == "__main__":
    main()
