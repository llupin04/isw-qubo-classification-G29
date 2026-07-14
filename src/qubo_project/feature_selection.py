import argparse
import time
import json
import os
import pandas as pd
import numpy as np
import neal
from sklearn.model_selection import train_test_split


def select_features(
        normalized_csv: str,
        reducedTrain_csv: str,
        reducedTest_csv: str,
        output_ottim_csv: str,
        output_json: str,
        target_column: str,
        percTest: float = 0.30,
        percSelected: float = 0.20,
        allowance: int = 1,
        seed: int = 42,
        alpha_computations: int = 100
):
    """
    Selects a subset of features using a QUBO formulation solved via Simulated Annealing.
    Balances influence on the target (alpha) vs. independence from other features (1-alpha).
    """
    # Load Data
    df = pd.read_csv("../../outputs/" + normalized_csv)

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in the dataset.")

    features = [col for col in df.columns if col != target_column]
    m = len(features)
    target_k = int(round(percSelected * m))

    # Compute Base Matrices (Spearman Correlation)
    t_q_start = time.time()
    corr_matrix = df.corr(method='spearman').abs()
    corr_y = corr_matrix[target_column].drop(labels=[target_column])
    corr_X = corr_matrix.drop(index=[target_column], columns=[target_column])
    q_matrix_creation_time = time.time() - t_q_start

    # Alpha Search via QUBO & Simulated Annealing
    sampler = neal.SimulatedAnnealingSampler()

    low, high = 0.0, 1.0
    best_alpha = 0.5
    best_vector = []
    best_n = -1
    best_dist = float('inf')

    attempts_log = []
    opt_times = []

    for attempt in range(alpha_computations):
        alpha = (low + high) / 2.0

        # Build Q matrix dictionary
        Q = {}
        for i in range(m):
            f_i = features[i]
            Q[(f_i, f_i)] = -alpha * corr_y[f_i]
            for j in range(i + 1, m):
                f_j = features[j]
                Q[(f_i, f_j)] = (1 - alpha) * corr_X.loc[f_i, f_j]

        # Solve QUBO
        t_opt_start = time.time()
        sampleset = sampler.sample_qubo(Q, seed=seed)
        opt_time = time.time() - t_opt_start
        opt_times.append(opt_time)

        # Extract best sample
        best_sample = sampleset.first.sample
        energy = float(sampleset.first.energy)

        selected_vector = [int(best_sample[f]) for f in features]
        n_selected = int(sum(selected_vector))

        attempts_log.append({
            "alpha": alpha,
            "n_selected": n_selected,
            "cost_energy": energy,
            "optimization_time": opt_time
        })

        # Track best match
        dist = abs(n_selected - target_k)
        if dist < best_dist:
            best_dist = dist
            best_alpha = alpha
            best_vector = selected_vector
            best_n = n_selected

        # Binary Search Logic
        if abs(n_selected - target_k) <= allowance:
            break
        elif n_selected < target_k - allowance:
            low = alpha
        else:
            high = alpha

    # Ensure output directories exist
    os.makedirs(os.path.dirname("../../outputs" + reducedTrain_csv) or '.', exist_ok=True)
    os.makedirs(os.path.dirname("../../outputs" + reducedTest_csv) or '.', exist_ok=True)
    os.makedirs(os.path.dirname("../../outputs" + output_ottim_csv) or '.', exist_ok=True)
    os.makedirs(os.path.dirname("../../outputs" + output_json) or '.', exist_ok=True)

    # Save Alpha Attempts Log
    pd.DataFrame(attempts_log).sort_values(by='alpha').to_csv(output_ottim_csv, index=False)

    # Filter Dataset & Split Train/Test
    selected_feature_names = [features[i] for i in range(m) if best_vector[i] == 1]
    df_reduced = df[selected_feature_names + [target_column]]

    df_train, df_test = train_test_split(df_reduced, test_size=percTest, random_state=seed)

    df_train.to_csv(reducedTrain_csv, index=False)
    df_test.to_csv(reducedTest_csv, index=False)

    # Generate and Save JSON Report
    report = {
        "n_features": m,
        "target_ratio": percSelected,
        "target_k": target_k,
        "allowance": allowance,
        "n_selected": best_n,
        "alpha": round(best_alpha, 3),
        "selected_vector": best_vector,
        "selected_feature_names": selected_feature_names,
        "algorithm": "simulated_annealing",
        "seed": seed,
        "alpha_computations": len(attempts_log),
        "percTest": percTest,
        "training_dataset_size": len(df_train),
        "test_dataset_size": len(df_test),
        "q_matrix_creation_time": round(q_matrix_creation_time, 2),
        "mean_optimization_time": round(float(np.mean(opt_times)), 2),
        "std_dev_optimization_time": round(float(np.std(opt_times)), 3)
    }

    with open(output_json, "w") as f:
        json.dump(report, f, indent=4)

    return report


def main():
    parser = argparse.ArgumentParser(description="QUBO-based Feature Selection for binary classification.")

    # Define expected CLI arguments
    parser.add_argument("--in-normalized", required=True, help="Path to the input normalized CSV dataset.")
    parser.add_argument("--out-train", required=True, help="Path for the output reduced training CSV file.")
    parser.add_argument("--out-test", required=True, help="Path for the output reduced test CSV file.")
    parser.add_argument("--out-optimizations", required=True, help="Path for the output optimizations CSV log.")
    parser.add_argument("--out-json", required=True, help="Path for the output statistics JSON file.")
    parser.add_argument("--target", required=True, help="Name of the target column.")
    parser.add_argument("--perc-selected", type=float, default=0.20,
                        help="Percentage of features to select (default: 0.20).")
    parser.add_argument("--allowance", type=int, default=1,
                        help="Allowance around the target feature count (default: 1).")
    parser.add_argument("--perc-test", type=float, default=0.30, help="Percentage of test data (default: 0.30).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42).")
    parser.add_argument("--alpha-computations", type=int, default=10,
                        help="Max iterations for alpha search (default: 10).")

    args = parser.parse_args()

    try:
        report = select_features(
            normalized_csv=args.in_normalized,
            reducedTrain_csv="../../outputs/" + args.out_train,
            reducedTest_csv="../../outputs/" + args.out_test,
            output_ottim_csv="../../outputs/" + args.out_optimizations,
            output_json="../../outputs/" + args.out_json,
            target_column=args.target,
            percTest=args.perc_test,
            percSelected=args.perc_selected,
            allowance=args.allowance,
            seed=args.seed,
            alpha_computations=args.alpha_computations
        )

        print(f"outputs/{args.out_train}")
        print(f"outputs/{args.out_test}")
        print(f"outputs/{args.out_optimizations}")
        print(f"outputs/{args.out_json}")

    except Exception as e:
        print(f"[-] An error occurred during feature selection: {e}")


if __name__ == "__main__":
    main()