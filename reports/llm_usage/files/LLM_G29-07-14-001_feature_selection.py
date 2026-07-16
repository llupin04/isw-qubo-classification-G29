import time
import json
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
    # ---------------------------------------------------------
    # 1. Load Data & Prepare Target Parameters
    # ---------------------------------------------------------
    df = pd.read_csv(normalized_csv)

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found.")

    features = [col for col in df.columns if col != target_column]
    m = len(features)
    target_k = int(round(percSelected * m))

    # ---------------------------------------------------------
    # 2. Compute Base Matrices (Spearman Correlation)
    # ---------------------------------------------------------
    t_q_start = time.time()

    # Calculate absolute Spearman correlation matrix for the whole dataset
    corr_matrix = df.corr(method='spearman').abs()

    # Extract influence (feature vs target) and redundancy (feature vs feature)
    corr_y = corr_matrix[target_column].drop(labels=[target_column])
    corr_X = corr_matrix.drop(index=[target_column], columns=[target_column])

    q_matrix_creation_time = time.time() - t_q_start

    # ---------------------------------------------------------
    # 3. Alpha Search via QUBO & Simulated Annealing
    # ---------------------------------------------------------
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

        # Build Q matrix dictionary for D-Wave Neal
        Q = {}
        for i in range(m):
            f_i = features[i]
            # Linear terms (reward: negative cost)
            Q[(f_i, f_i)] = -alpha * corr_y[f_i]

            # Quadratic terms (penalty: positive cost)
            for j in range(i + 1, m):
                f_j = features[j]
                Q[(f_i, f_j)] = (1 - alpha) * corr_X.loc[f_i, f_j]

        # Solve QUBO
        t_opt_start = time.time()
        sampleset = sampler.sample_qubo(Q, seed=seed)
        opt_time = time.time() - t_opt_start
        opt_times.append(opt_time)

        # Extract best sample from this run
        best_sample = sampleset.first.sample
        energy = sampleset.first.energy

        selected_vector = [best_sample[f] for f in features]
        n_selected = sum(selected_vector)

        attempts_log.append({
            "alpha": alpha,
            "n_selected": n_selected,
            "cost_energy": energy,
            "optimization_time": opt_time
        })

        # Track the closest match
        dist = abs(n_selected - target_k)
        if dist < best_dist:
            best_dist = dist
            best_alpha = alpha
            best_vector = selected_vector
            best_n = n_selected

        # Binary Search Logic
        if abs(n_selected - target_k) <= allowance:
            break  # Target range achieved
        elif n_selected < target_k - allowance:
            # Too few features -> increase alpha (increase influence reward)
            low = alpha
        else:
            # Too many features -> decrease alpha (increase redundancy penalty)
            high = alpha

    # ---------------------------------------------------------
    # 4. Save Alpha Attempts Log
    # ---------------------------------------------------------
    # Sort logs by alpha as requested
    pd.DataFrame(attempts_log).sort_values(by='alpha').to_csv(output_ottim_csv, index=False)

    # ---------------------------------------------------------
    # 5. Filter Dataset & Split Train/Test
    # ---------------------------------------------------------
    selected_feature_names = [features[i] for i in range(m) if best_vector[i] == 1]

    # Keep selected features + the target at the end
    df_reduced = df[selected_feature_names + [target_column]]

    df_train, df_test = train_test_split(df_reduced, test_size=percTest, random_state=seed)

    df_train.to_csv(reducedTrain_csv, index=False)
    df_test.to_csv(reducedTest_csv, index=False)

    # ---------------------------------------------------------
    # 6. Generate and Save JSON Report
    # ---------------------------------------------------------
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