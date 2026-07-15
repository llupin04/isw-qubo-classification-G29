import sys
import os
import json
import pytest
import pandas as pd
import numpy as np
import joblib

# Add the 'src' directory to the Python path so tests can import your modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from qubo_project.preprocessing import fit_normalize
from qubo_project.feature_selection import select_features
from qubo_project.model import train, predict


# =====================================================================
# Fixtures: Setup & Data Generation
# =====================================================================

@pytest.fixture(scope="module")
def paths():
    """Defines and creates all necessary paths for the test suite."""
    os.makedirs('data', exist_ok=True)
    os.makedirs('outputs/tests', exist_ok=True)

    return {
        'raw_data': 'data/sample_test_dataset.csv',
        'norm_data': 'outputs/tests/normalized.csv',
        'preproc_json': 'outputs/tests/preproc_res.json',
        'train_data': 'outputs/tests/train_reduced.csv',
        'test_data': 'outputs/tests/test_reduced.csv',
        'optim_csv': 'outputs/tests/optimizations.csv',
        'fs_json': 'outputs/tests/fs_result.json',
        'model': 'outputs/tests/model.joblib',
        'train_json': 'outputs/tests/train_metrics.json',
        'pred_csv': 'outputs/tests/predictions.csv',
        'pred_json': 'outputs/tests/predict_stats.json',
        'target_col': 'target'
    }


@pytest.fixture(scope="module")
def mock_dataset(paths):
    """Generates a small dataset tailored to test pipeline edge cases."""
    np.random.seed(42)
    n_rows = 150
    n_features = 20

    data = {}
    for i in range(n_features):
        if i < 3:
            # Sparse columns to test dropping mechanism (mostly NaNs)
            col = np.zeros(n_rows)
            col[col == 0] = np.nan
            col[:5] = np.random.randn(5)
            data[f'feature_sparse_{i}'] = col
        elif i < 6:
            # Columns with a few random missing values to test filling/dropping
            col = np.random.randn(n_rows) * 10
            col[::15] = np.nan
            data[f'feature_missing_{i}'] = col
        else:
            # Dense columns for scaling
            data[f'feature_dense_{i}'] = np.random.randn(n_rows) * 10 + 50

    # Add a target column that is mathematically correlated with a few dense features
    # so the QUBO solver actually has a "reward" to find.
    base_signal = data['feature_dense_10'] + data['feature_dense_11']
    target_values = (base_signal > base_signal.median()).astype(int)
    data[paths['target_col']] = target_values

    df = pd.DataFrame(data)
    df.to_csv(paths['raw_data'], index=False)
    return paths


# =====================================================================
# Fixtures: Pipeline Execution (Runs each phase once for all tests)
# =====================================================================

@pytest.fixture(scope="module")
def run_preprocessing(paths):
    fit_normalize(
        input_csv=paths['raw_data'],
        target_column=paths['target_col'],
        normalized_csv=paths['norm_data'],
        outInitalRes_json=paths['preproc_json'],
        minPercValid=0.05
    )
    return paths


@pytest.fixture(scope="module")
def run_feature_selection(run_preprocessing):
    paths = run_preprocessing
    select_features(
        normalized_csv=paths['norm_data'],
        reducedTrain_csv=paths['train_data'],
        reducedTest_csv=paths['test_data'],
        output_ottim_csv=paths['optim_csv'],
        output_json=paths['fs_json'],
        target_column=paths['target_col'],
        percTest=0.30,
        percSelected=0.20,
        allowance=2,
        alpha_computations=15
    )
    return paths


@pytest.fixture(scope="module")
def run_training(run_feature_selection):
    paths = run_feature_selection
    train(
        classifier="random_forest",
        reducedTrain_csv=paths['train_data'],
        target_column=paths['target_col'],
        model_path=paths['model'],
        metrics_json=paths['train_json']
    )
    return paths


@pytest.fixture(scope="module")
def run_prediction(run_training):
    paths = run_training
    predict(
        reduced_Test_csv=paths['test_data'],
        target_column=paths['target_col'],
        model_path=paths['model'],
        predictions_csv=paths['pred_csv'],
        classif_stats_json=paths['pred_json']
    )
    return paths


# =====================================================================
# The 7 Specific Tests
# =====================================================================

# 1. Test numeric columns
def test_preprocessing_numeric_columns(run_preprocessing):
    df = pd.read_csv(run_preprocessing['norm_data'])
    for col in df.columns:
        assert pd.api.types.is_numeric_dtype(df[col]), f"Column '{col}' is not numeric."


# 2. Test missing values handling
def test_preprocessing_missing_values(run_preprocessing):
    df = pd.read_csv(run_preprocessing['norm_data'])
    assert df.isna().sum().sum() == 0, "Dataset still contains missing values (NaNs)."


# 3. Test normalization output validity (mean 0, std 1)
def test_normalization_valid_dataset(run_preprocessing):
    paths = run_preprocessing
    df = pd.read_csv(paths['norm_data'])

    # Exclude the target column to check only the features
    features = df.drop(columns=[paths['target_col']])

    means = features.mean()
    stds = features.std()

    assert np.allclose(means, 0, atol=1e-7), "Features are not correctly zero-centered."
    assert np.allclose(stds, 1, atol=1e-1), "Features do not have unit variance."


# 4. Test feature selection binary vector
def test_feature_selection_binary_vector(run_feature_selection):
    with open(run_feature_selection['fs_json'], 'r') as f:
        report = json.load(f)

    vector = report['selected_vector']
    assert set(vector).issubset({0, 1}), "Selected vector contains non-binary values."


# 5. Test approximately 20% selected (and not less than 10%)
def test_feature_selection_percentage(run_feature_selection):
    with open(run_feature_selection['fs_json'], 'r') as f:
        report = json.load(f)

    n_features = report['n_features']
    n_selected = report['n_selected']
    target_k = report['target_k']
    allowance = report['allowance']

    assert n_selected >= (
                n_features * 0.10), f"Too few features selected ({n_selected} out of {n_features}). Must be >= 10%."
    assert abs(
        n_selected - target_k) <= allowance, f"Selected count ({n_selected}) outside allowance range of target ({target_k})."


# 6. Test training produces a saved model
def test_training_saved_model(run_training):
    model_path = run_training['model']

    assert os.path.exists(model_path), "Trained model (.joblib) was not saved."

    # Verify the file is a valid joblib object by loading it
    model = joblib.load(model_path)
    assert model is not None, "Saved model could not be loaded."
    assert hasattr(model, 'predict'), "Loaded object is not a valid scikit-learn estimator."


# 7. Test prediction CSV columns
def test_prediction_required_columns(run_prediction):
    df_pred = pd.read_csv(run_prediction['pred_csv'])
    expected_columns = ['row_n', 'target', 'prediction', 'score']

    for col in expected_columns:
        assert col in df_pred.columns, f"Required column '{col}' is missing from predictions output."

    assert len(df_pred) > 0, "Predictions CSV is empty."


"""
pytest .\tests\automated_tests.py -v
"""