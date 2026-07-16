# QUBO-based Feature Selection & Binary Classification Pipeline

## Overview
This repository contains a machine learning pipeline for binary classification tasks.\
The core feature of this project is the use of a QUBO formulation to perform feature selection before model training. 

## Prerequisites
*   **Python:** Version 3.11 or higher is mandatory.

* Install the **required dependencies**:
    ```bash
    pip install -r requirements.txt
    ```



## Command Line Interface (CLI) Usage

The pipeline can be executed via the command line.\
Remember to follow step 1-4 sequentially.

### 1. Preprocessing

Reads the dataset, drops columns with excessive missing/zero values, normalizes the remaining features (Z-score), and exports the result:

```bash
python src/qubo_project/preprocessing.py --input data/input_dataset.csv --target target --out-data outputs/normalized.csv --out-json outputs/preprocessing_result.json --min-perc-valid 0.05
```

### 2. Feature Selection (QUBO)

Selects the optimal subset of features by balancing target influence and feature independence via Simulated Annealing, then splits the data into training and test sets:

```bash
python src/qubo_project/feature_selection.py --in-normalized outputs/normalized.csv --out-train outputs/training_reduced.csv --out-test outputs/test_reduced.csv --out-optimizations outputs/optimizations.csv --out-json outputs/feature_selection_result.json --target target --perc-test 0.30 --allowance 1 --seed 42 --perc-selected 0.20 --alpha-computations 10
```

### 3. Model Training

Trains a binary classifier (`random_forest`, `logistic_regression`, or `gradient_boosting`) on the reduced features and saves the model:

```bash
python src/qubo_project/model.py train --classifier random_forest --in-reduced outputs/training_reduced.csv --target target --out-model outputs/model.joblib --out-metrics outputs/training_metrics.json --seed 42
```

### 4. Model Prediction

Loads the trained model to perform inference on the test set, computing various classification metrics:

```bash
python src/qubo_project/model.py predict --input-testset outputs/test_reduced.csv --target target --model outputs/model.joblib --out-predictions outputs/predictions.csv --out-stats outputs/classification_stats.json
```

## Graphical User Interface (GUI)

If you don't want to use the CLI, you can execute the pipeline via GUI:

```bash
streamlit run src/qubo_project/gui.py
```

## Automated Testing

To run the test suite and verify the integrity of the data transformations and model compatibility:

```bash
pytest tests/automated_tests.py -v
```
