import os
import json
import pandas as pd
import streamlit as st

# Import your pipeline functions
from preprocessing import fit_normalize
from feature_selection import select_features
from model import train, predict

# --- Configuration & Path Setup ---
st.set_page_config(page_title="QUBO Loan Classification", layout="wide")

# Define default paths (assuming the app is run from the project root)
PATHS = {
    "raw_data": "data/gui_uploaded_dataset.csv",
    "norm_data": "outputs/gui/gui_normalized.csv",
    "preproc_json": "outputs/gui/gui_preprocessing_result.json",
    "train_data": "outputs/gui/gui_training_reduced.csv",
    "test_data": "outputs/gui/gui_test_reduced.csv",
    "optim_csv": "outputs/gui/gui_optimizations.csv",
    "fs_json": "outputs/gui/gui_feature_selection.json",
    "model": "outputs/gui/gui_model.joblib",
    "train_json": "outputs/gui/gui_training_metrics.json",
    "pred_csv": "outputs/gui/gui_predictions.csv",
    "pred_json": "outputs/gui/gui_classification_stats.json"
}

# Ensure directories exist
os.makedirs("data", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# --- App Header ---
st.title("QUBO-based Loan Classification Pipeline")
st.markdown(
    "Run the binary classification pipeline end-to-end: Preprocessing ➔ QUBO Feature Selection ➔ Training ➔ Evaluation.")


# --- Session State Initialization ---
if 'step_completed' not in st.session_state:
    st.session_state['step_completed'] = {
        'preprocessing': False,
        'feature_selection': False,
        'training': False,
        'prediction': False
    }

# --- Main Tabs ---
tab1, tab2, tab3, tab4 = st.tabs(["1. Preprocessing", "2. Feature Selection", "3. Training", "4. Prediction"])

# ==========================================
# TAB 1: Preprocessing
# ==========================================
with tab1:
    st.header("Step 1: Data Preprocessing")

    uploaded_file = st.file_uploader("Upload your raw CSV dataset", type=["csv"])
    target_col = st.text_input("Target Column Name", value="target")
    min_perc_valid = st.slider("Minimum % of valid data to keep a column", 0.0, 1.0, 0.06, 0.01)

    if st.button("Run Preprocessing", type="primary"):
        if uploaded_file is None:
            st.error("Please upload a dataset first.")
        elif not target_col:
            st.error("Please specify a target column.")
        else:
            with st.spinner("Saving file and running preprocessing..."):
                try:
                    # Save uploaded file to disk so our functions can read the path
                    with open(PATHS["raw_data"], "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    # Run Pipeline Function
                    report = fit_normalize(
                        input_csv=PATHS["raw_data"],
                        target_column=target_col,
                        normalized_csv=PATHS["norm_data"],
                        outInitalRes_json=PATHS["preproc_json"],
                        minPercValid=min_perc_valid
                    )

                    st.session_state['step_completed']['preprocessing'] = True
                    st.success("Preprocessing completed successfully!")


                    st.subheader("Preprocessing Report")
                    # Create a row of 5 columns
                    col1, col2, col3, col4, col5 = st.columns(5)
                    col1.metric('Input Features', report.get('n_input_features', 0))
                    col2.metric('Kept Features', report.get('n_kept_features', 0))
                    col3.metric('Total Rows', report.get('dataset_size', 0))
                    col4.metric('Input Time', report.get('dataset_input_time', 0))
                    col5.metric('Preprocessing Time', report.get('dataset_preprocessing_time', 0))

                    with st.expander("View Raw JSON Report"):
                        st.json(report)

                    st.subheader("Normalized Data Preview")
                    st.dataframe(pd.read_csv(PATHS["norm_data"]))

                except Exception as e:
                    st.error(f"Error during preprocessing: {e}")

# ==========================================
# TAB 2: Feature Selection (QUBO)
# ==========================================
with tab2:
    st.header("Step 2: QUBO Feature Selection")

    ## Lock check: did they finish Preprocessing?
    if not st.session_state['step_completed']['preprocessing']:
        st.warning("**Locked**: You must complete Step 1 (Preprocessing) before accessing Feature Selection.")
    else:
        perc_selected = st.slider("Target % of features to select", 0.05, 1.0, 0.20, 0.05)
        perc_test = st.slider("Test set split percentage", 0.1, 0.5, 0.30, 0.05)
        alpha_comps = st.number_input("Max Alpha Computations (Iterations)", value=10, min_value=1)
        allowance = st.number_input("Feature count allowance", value=1, min_value=0)
        seed = st.number_input("Random Seed", value=42, step=1)

        if st.button("Run QUBO Feature Selection", type="primary"):
            if not st.session_state['step_completed']['preprocessing']:
                st.warning("Please complete Preprocessing first.")
            else:
                with st.spinner("Running Simulated Annealing QUBO search... (this may take a moment)"):
                    try:
                        report = select_features(
                            normalized_csv=PATHS["norm_data"],
                            reducedTrain_csv=PATHS["train_data"],
                            reducedTest_csv=PATHS["test_data"],
                            output_ottim_csv=PATHS["optim_csv"],
                            output_json=PATHS["fs_json"],
                            target_column=target_col,
                            percTest=perc_test,
                            percSelected=perc_selected,
                            allowance=allowance,
                            seed=seed,
                            alpha_computations=alpha_comps
                        )

                        st.session_state['step_completed']['feature_selection'] = True
                        st.success("Feature Selection completed!")


                        st.subheader("Feature Selection Report")
                        col11, col12, col13, col14, col15 = st.columns(5)
                        col11.metric('Number of Features', report.get('n_features', 0))
                        col12.metric('Target Ratio', report.get('target_ratio', 0))
                        col13.metric('Target K', report.get('target_k', 0))
                        col14.metric('Allowance', report.get('allowance', 0))
                        col15.metric('Features Selected', report.get('n_selected', 0))

                        col21, col22, col23, col24, col25 = st.columns(5)
                        col21.metric('Alpha', report.get('alpha', 0))
                        col22.metric('Algorithm', report.get('algorithm', 0))
                        col23.metric('Seed', report.get('seed', 0))
                        col24.metric('Alpha Computations', report.get('alpha_computations', 0))
                        col25.metric('Percentage of Test ', report.get('percTest', 0))

                        col31, col32, col33, col34, col35 = st.columns(5)
                        col31.metric('Training Dataset Size', report.get('training_dataset_size', 0))
                        col32.metric('Test Dataset Size', report.get('test_dataset_size', 0))
                        col33.metric('Q-Matrix Creation Time', report.get('q_matrix_creation_time', 0))
                        col34.metric('Mean Optimization Time', report.get('mean_optimization_time', 0))
                        col35.metric('StDev Optimization Time', report.get('std_dev_optimization_time', 0))

                        with st.expander("View Raw JSON Report"):
                            st.json(report)


                        st.subheader("Optimization Alpha Search Log")
                        st.dataframe(pd.read_csv(PATHS["norm_data"]))

                    except Exception as e:
                        st.error(f"Error during feature selection: {e}")

# ==========================================
# TAB 3: Training
# ==========================================
with tab3:
    st.header("Step 3: Train Classifier")

    ## Lock check: did they finish Preprocessing?
    if not st.session_state['step_completed']['feature_selection']:
        st.warning("**Locked**: You must complete Step 2 (Feature Selection) before training a model.")
    else:

        classifier_choice = st.selectbox(
            "Select Classification Algorithm",
            ["random_forest", "logistic_regression", "gradient_boosting"]
        )
        if st.button("Train Model", type="primary"):
            if not st.session_state['step_completed']['feature_selection']:
                st.warning("Please complete Feature Selection first.")
            else:
                with st.spinner(f"Training {classifier_choice}..."):
                    try:
                        report = train(
                            classifier=classifier_choice,
                            reducedTrain_csv=PATHS["train_data"],
                            target_column=target_col,
                            model_path=PATHS["model"],
                            metrics_json=PATHS["train_json"],
                            seed=seed
                        )

                        st.session_state['step_completed']['training'] = True
                        st.success(f"Model trained and saved to {PATHS['model']}!")

                        st.subheader("Training Metrics")
                        col11, col12, col13, col14 = st.columns(4)
                        col11.metric('Classifier', report.get('classifier', 0))
                        col12.metric('Seed', report.get('seed', 0))
                        col13.metric('Target Column', report.get('target_column', 0))
                        col14.metric('Number of Samples', report.get('n_samples', 0))

                        col21, col22, col23, col24 = st.columns(4)
                        col21.metric('Number of Features', report.get('n_features', 0))
                        col22.metric('Target 1 Percentage', report.get('target_1_percentage', 0))
                        col23.metric('Dataset Input Time', report.get('dataset_input_time', 0))
                        col24.metric('Training Time', report.get('training_time', 0))

                        with st.expander("View Raw JSON Report"):
                            st.json(report)

                        # Provide a download button for the model
                        with open(PATHS["model"], "rb") as f:
                            st.download_button(
                                label="Download Trained Model (.joblib)",
                                data=f,
                                file_name=f"{classifier_choice}_model.joblib",
                                mime="application/octet-stream"
                            )

                    except Exception as e:
                        st.error(f"Error during training: {e}")

# ==========================================
# TAB 4: Prediction & Evaluation
# ==========================================
with tab4:
    st.header("Step 4: Inference on Test Set")

    ## Lock check: did they finish Preprocessing?
    if not st.session_state['step_completed']['training']:
        st.warning("**Locked**: You must complete Step 3 (Training) before running predictions.")
    else:
        st.info(f"This will evaluate the model using the test split generated in Step 2: `{PATHS['test_data']}`")

        if st.button("Run Prediction & Evaluate", type="primary"):
            if not st.session_state['step_completed']['training']:
                st.warning("Please train a model first.")
            else:
                with st.spinner("Running predictions and computing stats..."):
                    try:
                        stats = predict(
                            reduced_Test_csv=PATHS["test_data"],
                            target_column=target_col,
                            model_path=PATHS["model"],
                            predictions_csv=PATHS["pred_csv"],
                            classif_stats_json=PATHS["pred_json"]
                        )

                        st.session_state['step_completed']['prediction'] = True
                        st.success("Inference completed!")

                        # Display top level metrics
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Accuracy", f"{stats['accuracy']:.4f}")
                        c2.metric("ROC-AUC", f"{stats['roc_auc']:.4f}")
                        c3.metric("Class 1 F1-Score", f"{stats['class_1']['f1']:.4f}")

                        col1, col2 = st.columns(2)
                        with col1:
                            st.subheader("Classification Stats")
                            with st.expander("View Raw JSON Report"):
                                st.json(stats)
                        with col2:
                            st.subheader("Predictions Output")
                            pred_df = pd.read_csv(PATHS["pred_csv"])
                            st.dataframe(pred_df)

                            # Download predictions
                            csv = pred_df.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="Download Predictions CSV",
                                data=csv,
                                file_name="predictions.csv",
                                mime="text/csv",
                            )

                    except Exception as e:
                        st.error(f"Error during prediction: {e}")