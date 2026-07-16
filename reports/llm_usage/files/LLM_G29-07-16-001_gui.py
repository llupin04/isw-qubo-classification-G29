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
    "norm_data": "outputs/gui_normalized.csv",
    "preproc_json": "reports/gui_preprocessing_result.json",
    "train_data": "outputs/gui_training_reduced.csv",
    "test_data": "outputs/gui_test_reduced.csv",
    "optim_csv": "outputs/gui_optimizations.csv",
    "fs_json": "reports/gui_feature_selection.json",
    "model": "outputs/gui_model.joblib",
    "train_json": "reports/gui_training_metrics.json",
    "pred_csv": "outputs/gui_predictions.csv",
    "pred_json": "reports/gui_classification_stats.json"
}

# Ensure directories exist
os.makedirs("data", exist_ok=True)
os.makedirs("outputs", exist_ok=True)
os.makedirs("reports", exist_ok=True)

# --- App Header ---
st.title("🏦 QUBO-based Loan Classification Pipeline")
st.markdown(
    "Run the binary classification pipeline end-to-end: Preprocessing ➔ QUBO Feature Selection ➔ Training ➔ Evaluation.")

# --- Sidebar Configuration ---
st.sidebar.header("⚙️ Global Parameters")
target_col = st.sidebar.text_input("Target Column Name", value="target")
seed = st.sidebar.number_input("Random Seed", value=42, step=1)

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
    min_perc_valid = st.slider("Minimum % of valid data to keep a column", 0.0, 1.0, 0.05, 0.01)

    if st.button("Run Preprocessing", type="primary"):
        if uploaded_file is None:
            st.error("❌ Please upload a dataset first.")
        elif not target_col:
            st.error("❌ Please specify a target column in the sidebar.")
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
                    st.success("✅ Preprocessing completed successfully!")

                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Preprocessing Report")
                        st.json(report)
                    with col2:
                        st.subheader("Normalized Data Preview")
                        st.dataframe(pd.read_csv(PATHS["norm_data"]).head(10))

                except Exception as e:
                    st.error(f"❌ Error during preprocessing: {e}")

# ==========================================
# TAB 2: Feature Selection (QUBO)
# ==========================================
with tab2:
    st.header("Step 2: QUBO Feature Selection")

    perc_selected = st.slider("Target % of features to select", 0.05, 1.0, 0.20, 0.05)
    perc_test = st.slider("Test set split percentage", 0.1, 0.5, 0.30, 0.05)
    alpha_comps = st.number_input("Max Alpha Computations (Iterations)", value=10, min_value=1)
    allowance = st.number_input("Feature count allowance", value=1, min_value=0)

    if st.button("Run QUBO Feature Selection", type="primary"):
        if not st.session_state['step_completed']['preprocessing']:
            st.warning("⚠️ Please complete Preprocessing first.")
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
                    st.success("✅ Feature Selection completed!")

                    col1, col2 = st.columns([1, 1])
                    with col1:
                        st.subheader("Feature Selection Report")
                        st.json(report)
                    with col2:
                        st.subheader("Optimization Alpha Search Log")
                        st.dataframe(pd.read_csv(PATHS["optim_csv"]))

                except Exception as e:
                    st.error(f"❌ Error during feature selection: {e}")

# ==========================================
# TAB 3: Training
# ==========================================
with tab3:
    st.header("Step 3: Train Classifier")

    classifier_choice = st.selectbox(
        "Select Classification Algorithm",
        ["random_forest", "logistic_regression", "gradient_boosting"]
    )

    if st.button("Train Model", type="primary"):
        if not st.session_state['step_completed']['feature_selection']:
            st.warning("⚠️ Please complete Feature Selection first.")
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
                    st.success(f"✅ Model trained and saved to {PATHS['model']}!")

                    st.subheader("Training Metrics")
                    st.json(report)

                    # Provide a download button for the model
                    with open(PATHS["model"], "rb") as f:
                        st.download_button(
                            label="💾 Download Trained Model (.joblib)",
                            data=f,
                            file_name=f"{classifier_choice}_model.joblib",
                            mime="application/octet-stream"
                        )

                except Exception as e:
                    st.error(f"❌ Error during training: {e}")

# ==========================================
# TAB 4: Prediction & Evaluation
# ==========================================
with tab4:
    st.header("Step 4: Inference on Test Set")
    st.info(f"This will evaluate the model using the test split generated in Step 2: `{PATHS['test_data']}`")

    if st.button("Run Prediction & Evaluate", type="primary"):
        if not st.session_state['step_completed']['training']:
            st.warning("⚠️ Please train a model first.")
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
                    st.success("✅ Inference completed!")

                    # Display top level metrics
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Accuracy", f"{stats['accuracy']:.4f}")
                    c2.metric("ROC-AUC", f"{stats['roc_auc']:.4f}")
                    c3.metric("Class 1 F1-Score", f"{stats['class_1']['f1']:.4f}")

                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Classification Stats")
                        st.json(stats)
                    with col2:
                        st.subheader("Predictions Output")
                        pred_df = pd.read_csv(PATHS["pred_csv"])
                        st.dataframe(pred_df.head(10))

                        # Download predictions
                        csv = pred_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="💾 Download Predictions CSV",
                            data=csv,
                            file_name="predictions.csv",
                            mime="text/csv",
                        )

                except Exception as e:
                    st.error(f"❌ Error during prediction: {e}")