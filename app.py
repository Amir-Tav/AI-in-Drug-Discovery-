import streamlit as st
import os
import tempfile
import torch
import numpy as np
import pandas as pd

from utils import (
    FingerprintDataset, CNN1D, evaluate_model, train_model,
    plot_classification_metrics, explain_prediction_with_lime,
    explain_prediction_shap_deep, clean_and_save_drug_csv
)

from sklearn.preprocessing import LabelEncoder

# ====================
# Configurations
# ====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "models/cnn_model.pt"

# ====================
# Streamlit UI
# ====================
st.title("Drug-Protein Interaction Classifier (1D CNN + XAI)")

uploaded_file = st.file_uploader("Upload a cleaned or raw CSV file", type=["csv"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
        tmp_file.write(uploaded_file.read())
        raw_csv_path = tmp_file.name

    # Try to read file as cleaned (MultiIndex with meta info)
    try:
        df = pd.read_csv(raw_csv_path, header=[0, 1])
        required_meta_cols = ("meta", "bond_type") in df.columns and ("meta", "drug_name") in df.columns

        if required_meta_cols:
            st.success("✅ Correct input data")
            cleaned = True
        else:
            raise ValueError("Missing required meta columns")

    except Exception:
        st.error("❌ Incorrect input data format.")
        cleaned = False

    # If not cleaned, show button to clean
    if not cleaned:
        if st.button("Clean the data"):
            cleaned_dir = tempfile.mkdtemp()
            drug_name_guess = os.path.basename(raw_csv_path).split(".")[0]

            drug_files = {
                drug_name_guess: {
                    "path": raw_csv_path,
                    "bond_type": "unknown",  # optional: prompt user input
                    "drug_name": drug_name_guess
                }
            }

            clean_and_save_drug_csv(drug_files, cleaned_dir)
            cleaned_path = os.path.join(cleaned_dir, f"cleaned_{drug_name_guess}.csv")

            df = pd.read_csv(cleaned_path, header=[0, 1])
            st.success("✅ Data cleaned successfully!")
            st.write("Cleaned Data Preview:", df.head())
            cleaned = True

    # Proceed only if cleaned correctly
    if cleaned:
        st.write("Preview of Uploaded Data:", df.head())

        # Extract features and labels
        X = df.loc[:, df.columns.get_level_values(0) != "meta"].to_numpy(dtype=np.float32)
        y = df[("meta", "bond_type")].values
        drug_name = df[("meta", "drug_name")].values[0] if ("meta", "drug_name") in df.columns else "drug"

        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        y_labels = le.classes_
        feature_names = [f"{i}" for i in range(X.shape[1])]

        dataset = FingerprintDataset(X, y_encoded)
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

        # ====================
        # Train Model
        # ====================
        st.subheader("Training Model...")
        model = CNN1D(input_length=X.shape[1], num_classes=len(y_labels))
        train_model(model, train_loader, device, epochs=10)  # Fast demo

        os.makedirs("models", exist_ok=True)
        torch.save(model.state_dict(), MODEL_PATH)

        # ====================
        # Evaluate Model
        # ====================
        st.subheader("Evaluation")
        y_true, y_pred = evaluate_model(model, train_loader, device)

        st.text("Classification Report + ROC:")
        plot_classification_metrics(y_true, y_pred, y_labels)

        # ====================
        # LIME Explanation
        # ====================
        st.subheader("LIME Explanation")
        index = st.slider("Select Frame Index for LIME", 0, len(dataset)-1, 0)
        explain_prediction_with_lime(model, dataset, index, y_labels, feature_names, device)
        st.markdown("Saved LIME explanation as `lime_explanation.html`.")

        # ====================
        # SHAP Explanation
        # ====================
        st.subheader("SHAP Explanation")
        if st.button("Run SHAP Explanation"):
            explain_prediction_shap_deep(model, X, X, feature_names, frame_index=index, device=device)
