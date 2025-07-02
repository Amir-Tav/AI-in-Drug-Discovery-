import streamlit as st
import os
import pandas as pd
import torch
import numpy as np
import streamlit.components.v1 as components

from utils import (
    FingerprintDataset,
    ResNet1D,
    clean_and_save_drug_csv,
    evaluate_on_new_csv,
    explain_with_lime,
    explain_with_shap
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
test_save_dir = "Data/Test"
os.makedirs(test_save_dir, exist_ok=True)

st.set_page_config(page_title="DAT Classifier", layout="centered")
st.title("🧬 DAT Bond Type Classifier")

uploaded_file = st.file_uploader("📂 Upload a cleaned or raw CSV file", type=["csv"])

def highlight_wrong_preds(row):
    return ['background-color: #ffcccc' if row['True Label'] != row['Predicted'] else '' for _ in row]

if uploaded_file:
    file_name = uploaded_file.name
    raw_path = os.path.join(test_save_dir, file_name)

    with open(raw_path, "wb") as f:
        f.write(uploaded_file.read())
    st.write(f"📄 Uploaded: `{file_name}`")

    try:
        df = pd.read_csv(raw_path, header=[0, 1])
        if ("meta", "bond_type") in df.columns and ("meta", "drug_name") in df.columns:
            st.success("✅ Cleaned format detected.")
            cleaned_path = raw_path
        else:
            raise Exception("Missing required meta columns")
    except Exception:
        st.warning("⚠️ Raw file detected. Please clean it first.")
        cleaned_path = None

    if cleaned_path is None and st.button("🧼 Clean Uploaded File"):
        guessed_name = os.path.splitext(file_name)[0]
        config = {
            guessed_name: {
                "path": raw_path,
                "bond_type": "unknown",
                "drug_name": guessed_name
            }
        }
        clean_and_save_drug_csv(config, test_save_dir)
        cleaned_path = os.path.join(test_save_dir, f"cleaned_{guessed_name}.csv")
        df = pd.read_csv(cleaned_path, header=[0, 1])
        st.success("🧹 File cleaned and saved.")
        st.write("Preview of cleaned data:", df.head())

    if "results" not in st.session_state:
        st.session_state["results"] = None
    if "fig_cm" not in st.session_state:
        st.session_state["fig_cm"] = None

    if cleaned_path:
        st.markdown("---")
        st.subheader("🧠 Choose a Model")
        model_options = {
            "ResNet1D (default)": "models/transfer/resnet1d_model.pt"
        }
        model_choice = st.selectbox("Available Models", list(model_options.keys()))
        model_path = model_options[model_choice]

        st.markdown("🔍 Press predict to classify bond type.")
        if st.button("🚀 Predict Bond Type"):
            try:
                y_labels = np.load("models/transfer/y_labels.npy", allow_pickle=True)
            except FileNotFoundError:
                st.error("❌ y_labels.npy not found in 'models/transfer'")
            else:
                st.info("Predicting on uploaded data...")
                results, fig_cm = evaluate_on_new_csv(cleaned_path, model_path, y_labels, device)
                st.session_state["results"] = results
                st.session_state["fig_cm"] = fig_cm
                st.session_state["y_labels"] = y_labels
                st.session_state["model_path"] = model_path
                st.session_state["cleaned_path"] = cleaned_path
                st.success("✅ Prediction Complete")

        if st.session_state["results"] is not None:
            st.subheader("📊 Sample Predictions (Scrollable)")
            styled_df = st.session_state["results"].style.apply(highlight_wrong_preds, axis=1)
            st.dataframe(styled_df, height=400)

            st.subheader("📊 Confusion Matrix")
            st.pyplot(st.session_state["fig_cm"])

            # Single selectbox for explanation method
            explanation_method = st.selectbox("🧪 Choose Explanation Method", options=["None", "LIME", "SHAP"])

            if explanation_method == "LIME":
                st.info("Running LIME explanation...")

                y_labels = st.session_state.get("y_labels")
                if y_labels is None:
                    y_labels = np.load("models/transfer/y_labels.npy", allow_pickle=True)
                model_path = st.session_state.get("model_path")
                cleaned_path = st.session_state.get("cleaned_path")

                model = ResNet1D(input_channels=1, num_classes=len(y_labels))
                model.load_state_dict(torch.load(model_path, map_location=device))
                model.to(device)
                model.eval()

                df = pd.read_csv(cleaned_path, header=[0, 1])
                X_test = df.loc[:, df.columns.get_level_values(0) != "meta"].to_numpy(dtype=np.float32)
                feature_names = [f"{res}-{inter}" for res, inter in df.columns if res != "meta"]

                lime_html = explain_with_lime(model, X_test, y_labels, feature_names, device, frame_index=20)

                # Replace default LIME colors: orange->red, green->bright green
                lime_html = lime_html.replace("#ff7f0e", "#d62728")  # orange -> red
                lime_html = lime_html.replace("#2ca02c", "#2ecc40")  # green -> bright green

                components.html(lime_html, height=800)

            elif explanation_method == "SHAP":
                st.info("Running SHAP explanation...")

                y_labels = st.session_state.get("y_labels")
                if y_labels is None:
                    y_labels = np.load("models/transfer/y_labels.npy", allow_pickle=True)
                model_path = st.session_state.get("model_path")
                cleaned_path = st.session_state.get("cleaned_path")

                model = ResNet1D(input_channels=1, num_classes=len(y_labels))
                model.load_state_dict(torch.load(model_path, map_location=device))
                model.to(device)
                model.eval()

                df = pd.read_csv(cleaned_path, header=[0, 1])
                X_test = df.loc[:, df.columns.get_level_values(0) != "meta"].to_numpy(dtype=np.float32)
                feature_names = [f"{res}-{inter}" for res, inter in df.columns if res != "meta"]

                fig, pred_label, confidence = explain_with_shap(model, X_test, y_labels, feature_names, device, frame_index=20)
                st.write(f"Prediction for frame 20: **{pred_label.upper()}** (Confidence: {confidence:.2f})")
                st.pyplot(fig)
