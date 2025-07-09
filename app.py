import streamlit as st
import os
import pandas as pd
import torch
import numpy as np
import streamlit.components.v1 as components
import torch.nn.functional as F
import matplotlib.pyplot as plt
import shutil
import glob


from utils import (
    FingerprintDataset,
    ExpandedResNet1D,
    preprocess_input_csv,
    evaluate_on_new_csv,
    evaluate_with_minirocket,
    explain_with_lime,
    explain_with_shap,
    explain_minirocket_lime,
    explain_minirocket_shap,
    evaluate_chunked_confidence
)

# Setup Streamlit and paths
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
test_save_dir = "Data/Test"
os.makedirs(test_save_dir, exist_ok=True)
tani_save_dir = os.path.join(test_save_dir, "Tanimoto")
os.makedirs(tani_save_dir, exist_ok=True)

# Clean up all files inside Data/Test and its Tanimoto subfolder
for folder in [test_save_dir, os.path.join(test_save_dir, "Tanimoto")]:
    if os.path.exists(folder):
        for f in glob.glob(os.path.join(folder, "*")):
            try:
                os.remove(f)
            except Exception as e:
                print(f"⚠️ Failed to remove {f}: {e}")

st.set_page_config(page_title="DAT Bond Type Classifier", layout="centered")
st.title("🧬 DAT Bond Type Classifier")

@st.cache_resource(show_spinner=False)
def get_lime_explainer(X_train, y_labels, feature_names):
    from lime.lime_tabular import LimeTabularExplainer
    return LimeTabularExplainer(
        training_data=X_train,
        mode="classification",
        class_names=y_labels.tolist(),
        feature_names=feature_names,
        discretize_continuous=False
    )

@st.cache_data(show_spinner=False)
def cached_predict_fn(_model_state_dict, input_channels, num_classes, device, inputs):
    model = ExpandedResNet1D(input_channels=input_channels, num_classes=num_classes)
    model.load_state_dict(_model_state_dict)
    model.to(device)
    model.eval()
    inputs_tensor = torch.tensor(inputs[:, np.newaxis, :], dtype=torch.float32).to(device)
    with torch.no_grad():
        outputs = model(inputs_tensor)
        return F.softmax(outputs, dim=1).cpu().numpy()

def show_tanimoto_tab():
    st.subheader("🔬 Tanimoto Comparison: Dual Dataset")

    col1, col2 = st.columns(2)
    with col1:
        file_1 = st.file_uploader("📂 Dataset A", type=["csv"], key="tani_a")
    with col2:
        file_2 = st.file_uploader("📂 Dataset B", type=["csv"], key="tani_b")

    if file_1 and file_2:
        def load_csv(uploaded_file):
            path = os.path.join(tani_save_dir, uploaded_file.name)
            with open(path, "wb") as f:
                f.write(uploaded_file.read())
            try:
                df = pd.read_csv(path, header=[0, 1])
                if ("meta", "frame") not in df.columns:
                    st.error(f"❌ Missing 'meta-frame' column in `{uploaded_file.name}`.")
                    return None
                return df
            except Exception as e:
                st.error(f"❌ Failed to read `{uploaded_file.name}`: {e}")
                return None

        df1 = load_csv(file_1)
        df2 = load_csv(file_2)

        if df1 is not None and df2 is not None:
            st.info("Running chunked confidence analysis on both datasets...")

            conf1 = evaluate_chunked_confidence(
                df=df1,
                model_path=st.session_state["model_path"],
                model_choice=st.session_state["model_choice"],
                y_labels=st.session_state["y_labels"],
                device=device,
                chunk_size=20           #20 frame 
            )

            conf2 = evaluate_chunked_confidence(
                df=df2,
                model_path=st.session_state["model_path"],
                model_choice=st.session_state["model_choice"],
                y_labels=st.session_state["y_labels"],
                device=device,
                chunk_size=20           #20 frame 
            )

            min_len = min(len(conf1), len(conf2))
            x = list(range(1, min_len + 1))

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(x, conf1[:min_len], marker='o', label=file_1.name, color='tab:blue')
            ax.plot(x, conf2[:min_len], marker='s', label=file_2.name, color='tab:orange')
            ax.set_xlabel("Chunk # (every 20 frames)")
            ax.set_ylabel("Model Confidence")
            ax.set_title("Model Confidence Comparison")
            ax.set_ylim(0, 1)
            ax.legend(loc='lower right', title="Datasets")
            st.pyplot(fig)

def show_explanation_tab():
    st.subheader("🧪 Explanation Methods")

    if st.session_state["model_choice"] == "MiniRocket + LogisticRegression":
        explanation_options = ["None", "LIME", "SHAP"]
        aggregate_lime_available = False
    else:
        explanation_options = ["None", "LIME", "SHAP", "Aggregate LIME"]
        aggregate_lime_available = True

    explanation_method = st.selectbox("Choose Explanation Method", options=explanation_options)
    df = pd.read_csv(st.session_state["cleaned_path"], header=[0, 1])
    X_test = df.loc[:, df.columns.get_level_values(0) != "meta"].to_numpy(dtype=np.float32)
    frame_index = st.number_input("Select Frame Number for Explanation", 0, len(X_test)-1, 20)

    if explanation_method == "LIME":
        st.info(f"Running LIME explanation for frame {frame_index}...")

        if st.session_state["model_choice"] == "MiniRocket + LogisticRegression":
            import joblib
            from sktime.transformations.panel.rocket import MiniRocket

            clf = joblib.load(st.session_state["model_path"])
            rocket = joblib.load("models/v2/minirocket_transformer.joblib")
            X_tf = rocket.transform(X_test.reshape(X_test.shape[0], 1, X_test.shape[1]))

            # 👇 Use generic but safe feature names
            feature_names = [f"MRF{i}" for i in range(X_tf.shape[1])]

            lime_html = explain_minirocket_lime(
                clf, X_tf, X_tf,
                st.session_state["y_labels"],
                frame_index,
                feature_names
            )
        else:
            feature_names = [f"{res}-{inter}" for res, inter in df.columns if res != "meta"]
            model = ExpandedResNet1D(input_channels=1, num_classes=len(st.session_state["y_labels"]))
            model.load_state_dict(torch.load(st.session_state["model_path"], map_location=device))
            model.to(device)
            model.eval()
            lime_html = explain_with_lime(model, X_test, st.session_state["y_labels"], feature_names, device, frame_index)

        wrapped_html = f"""
        <div style="background-color: white; padding: 15px; border-radius: 8px;">
            <h3>Frame {frame_index} Explanation</h3>
            {lime_html}
        </div>
        """
        components.html(wrapped_html, height=1000)

    elif explanation_method == "SHAP":
        st.info(f"Running SHAP explanation for frame {frame_index}...")

        if st.session_state["model_choice"] == "MiniRocket + LogisticRegression":
            import joblib
            from sktime.transformations.panel.rocket import MiniRocket

            clf = joblib.load(st.session_state["model_path"])
            rocket = joblib.load("models/v2/minirocket_transformer.joblib")
            X_tf = rocket.transform(X_test.reshape(X_test.shape[0], 1, X_test.shape[1]))

            # 👇 Use generic but aligned feature names
            feature_names = [f"MRF{i}" for i in range(X_tf.shape[1])]

            fig, pred_label, confidence = explain_minirocket_shap(
                clf, X_tf, X_tf,
                st.session_state["y_labels"],
                frame_index,
                feature_names
            )
        else:
            feature_names = [f"{res}-{inter}" for res, inter in df.columns if res != "meta"]
            model = ExpandedResNet1D(input_channels=1, num_classes=len(st.session_state["y_labels"]))
            model.load_state_dict(torch.load(st.session_state["model_path"], map_location=device))
            model.to(device)
            model.eval()
            fig, pred_label, confidence = explain_with_shap(model, X_test, st.session_state["y_labels"], feature_names, device, frame_index)

        st.write(f"Frame {frame_index} Prediction: **{pred_label.upper()}** (Confidence: {confidence:.2f})")
        st.pyplot(fig)

    elif explanation_method == "Aggregate LIME":
        if not aggregate_lime_available:
            st.warning("⚠️ Aggregate LIME is not supported for MiniRocket models.")
            return

        st.info("Computing class-specific aggregate LIME explanation...")
        selected_class = st.selectbox("Select Class for Aggregation", options=st.session_state["y_labels"])
        num_frames = st.number_input("Number of Frames to Aggregate", 10, min(200, len(X_test)), 50, step=10)

        feature_names = [f"{res}-{inter}" for res, inter in df.columns if res != "meta"]
        model_state_dict = torch.load(st.session_state["model_path"], map_location=device)

        def predict_fn_lime(inputs):
            return cached_predict_fn(
                _model_state_dict=model_state_dict,
                input_channels=1,
                num_classes=len(st.session_state["y_labels"]),
                device=device,
                inputs=inputs
            )

        explainer = get_lime_explainer(X_test, st.session_state["y_labels"], feature_names)
        importance, matched = {}, 0
        for i in range(len(X_test)):
            if matched >= num_frames:
                break
            if st.session_state["results"].iloc[i]["Predicted"] != selected_class:
                continue
            matched += 1
            explanation = explainer.explain_instance(
                data_row=X_test[i],
                predict_fn=predict_fn_lime,
                num_features=20,
                top_labels=1,
                num_samples=500
            )
            for feat, weight in explanation.as_list(label=explanation.available_labels()[0]):
                importance[feat] = importance.get(feat, 0) + abs(weight)

        if matched == 0:
            st.warning(f"No frames found with predicted class '{selected_class}'.")
        else:
            sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:20]
            feat_names, weights = zip(*sorted_importance)
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.barh(feat_names[::-1], weights[::-1], color='darkorange')
            ax.set_xlabel(f"Aggregate LIME Importance for '{selected_class}'")
            ax.set_title(f"Top Features Across {matched} '{selected_class}' Frames")
            st.pyplot(fig)


# ========================
# Main Workflow
# ========================
uploaded_file = st.file_uploader("📂 Upload a cleaned or raw CSV file", type=["csv"])
if uploaded_file:
    file_name = uploaded_file.name
    raw_path = os.path.join(test_save_dir, file_name)
    with open(raw_path, "wb") as f:
        f.write(uploaded_file.read())
    st.write(f"📄 Uploaded: `{file_name}`")

    try:
        df = pd.read_csv(raw_path, header=[0, 1])
        cleaned_path = raw_path if ("meta", "frame") in df.columns else None
    except Exception:
        cleaned_path = None

    if cleaned_path is None and st.button("Clean Uploaded File"):
        cleaned_path = preprocess_input_csv(raw_path, test_save_dir)
        df = pd.read_csv(cleaned_path, header=[0, 1])
        st.success(" File cleaned and saved.")
        st.write("Preview of cleaned data:", df.head())

    if "results" not in st.session_state:
        st.session_state["results"] = None

    if cleaned_path:
        st.markdown("---")
        with st.expander("📌 Select Model and Predict"):
            model_options = {
                "ExpandedResNet1D (v2)": "models/v2/v2_model.pt",
                "Experimental model": "models/v2/resnet1d_final.pt",
                "MiniRocket + LogisticRegression": "models/v2/minirocket_logistic.joblib"
            }
            model_choice = st.selectbox("Available Models", list(model_options.keys()))
            model_path = model_options[model_choice]

            if st.button("🚀 Predict Bond Type"):
                try:
                    y_labels = np.load("models/transfer/y_labels.npy", allow_pickle=True)
                except FileNotFoundError:
                    st.error("❌ y_labels.npy not found in 'models/transfer'")
                else:
                    if model_choice == "MiniRocket + LogisticRegression":
                        results = evaluate_with_minirocket(cleaned_path, "models/v2/minirocket_transformer.joblib", model_path, y_labels)
                    else:
                        results, _ = evaluate_on_new_csv(cleaned_path, model_path, y_labels, device, model_class=ExpandedResNet1D)

                    st.session_state.update({
                        "results": results,
                        "y_labels": y_labels,
                        "model_path": model_path,
                        "cleaned_path": cleaned_path,
                        "model_choice": model_choice
                    })
                    st.success("✅ Prediction Complete")

    if st.session_state["results"] is not None:
        st.subheader("📊 Sample Predictions")
        sample_df = st.session_state["results"][["Predicted"]].copy()
        color_map = {"occluded": "color: #f9a825", "outward": "color: #d32f2f", "inward": "color: #388e3c"}
        styled_df = sample_df.style.applymap(lambda val: color_map.get(val, ""))
        st.dataframe(styled_df, height=400)

        st.subheader("📈 Overall Stats")
        counts = st.session_state["results"]["Predicted"].value_counts()
        for label in st.session_state["y_labels"]:
            st.write(f"{label}: {counts.get(label, 0)} frames")

        st.markdown("---")
        tab1, tab2 = st.tabs(["🔬 Tanimoto Comparison", "🧪 Explanation Methods"])
        with tab1:
            show_tanimoto_tab()
        with tab2:
            show_explanation_tab()
