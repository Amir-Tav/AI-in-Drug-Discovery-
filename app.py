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
import plotly.graph_objects as go
from matplotlib.collections import LineCollection
import matplotlib.colors as mcolors
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns



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
    evaluate_chunked_confidence,
    evaluate_minirocket_mlp,
    evaluate_minirocket_mlp_v2,
    evaluate_minirocket_mlp_v5,
    chunk_confidence_to_df
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
    """Tab that compares chunk‑level confidence between two datasets."""
    st.subheader("🔬 Tanimoto Comparison: Confidence Over Chunks")

    # ── Upload two CSVs ─────────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        file_1 = st.file_uploader("📂 Dataset A", type=["csv"], key="tani_a")
    with col2:
        file_2 = st.file_uploader("📂 Dataset B", type=["csv"], key="tani_b")

    if file_1 and file_2:
        # helper to persist uploaded CSVs
        def save_upload(uploaded_file):
            path = os.path.join(tani_save_dir, uploaded_file.name)
            with open(path, "wb") as f:
                f.write(uploaded_file.read())
            return path

        path1, path2 = save_upload(file_1), save_upload(file_2)

        # ── Choose model ───────────────────────────────────────────
        st.markdown("### ⚙️ Select Model")
        model_options = {
            "ExpandedResNet1D (v2)":           "models/v2/v2_model.pt",
            "Experimental model":              "models/v2/resnet1d_final.pt",
            "MiniRocket + LogisticRegression": "models/v2/minirocket_logistic.joblib",
            "MiniRocket + MLP":                "models/v3/Mini-RocketMLP.pt",
            "MiniRocketMLP_V2 (fold‑5)":       "models/v4/MiniRocketMLP_fold5.pt",
            "MiniRocketMLP_V5":                "models/v5/rocket_mlp_weights.pt",   # ← NEW
        }
        model_choice = st.selectbox("Available Models",
                                    list(model_options.keys()),
                                    key="tani_model")
        model_path = model_options[model_choice]

        # ── Chunk size slider ──────────────────────────────────────
        chunk_size = st.slider("🔢 Frames per chunk", 5, 100, 20, step=5)

        if st.button("📊 Compare Chunked Confidence",
                     key="run_tani_comparison"):
            try:
                y_labels = np.load("models/transfer/y_labels.npy",
                                   allow_pickle=True)
            except FileNotFoundError:
                st.error("❌ y_labels.npy not found in 'models/transfer'")
                return

            df1 = pd.read_csv(path1, header=[0, 1])
            df2 = pd.read_csv(path2, header=[0, 1])

            conf1, winners1 = evaluate_chunked_confidence(
                df=df1,
                model_path=model_path,
                model_choice=model_choice,
                y_labels=y_labels,
                device=device,
                chunk_size=chunk_size,
            )
            conf2, winners2 = evaluate_chunked_confidence(
                df=df2,
                model_path=model_path,
                model_choice=model_choice,
                y_labels=y_labels,
                device=device,
                chunk_size=chunk_size,
            )

            # ── Offer CSV downloads ────────────────────────────────
            df_conf1 = chunk_confidence_to_df(conf1, winners1, chunk_size)
            df_conf2 = chunk_confidence_to_df(conf2, winners2, chunk_size)

            col_dl1, col_dl2 = st.columns(2)
            with col_dl1:
                st.download_button(
                    "💾 Download A confidence CSV",
                    df_conf1.to_csv(index=False).encode(),
                    file_name=f"{Path(path1).stem}_confidence.csv",
                    mime="text/csv",
                )
            with col_dl2:
                st.download_button(
                    "💾 Download B confidence CSV",
                    df_conf2.to_csv(index=False).encode(),
                    file_name=f"{Path(path2).stem}_confidence.csv",
                    mime="text/csv",
                )

            # ── Colour map per class for plotting ──────────────────
            class_colors = {
                y_labels[0]: "tab:green",
                y_labels[1]: "tab:orange",
                y_labels[2]: "tab:red",
            }

            def multicolour_line(x, y, cls_idx, label_prefix):
                points = np.array([x, y]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                seg_cols = [class_colors[y_labels[i]] for i in cls_idx[:-1]]
                lc = LineCollection(segments, colors=seg_cols, linewidths=2)
                ax.add_collection(lc)
                ax.plot([], [], color=seg_cols[0], lw=2, label=label_prefix)

            # ── Plot ───────────────────────────────────────────────
            fig, ax = plt.subplots(figsize=(11, 6))
            x1 = np.arange(1, len(conf1) + 1)
            x2 = np.arange(1, len(conf2) + 1)

            label1 = Path(path1).stem
            label2 = Path(path2).stem

            multicolour_line(x1, conf1, winners1, label1)
            multicolour_line(x2, conf2, winners2, label2)

            # midpoint tags
            ax.text(x1[len(x1)//2], conf1[len(x1)//2]-0.05, label1,
                    ha='center', va='top', fontsize=10, fontweight='bold')
            ax.text(x2[len(x2)//2], conf2[len(x2)//2]-0.05, label2,
                    ha='center', va='top', fontsize=10, fontweight='bold')

            ax.set_xlabel(f"Chunk # (every {chunk_size} frames)")
            ax.set_ylabel("Max‑class Confidence")
            ax.set_ylim(0, 1)
            ax.set_title("Confidence Line Coloured by Predicted Class")

            handles = [plt.Line2D([0], [0], color=c, lw=4)
                       for c in class_colors.values()]
            ax.legend(handles, class_colors.keys(),
                      title="Predicted class",
                      bbox_to_anchor=(1.02, 1), loc="upper left")
            ax.grid(alpha=0.3)
            st.pyplot(fig)

def show_diagnostics_tab():
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix, classification_report
    from sklearn.calibration import calibration_curve
    from utils import evaluate_chunked_confidence, chunk_confidence_to_df
    from matplotlib.collections import LineCollection
    from pathlib import Path

    st.subheader("🩺 Diagnostics: True vs Predicted Chunk Labels")

    # ─────────── Upload CSV ───────────
    file_ = st.file_uploader(
        "📂 Upload labelled CSV (must contain 'bond_type' column)",
        type=["csv"],
        key="diag_csv",
    )
    if not file_:
        return

    csv_path = os.path.join(tani_save_dir, file_.name)
    with open(csv_path, "wb") as f:
        f.write(file_.read())

    # ─────────── Model selector ───────
    st.markdown("### ⚙️ Select Model")
    model_options = {
        "ExpandedResNet1D (v2)":           "models/v2/v2_model.pt",
        "Experimental model":              "models/v2/resnet1d_final.pt",
        "MiniRocket + LogisticRegression": "models/v2/minirocket_logistic.joblib",
        "MiniRocket + MLP":                "models/v3/Mini-RocketMLP.pt",
        "MiniRocketMLP_V2 (fold‑5)":       "models/v4/MiniRocketMLP_fold5.pt",
        "MiniRocketMLP_V5":                "models/v5/rocket_mlp_weights.pt",  # ← NEW
    }
    model_choice = st.selectbox("Available Models",
                                list(model_options.keys()), key="diag_model")
    model_path = model_options[model_choice]

    # Chunk slider 0‑100 (0 → 1)
    raw_chunk = st.slider("🔢 Frames per chunk", 0, 100, 20, 5)
    chunk_size = max(1, raw_chunk)

    if st.button("🔍 Run diagnostics"):
        try:
            y_labels = np.load("models/transfer/y_labels.npy",
                               allow_pickle=True).tolist()
        except FileNotFoundError:
            st.error("❌ y_labels.npy not found"); return

        # helper: build timeline segments
        def build_bar_segments(winners, chunk, color_map):
            segments = []
            if not winners: return segments
            last, start = winners[0], 0
            for i, cls in enumerate(winners, 1):
                if cls != last:
                    segments.append(((start*chunk, (i-start)*chunk),
                                     {'facecolor': color_map[y_labels[last]]}))
                    start, last = i, cls
            segments.append(((start*chunk, (len(winners)-start)*chunk),
                             {'facecolor': color_map[y_labels[last]]}))
            return segments

        # Load CSV
        df = pd.read_csv(csv_path, header=[0, 1])
        if ("meta", "bond_type") not in df.columns:
            st.error("❌ Column ('meta','bond_type') not found in CSV"); return

        # Truth winners (majority vote)
        truth_idx = [y_labels.index(lbl) for lbl in df[("meta", "bond_type")]]
        true_winners = [
            max(set(truth_idx[i:i+chunk_size]),
                key=truth_idx[i:i+chunk_size].count)
            for i in range(0, len(truth_idx), chunk_size)
        ]

        # Predictions + confidences
        confidences, pred_winners = evaluate_chunked_confidence(
            df=df, model_path=model_path, model_choice=model_choice,
            y_labels=y_labels, device=device, chunk_size=chunk_size
        )

        n = min(len(true_winners), len(pred_winners))
        true_winners, pred_winners, confidences = (
            true_winners[:n], pred_winners[:n], confidences[:n])

        # colour palette
        class_colors = {y_labels[0]: "tab:green",
                        y_labels[1]: "tab:orange",
                        y_labels[2]: "tab:red"}

        # timeline plot
        true_seg = build_bar_segments(true_winners,  chunk_size, class_colors)
        pred_seg = build_bar_segments(pred_winners, chunk_size, class_colors)
        fig, ax = plt.subplots(figsize=(12,3))
        for xr, stl in true_seg: ax.broken_barh([xr], (1.5,0.8), **stl)
        for xr, stl in pred_seg: ax.broken_barh([xr], (0.2,0.8), **stl)
        ax.set_yticks([1.9,0.6]); ax.set_yticklabels(["True","Pred"])
        ax.set_xlabel(f"Frame # (chunk = {chunk_size} frame"
                      f"{'' if chunk_size==1 else 's'})")
        ax.set_xlim(0, len(true_winners)*chunk_size); ax.set_ylim(0,2.5)
        ax.set_title("Chunk‑level True vs Predicted Timeline"); ax.grid(axis="x", alpha=.3)
        ax.legend([plt.Line2D([0],[0], color=c, lw=8) for c in class_colors.values()],
                  class_colors.keys(), title="Class colour", loc="upper right")
        st.pyplot(fig)

        # 1) accuracy bar
        with st.expander("accuracy bar"):
            acc = (np.array(true_winners)==np.array(pred_winners)).astype(int)
            fig2, ax2 = plt.subplots(figsize=(12,1.3))
            ax2.bar(range(1,len(acc)+1), acc,
                    color=['tab:green' if a else 'tab:red' for a in acc])
            ax2.set_ylim(0,1.1); ax2.set_yticks([0,1]); ax2.set_title("Chunk accuracy")
            st.pyplot(fig2)

        # 2) confusion matrix
        with st.expander("📊 Confusion Matrix"):
            cm = confusion_matrix(true_winners, pred_winners,
                                  labels=range(len(y_labels)))
            fig_cm, ax_cm = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                        xticklabels=y_labels, yticklabels=y_labels, ax=ax_cm)
            ax_cm.set_xlabel("Predicted"); ax_cm.set_ylabel("True")
            st.pyplot(fig_cm)

        # 3) per‑class metrics
        with st.expander("per-class metrics"):
            unique_idx = sorted(set(true_winners) | set(pred_winners))
            unique_names = [y_labels[i] for i in unique_idx]
            st.text(classification_report(true_winners,
                                          pred_winners,
                                          labels=unique_idx,
                                          target_names=unique_names))

        # 4) calibration curve
        with st.expander("calibration curve"):
            prob_true, prob_pred = calibration_curve(
                (np.array(true_winners)==np.array(pred_winners)).astype(int),
                confidences, n_bins=10)
            fig_cal, ax_cal = plt.subplots()
            ax_cal.plot(prob_pred, prob_true, marker='o'); ax_cal.plot([0,1],[0,1],'--')
            ax_cal.set_title("Calibration curve"); st.pyplot(fig_cal)

        # 5) wrong‑confidence histogram
        with st.expander("wrong-confidence histogram"):
            wrong_conf = [c for c,t,p in zip(confidences,true_winners,pred_winners) if t!=p]
            fig_hist, ax_hist = plt.subplots()
            ax_hist.hist(wrong_conf, bins=20, color='tab:red')
            ax_hist.set_title("Confidence distribution (wrong chunks)")
            st.pyplot(fig_hist)

        # 6) interactive table
        with st.expander("summary table"):
            df_diag = chunk_confidence_to_df(confidences, pred_winners, chunk_size)
            df_diag["true_class"] = [y_labels[i] for i in true_winners]
            df_diag["pred_class"] = [y_labels[i] for i in pred_winners]
            st.dataframe(df_diag)


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
                "MiniRocket + LogisticRegression": "models/v2/minirocket_logistic.joblib",
                "MiniRocket + MLP": "models/v3/Mini-RocketMLP.pt",
                "MiniRocketMLP_V2 (fold‑5)": "models/v4/MiniRocketMLP_fold5.pt",
                "MiniRocketMLP_V5": "models/v5/rocket_mlp_weights.pt"
            }
            model_choice = st.selectbox("Available Models", list(model_options.keys()))
            model_path = model_options[model_choice]

## Model types
            if st.button("🚀 Predict Bond Type"):
                try:
                    y_labels = np.load("models/transfer/y_labels.npy", allow_pickle=True)
                except FileNotFoundError:
                    st.error("❌ y_labels.npy not found in 'models/transfer'")
                else:
                    # ------------------------------------------------------------------
                    if model_choice == "MiniRocket + LogisticRegression":
                        results = evaluate_with_minirocket(
                            cleaned_path,
                            "models/v2/minirocket_transformer.joblib",
                            model_path,
                            y_labels,
                        )

                    elif model_choice == "MiniRocket + MLP":
                        results = evaluate_minirocket_mlp(
                            cleaned_path,
                            "models/v3/minirocket_transformer.joblib",
                            model_path,
                            y_labels,
                            device,
                        )

                    elif model_choice == "MiniRocketMLP_V2 (fold‑5)":
                        results = evaluate_minirocket_mlp_v2(
                            cleaned_path,
                            "models/v4/rocket_fold5.joblib",
                            model_path,
                            y_labels,
                            device,
                        )

                    elif model_choice == "MiniRocketMLP_V5":
                        # ---- new branch ------------------------------------------------
                        preds, probs, acc = evaluate_minirocket_mlp_v5(
                            cleaned_path, device=device
                        )
                        # Build a results DataFrame to match other branches
                        df_tmp = pd.read_csv(cleaned_path, header=[0, 1])
                        frames = df_tmp[("meta", "frame")] if ("meta", "frame") in df_tmp.columns else np.arange(len(preds))
                        results = pd.DataFrame({
                            "Frame": frames,
                            "Predicted": preds,
                            "Confidence": probs.max(1),
                        })
                        if acc is not None:
                            st.info(f"Validation accuracy on this file: **{acc:.3f}**")

                    else:  # ExpandedResNet1D or other CNN baselines
                        results, _ = evaluate_on_new_csv(
                            cleaned_path,
                            model_path,
                            y_labels,
                            device,
                            model_class=ExpandedResNet1D,
                        )

                    # ------------------------------------------------------------------
                    st.session_state.update(
                        {
                            "results": results,
                            "y_labels": y_labels,
                            "model_path": model_path,
                            "cleaned_path": cleaned_path,
                            "model_choice": model_choice,
                        }
                    )
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
        tab1, tab2, tab3 = st.tabs(["🔬 Tanimoto Comparison", "Diagnostics" ,"🧪 Explanation Methods"])
        with tab1:
            show_tanimoto_tab()
        with tab2:
            show_diagnostics_tab()
        with tab3:
            show_explanation_tab()


