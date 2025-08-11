# mini_rocket_v5_app.py
import streamlit as st
import pandas as pd
import numpy as np
import torch
import joblib
from pathlib import Path

# ------------------ SETTINGS ------------------
MODEL_DIR = Path("D:/Coding Projects/AI-in-Drug-Discovery-/models/v5")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PAD_LEN = 9  # as used in training (MiniRocketMultivariate needs >=9)

# ------------------ LOAD MODEL ------------------
@st.cache_resource
def load_minirocket_mlp_v5(model_dir: Path, device):
    rocket = joblib.load(model_dir / "minirocket_transformer.pkl")

    class RocketMLP(torch.nn.Module):
        def __init__(self, in_dim, n_classes):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Linear(in_dim, 512),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.3),
                torch.nn.Linear(512, 256),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.3),
                torch.nn.Linear(256, n_classes),
            )
        def forward(self, x):
            return self.net(x)

    label_names = np.load(model_dir / "label_encoder_classes.npy", allow_pickle=True)
    input_dim = int(np.load(model_dir / "minirocket_input_dim.npy")[0])

    model = RocketMLP(in_dim=input_dim, n_classes=len(label_names))
    model.load_state_dict(torch.load(model_dir / "rocket_mlp_weights.pt", map_location=device))
    model.to(device).eval()
    return rocket, model, label_names, input_dim

rocket, model, label_names, INPUT_DIM = load_minirocket_mlp_v5(MODEL_DIR, DEVICE)

# ------------------ PREPROCESS (mirror training) ------------------
def clean_and_shape(df_raw: pd.DataFrame):
    """
    1) Rename columns: first->frame_idx, last-2->bond_type, last-1->drug_name
    2) Drop the mini-header row if present
    3) Extract fingerprint columns (exclude frame_idx, bond_type, drug_name)
    4) Cast to float32
    5) Build (N, C, 1) then pad to (N, C, 9)
    """
    df = df_raw.copy()
    cols = list(df.columns)
    if len(cols) >= 3:
        cols[0]  = "frame_idx"
        cols[-2] = "bond_type"
        cols[-1] = "drug_name"
        df.columns = cols

    # Drop accidental header row (exactly how you saved them)
    if "bond_type" in df.columns and "drug_name" in df.columns:
        try:
            if (df.loc[0, "bond_type"] == "bond_type") or (df.loc[0, "drug_name"] == "drug_name"):
                df = df.iloc[1:].reset_index(drop=True)
        except Exception:
            pass

    ignore = {"frame_idx", "bond_type", "drug_name"}
    fp_cols = [c for c in df.columns if c not in ignore]

    fp_df = df[fp_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).astype("float32")
    X = fp_df.to_numpy(dtype=np.float32)[..., None]  # (N, C, 1)
    # Pad time axis to 9
    pad_width = ((0, 0), (0, 0), (0, max(0, PAD_LEN - X.shape[2])))
    X = np.pad(X, pad_width, mode="constant")

    return df, X

# ------------------ INFERENCE ------------------
def predict_framewise(df_in: pd.DataFrame):
    df_clean, X_fp = clean_and_shape(df_in)
    X_tf = rocket.transform(X_fp)              # (N, 9996)
    X_tf = np.asarray(X_tf, dtype=np.float32)
    with torch.no_grad():
        logits = model(torch.tensor(X_tf, dtype=torch.float32, device=DEVICE))
        probs = torch.softmax(logits, dim=1).cpu().numpy()  # (N, num_classes)
        preds = probs.argmax(axis=1)                        # (N,)
    return df_clean, probs, preds

# ------------------ CHUNK AGGREGATION ------------------
def chunk_summary(df_clean: pd.DataFrame, probs: np.ndarray, preds: np.ndarray, chunk_size: int):
    """
    Returns a DataFrame with columns:
    chunk_id, start_frame, end_frame, max_conf, winner_class, true_class, pred_class
    """
    n = len(preds)
    n_chunks = int(np.ceil(n / chunk_size))
    rows = []

    # Prepare optional ground truth (majority vote per chunk)
    has_truth = "bond_type" in df_clean.columns
    frames = df_clean["frame_idx"] if "frame_idx" in df_clean.columns else pd.Series(np.arange(n))

    for i in range(n_chunks):
        s = i * chunk_size
        e = min((i + 1) * chunk_size, n)
        chunk_probs = probs[s:e]                         # (L, num_classes)
        mean_probs = chunk_probs.mean(axis=0)            # (num_classes,)
        winner_idx = int(mean_probs.argmax())
        max_conf = float(mean_probs[winner_idx])

        # majority predicted label inside the chunk (string)
        # (alternatively, use label_names[winner_idx] to mirror mean-prob winner)
        pred_label = label_names[winner_idx]

        # optional ground truth majority
        if has_truth:
            true_label = df_clean.loc[s:e-1, "bond_type"].mode()
            true_label = true_label.iloc[0] if not true_label.empty else ""
        else:
            true_label = ""

        start_frame = int(frames.iloc[s])
        end_frame   = int(frames.iloc[e - 1])

        rows.append({
            "chunk_id":      i + 1,               # 1-based
            "start_frame":   start_frame,
            "end_frame":     end_frame,
            "max_conf":      max_conf,
            "winner_class":  winner_idx,          # numeric index in label_names
            "true_class":    str(true_label),
            "pred_class":    str(pred_label),
        })

    return pd.DataFrame(rows, columns=[
        "chunk_id","start_frame","end_frame","max_conf","winner_class","true_class","pred_class"
    ])

# ------------------ UI ------------------
st.title("MiniRocketMLP_v5 – Chunked Summary Export")
st.write("Upload a `cleaned_*.csv`, choose a chunk size, and download a compact summary CSV.")

uploaded = st.file_uploader("📂 Choose CSV", type=["csv"])
chunk_size = st.slider("Frames per chunk", min_value=5, max_value=100, value=20, step=5)

if uploaded:
    df_in = pd.read_csv(uploaded)
    st.write("Preview:", df_in.head())

    df_clean, probs, preds = predict_framewise(df_in)
    summary = chunk_summary(df_clean, probs, preds, chunk_size)

    st.subheader("Chunked Summary (preview)")
    st.dataframe(summary.head(12))

    st.download_button(
        "💾 Download chunked summary CSV",
        summary.to_csv(index=False).encode(),
        file_name=f"{Path(uploaded.name).stem}_chunked_{chunk_size}.csv",
        mime="text/csv",
    )
