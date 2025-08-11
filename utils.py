import os
import lime
import torch
import joblib
import shap
from pathlib import Path
import pandas as pd
import numpy as np
import torch.nn as nn
import streamlit.components.v1 as components
from lime.lime_tabular import LimeTabularExplainer
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

# ============================
# 1. Dataset Class
# ============================
class FingerprintDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx].unsqueeze(0), self.y[idx]

# ============================
# 2. ResNet1D Model
# ============================
class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, downsample=None, dropout=0.2):
        super(ResidualBlock1D, self).__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out = out + identity
        return self.relu(out)

class ExpandedResNet1D(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(ExpandedResNet1D, self).__init__()
        
        self.initial_conv = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )
        
        self.layer1 = self._make_layer(64, 64, blocks=3)
        self.layer2 = self._make_layer(64, 128, blocks=4, stride=2)
        self.layer3 = self._make_layer(128, 256, blocks=6, stride=2)
        self.layer4 = self._make_layer(256, 512, blocks=3, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(512, num_classes)
        
    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )
        layers = []
        layers.append(ResidualBlock1D(in_channels, out_channels, stride=stride, downsample=downsample))
        for _ in range(1, blocks):
            layers.append(ResidualBlock1D(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.initial_conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        return self.fc(x)

# ============================
# 3. Data Cleaning for Input (No labels required)
# ============================
def preprocess_input_csv(file_path: str, output_dir: str):
    import pandas as pd
    import os
    os.makedirs(output_dir, exist_ok=True)

    # Load original CSV
    df = pd.read_csv(file_path)

    # Drop third row (index 2) with mostly NaNs
    df_cleaned = df.drop(index=2).reset_index(drop=True)

    # Extract header rows for multi-index
    header_residues = df_cleaned.iloc[0]
    header_types = df_cleaned.iloc[1]

    multi_index = pd.MultiIndex.from_arrays([header_residues, header_types])

    # Drop header rows
    df_cleaned = df_cleaned.drop(index=[0, 1]).reset_index(drop=True)

    # Apply multi-index columns
    df_cleaned.columns = multi_index

    # Insert frame column
    frame_col = df.iloc[3:, 0].reset_index(drop=True)
    df_cleaned.insert(0, ("meta", "frame"), frame_col.astype(int))

    # Drop redundant columns if present
    if ("protein", "interaction") in df_cleaned.columns:
        df_cleaned = df_cleaned.drop(columns=[("protein", "interaction")])

    # Convert numeric and fill NaNs
    df_cleaned = df_cleaned.apply(pd.to_numeric, errors='ignore').fillna(0)

    # Save cleaned CSV
    base_name = os.path.basename(file_path)
    output_path = os.path.join(output_dir, f"cleaned_{base_name}")
    df_cleaned.to_csv(output_path, index=False)

    print(f"✅ Saved cleaned CSV: {output_path}")
    return output_path

# ============================
# 4. Model Evaluation
# ============================
def evaluate_on_new_csv(csv_path, model_path, y_labels, device, model_class=ExpandedResNet1D):
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder
    from torch.utils.data import DataLoader
    import torch.nn.functional as F

    df = pd.read_csv(csv_path, header=[0, 1])
    X = df.loc[:, df.columns.get_level_values(0) != "meta"].to_numpy(dtype=float)

    dummy_labels = np.zeros(len(X))  # just for dataset compatibility
    dataset = FingerprintDataset(X, dummy_labels)
    loader = DataLoader(dataset, batch_size=32)

    model = model_class(input_channels=1, num_classes=len(y_labels))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    all_preds = []

    with torch.no_grad():
        for X_batch, _ in loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            probs = F.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1).cpu().numpy()
            all_preds.extend(preds)

    result_df = pd.DataFrame({
        "Predicted": [y_labels[i] for i in all_preds]
    })
    print("\nSample Predictions:")
    print(result_df.head(10))

    return result_df, None


# ============================
# 5. LIME Explanation
# ============================
def explain_with_lime(model, X_test, y_labels, feature_names, device, frame_index=20):
    model.eval()
    explainer = LimeTabularExplainer(
        training_data=X_test,
        mode="classification",
        class_names=y_labels.tolist(),
        feature_names=feature_names,
        discretize_continuous=False
    )

    def predict_fn_lime(inputs):
        inputs_tensor = torch.tensor(inputs[:, np.newaxis, :], dtype=torch.float32).to(device)
        outputs = model(inputs_tensor)
        probs = F.softmax(outputs, dim=1).detach().cpu().numpy()
        return probs

    explanation = explainer.explain_instance(
        data_row=X_test[frame_index],
        predict_fn=predict_fn_lime,
        num_features=15,                                        #out display
        top_labels=1
    )

    lime_html = explanation.as_html(show_table=True)

    return lime_html

# ============================
# 6. SHAP Explanation
# ============================
def explain_with_shap(model, X_test, y_labels, feature_names, device, frame_index=20):
    model.eval()
    background = torch.tensor(X_test[:200]).unsqueeze(1).float().to(device)
    test_sample = torch.tensor(X_test[frame_index:frame_index+1]).unsqueeze(1).float().to(device)

    explainer = shap.DeepExplainer(model, background)

    with torch.no_grad():
        output = model(test_sample)
        probs = torch.softmax(output, dim=1).cpu().numpy()[0]
        pred_class = np.argmax(probs)
        pred_label = y_labels[pred_class]
        confidence = probs[pred_class]

    shap_values = explainer.shap_values(test_sample, check_additivity=False)
    shap_vector = shap_values[0][0, :, pred_class]
    base_value = explainer.expected_value[pred_class]

    explanation = shap.Explanation(
        values=shap_vector,
        base_values=base_value,
        data=X_test[frame_index],
        feature_names=feature_names
    )

    fig = plt.figure(figsize=(10, 6))
    shap.plots.waterfall(explanation, max_display=15, show=False)               #out display 
    plt.tight_layout()

    return fig, pred_label, confidence


#######################################         Mini ROcket + logistic           ################################################## 
# ============================
# 7. MiniRocket Evaluation
# ============================
def evaluate_with_minirocket(csv_path, rocket_path, clf_path, y_labels, output_dir="results"):
    import pandas as pd
    import os

    os.makedirs(output_dir, exist_ok=True)

    # Load transformer & model
    rocket = joblib.load(rocket_path)
    clf = joblib.load(clf_path)

    # Load CSV (assumes cleaned format with multi-index)
    df = pd.read_csv(csv_path, header=[0, 1])
    X = df.loc[:, df.columns.get_level_values(0) != "meta"].to_numpy(dtype=np.float32)
    
    # Reshape for MiniRocket
    X = X.reshape(X.shape[0], 1, X.shape[1])
    
    # Transform
    X_tf = rocket.transform(X)
    
    # Predict
    y_pred = clf.predict(X_tf)
    y_pred_labels = [y_labels[i] for i in y_pred]

    # Return results
    result_df = pd.DataFrame({
        "Frame": df[("meta", "frame")].values,
        "Predicted": y_pred_labels
    })
    
    out_path = os.path.join(output_dir, "minirocket_predictions.csv")
    result_df.to_csv(out_path, index=False)
    print(f"✅ Predictions saved to: {out_path}")
    
    return result_df

# ============================
# 8. LIME for MiniRocket + LogisticRegression
# ============================
def explain_minirocket_lime(clf, X_train_tf, X_test_tf, y_labels, frame_index=0, feature_names=None):

    X_train_tf = np.array(X_train_tf)
    X_test_tf = np.array(X_test_tf)

    if feature_names is None:
        feature_names = [f"MRF{i}" for i in range(X_train_tf.shape[1])]


    explainer = LimeTabularExplainer(
        training_data=X_train_tf,
        feature_names=feature_names,
        class_names=y_labels.tolist(),
        mode="classification",
        discretize_continuous=True
    )

    explanation = explainer.explain_instance(
        data_row=X_test_tf[frame_index],
        predict_fn=clf.predict_proba,
        num_features=15
    )

    return explanation.as_html(show_table=True)


# ============================
# 9. SHAP for MiniRocket + LogisticRegression
# ============================
def explain_minirocket_shap(clf, X_train_tf, X_test_tf, y_labels, frame_index=0, feature_names=None):

    X_train_tf = np.array(X_train_tf)
    X_test_tf = np.array(X_test_tf)

    prediction = clf.predict_proba(X_test_tf[frame_index:frame_index+1])[0]
    pred_class = np.argmax(prediction)
    pred_label = y_labels[pred_class]
    confidence = prediction[pred_class]

    explainer = shap.Explainer(clf, X_train_tf)
    shap_values = explainer(X_test_tf[frame_index:frame_index+1])

    if feature_names is None:
        feature_names = [f"MRF{i}" for i in range(X_test_tf.shape[1])]


    explanation = shap.Explanation(
        values=shap_values.values[0][:, pred_class],
        base_values=shap_values.base_values[0][pred_class],
        data=shap_values.data[0],
        feature_names=feature_names
    )

    fig = plt.figure(figsize=(10, 6))
    shap.plots.waterfall(explanation, max_display=15, show=False)
    plt.tight_layout()

    return fig, pred_label, confidence


###############################################       MiniRocket with MLP      ##############################################

# ============================
# MiniRocketMLP Model
# ============================
class MiniRocketMLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(MiniRocketMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.3)
        self.out = nn.Linear(64, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        return self.out(x)
    

class MiniRocketMLP_V2(nn.Module):
    """Fold‑5 MLP: two Linear layers + Dropout, no BatchNorm."""
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.3)
        self.out = nn.Linear(64, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        return self.out(x)


# Fold‑5 MiniRocket + MLP‑V2 evaluator
def evaluate_minirocket_mlp_v2(csv_path, rocket_path, model_path, y_labels, device):
    """
    Inference pipeline for the BatchNorm + SiLU MLP trained in fold‑5.
    """
    # 1. Load and reshape CSV --------------------------------------------------
    df = pd.read_csv(csv_path, header=[0, 1])
    X  = df.loc[:, df.columns.get_level_values(0) != "meta"] \
            .to_numpy(dtype=np.float32)
    X  = X.reshape(X.shape[0], 1, X.shape[1])  # (samples, 1, 87)

    # 2. Transform with fold‑5 MiniRocket -------------------------------------
    rocket = joblib.load(rocket_path)
    X_tf   = np.asarray(rocket.transform(X), dtype=np.float32)

    # 3. Load fold‑5 MLP‑V2 weights -------------------------------------------
    model = MiniRocketMLP_V2(input_dim=X_tf.shape[1],
                             num_classes=len(y_labels))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()

    # 4. Predict --------------------------------------------------------------
    with torch.no_grad():
        logits = model(torch.tensor(X_tf).to(device))
        preds  = torch.argmax(F.softmax(logits, dim=1), dim=1).cpu().numpy()

    # 5. Return as DataFrame ---------------------------------------------------
    return pd.DataFrame({
        "Frame": df[("meta", "frame")].values,
        "Predicted": [y_labels[i] for i in preds]
    })


# ============================
#   Evaluation for MiniRocket + MLP
# ============================
def evaluate_minirocket_mlp(csv_path, rocket_path, model_path, y_labels, device):
    # Load and prepare the CSV
    df = pd.read_csv(csv_path, header=[0, 1])
    X = df.loc[:, df.columns.get_level_values(0) != "meta"].to_numpy(dtype=np.float32)
    X = X.reshape(X.shape[0], 1, X.shape[1])  # shape = (samples, 1, 87)

    # Load MiniRocket transformer and transform
    rocket = joblib.load(rocket_path)
    X_tf = rocket.transform(X)
    X_tf = np.asarray(X_tf)  # make sure it's a proper ndarray

    # Load model
    model = MiniRocketMLP(input_dim=X_tf.shape[1], num_classes=len(y_labels))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Predict
    with torch.no_grad():
        X_tensor = torch.tensor(X_tf, dtype=torch.float32).to(device)
        logits = model(X_tensor)
        preds = torch.argmax(F.softmax(logits, dim=1), dim=1).cpu().numpy()

    # Return predictions in a DataFrame
    result_df = pd.DataFrame({
        "Frame": df[("meta", "frame")].values,
        "Predicted": [y_labels[i] for i in preds]
    })

    return result_df

################################################        Tanimoto-Style      ###################################################

# ============================
# 10. Chunked Confidence Evaluation (Tanimoto-style)
# ============================

def evaluate_chunked_confidence(
    df: pd.DataFrame,
    model_path: str,
    model_choice: str,
    y_labels: list,
    device,
    chunk_size: int = 20
):
    """
    Aggregates class probabilities over successive frame chunks
    and returns the max‑class confidence per chunk.
    """
    X = df.loc[:, df.columns.get_level_values(0) != "meta"] \
          .to_numpy(dtype=np.float32)
    num_classes = len(y_labels)
    num_chunks  = len(X) // chunk_size
    confidences = []
    win_classes = [] 

    # ------------------------------------------------------------------
    # Model‑specific frame‑wise probabilities
    # ------------------------------------------------------------------
    if model_choice == "MiniRocket + LogisticRegression":
        rocket = joblib.load("models/v2/minirocket_transformer.joblib")
        clf    = joblib.load(model_path)
        X_tf   = rocket.transform(X.reshape(X.shape[0], 1, X.shape[1]))
        X_tf   = X_tf.toarray() if hasattr(X_tf, "toarray") else X_tf
        probs_all = clf.predict_proba(np.asarray(X_tf, dtype=np.float32))

    elif model_choice == "MiniRocket + MLP":
        rocket = joblib.load("models/v3/minirocket_transformer.joblib")
        X_tf   = rocket.transform(X.reshape(X.shape[0], 1, X.shape[1]))
        X_tf   = X_tf.toarray() if hasattr(X_tf, "toarray") else X_tf
        X_tf   = np.asarray(X_tf, dtype=np.float32)
        input_dim = int(np.load("models/v3/minirocket_input_dim.npy")[0])
        model = MiniRocketMLP(input_dim=input_dim, num_classes=num_classes)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device).eval()
        with torch.no_grad():
            probs_all = F.softmax(model(torch.tensor(X_tf).to(device)), dim=1).cpu().numpy()

    elif model_choice == "MiniRocketMLP_V2 (fold‑5)":
        # -------- New branch: MiniRocketMLP_V2 + fold‑5 transformer ----------
        rocket = joblib.load("models/v4/rocket_fold5.joblib")
        X_tf   = rocket.transform(X.reshape(X.shape[0], 1, X.shape[1]))
        X_tf   = X_tf.toarray() if hasattr(X_tf, "toarray") else X_tf
        X_tf   = np.asarray(X_tf, dtype=np.float32)
        model = MiniRocketMLP_V2(input_dim=X_tf.shape[1], num_classes=num_classes)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device).eval()
        with torch.no_grad():
            probs_all = F.softmax(model(torch.tensor(X_tf).to(device)), dim=1).cpu().numpy()

    else:  # ExpandedResNet1D or other CNN baselines
        model = ExpandedResNet1D(input_channels=1, num_classes=num_classes)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device).eval()
        with torch.no_grad():
            outputs = model(torch.tensor(X[:, np.newaxis, :], dtype=torch.float32).to(device))
            probs_all = F.softmax(outputs, dim=1).cpu().numpy()

    # ------------------------------------------------------------------
    # Aggregate per‑chunk
    # ------------------------------------------------------------------
    for i in range(num_chunks):
        chunk_probs = probs_all[i * chunk_size:(i + 1) * chunk_size]
        mean_probs  = np.mean(chunk_probs, axis=0)
        confidences.append(np.max(mean_probs))
        win_classes.append(int(np.argmax(mean_probs))) 

    return confidences, win_classes 


#### getting model confidence as a CSV 

def chunk_confidence_to_df(confidences, winners, chunk_size):

    """
    Build a DataFrame with one row per chunk.

    Columns
    -------
    chunk_id      : 1‑based chunk index
    start_frame   : first frame number in the chunk (0‑based)
    end_frame     : last frame number in the chunk  (inclusive)
    max_conf      : highest averaged class probability in the chunk
    winner_class  : class index of that max_conf
    """
    rows = []
    for idx, (conf, win) in enumerate(zip(confidences, winners), start=1):
        start = (idx - 1) * chunk_size
        end   = start + chunk_size - 1
        rows.append({
            "chunk_id": idx,
            "start_frame": start,
            "end_frame": end,
            "max_conf": conf,
            "winner_class": win,
        })
    return pd.DataFrame(rows)


####################### mini rocket MLP V5
def _load_v5_artefacts(device="cpu"):
    """
    Internal helper to load the v5 MiniRocket transformer, MLP weights,
    input dimension, and label classes. Caches objects on first call.
    """
    cache = _load_v5_artefacts.__dict__          # simple function‑level cache
    if "rocket" not in cache:
        base = Path("models/v5")
        cache["rocket"]     = joblib.load(base / "minirocket_transformer.pkl")
        input_dim           = int(np.load(base / "minirocket_input_dim.npy")[0])
        cache["input_dim"]  = input_dim
        cache["label_vals"] = np.load(base / "label_encoder_classes.npy",
                              allow_pickle=True)
        # build the same architecture used in training
        from torch import nn
        class RocketMLP(nn.Module):
            def __init__(self, in_dim, n_classes):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(in_dim, 512),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(256, n_classes)
                )
            def forward(self, x):
                return self.net(x)
        model = RocketMLP(in_dim=input_dim, n_classes=len(cache["label_vals"]))
        model.load_state_dict(torch.load(base / "rocket_mlp_weights.pt",
                                         map_location=device))
        model.to(device).eval()
        cache["model"] = model
    return cache["rocket"], cache["model"], cache["label_vals"], cache["input_dim"]

def evaluate_minirocket_mlp_v5(csv_path, device="cpu"):
    """
    Predict `bond_type` for a cleaned fingerprint CSV using the v5
    MiniRocket + MLP model.

    Returns
    -------
    preds        : ndarray of string labels
    probs        : ndarray of soft‑probabilities, shape (N, 3)
    acc (or None): accuracy if ground‑truth column is present
    """
    import pandas as pd
    rocket, model, label_vals, _ = _load_v5_artefacts(device)

    # ── 1. Load CSV ─────────────────────────────────────────────────────────
    df = pd.read_csv(csv_path)

    # Drop pseudo‑header row if still present
    if df.loc[0, df.columns[-2]] == "bond_type":
        df = df.iloc[1:].reset_index(drop=True)

    # ── 2. Select numeric fingerprint columns only ──────────────────────────
    # Any non‑numeric column (frame_idx, bond_type, drug_name, etc.) is auto‑excluded
    num_cols = df.select_dtypes(include=["number"]).columns
    fp_cols  = [c for c in num_cols if c not in ("frame_idx",)]
    X = df[fp_cols].to_numpy(dtype=np.float32)[..., None]          # (N, C, 1)

    # Pad to 9 time‑points to satisfy MiniRocketMultivariate
    X = np.pad(X, ((0, 0), (0, 0), (0, 8)), mode="constant")

    # ── 3. Transform with fitted MiniRocket ────────────────────────────────
    X_tf = rocket.transform(X)
    X_tf = np.asarray(X_tf, dtype=np.float32)

    # ── 4. Predict with cached MLP -----------------------------------------
    with torch.no_grad():
        logits = model(torch.tensor(X_tf, device=device))
        probs  = F.softmax(logits, dim=1).cpu().numpy()
    preds = label_vals[probs.argmax(1)]

    # ── 5. Optional accuracy if ground truth available ---------------------
    acc = None
    if "bond_type" in df.columns:
        y_true = df["bond_type"].values
        acc    = accuracy_score(y_true, preds)

    return preds, probs, acc

