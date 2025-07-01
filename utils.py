import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay, classification_report,
    roc_curve, auc, roc_auc_score, accuracy_score, f1_score
)
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer
import shap 
import torch.nn.functional as F


# ======================
# 1D CNN Model
# ======================
class CNN1D(nn.Module):
    def __init__(self, input_length, num_classes):
        super(CNN1D, self).__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),
            nn.Flatten(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        return self.net(x)


# ======================
# PyTorch Dataset Class
# ======================
class FingerprintDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx].unsqueeze(0), self.y[idx]


# =======================
# Data Cleaning Function
# =======================
def clean_and_save_drug_csv(drug_files: dict, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    for drug, config in drug_files.items():
        df = pd.read_csv(config["path"])

        # Step 1: Drop third row
        df_cleaned = df.drop(index=2).reset_index(drop=True)

        # Step 2–3: MultiIndex creation
        header_residues = df_cleaned.iloc[0]
        header_types = df_cleaned.iloc[1]
        multi_index = pd.MultiIndex.from_arrays([header_residues, header_types])

        # Step 4–5: Clean header rows
        df_cleaned = df_cleaned.drop(index=[0, 1]).reset_index(drop=True)
        df_cleaned.columns = multi_index

        # Step 6: Insert frame column
        frame_col = df.iloc[3:, 0].reset_index(drop=True)
        df_cleaned.insert(0, ("meta", "frame"), frame_col.astype(int))

        # Step 6.5: Drop redundant column
        if ("protein", "interaction") in df_cleaned.columns:
            df_cleaned = df_cleaned.drop(columns=[("protein", "interaction")])

        # Step 7–8: Convert numerics and fill NaNs
        df_cleaned = df_cleaned.apply(pd.to_numeric, errors='ignore').fillna(0)

        # Step 9: Add meta columns
        df_cleaned[("meta", "bond_type")] = config["bond_type"]
        df_cleaned[("meta", "drug_name")] = config["drug_name"]

        # Step 10: Reorder columns
        df_cleaned = df_cleaned[
            [col for col in df_cleaned.columns if col[0] != "meta" or col[1] == "frame"]
            + [("meta", "bond_type"), ("meta", "drug_name")]
        ]

        # Save to CSV
        output_path = os.path.join(output_dir, f"cleaned_{config['drug_name']}.csv")
        df_cleaned.to_csv(output_path, index=False)
        print(f"Saved cleaned CSV for {drug}: {output_path}")


# ===================================
# NumPy Feature + Label Preparation
# ===================================
def prepare_numpy_data(drug_names: list, input_dir: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    X_list = []
    y_list = []

    for drug in drug_names:
        df = pd.read_csv(os.path.join(input_dir, f"cleaned_{drug}.csv"), header=[0, 1])
        X = df.loc[:, df.columns.get_level_values(0) != "meta"].to_numpy(dtype=np.float32)
        y = df[("meta", "bond_type")]
        X_list.append(X)
        y_list.append(y)

    X_all = np.vstack(X_list)
    y_all_series = pd.concat(y_list)

    le = LabelEncoder()
    y_all = le.fit_transform(y_all_series)

    np.save(os.path.join(output_dir, "X_all.npy"), X_all)
    np.save(os.path.join(output_dir, "y_all.npy"), y_all)
    np.save(os.path.join(output_dir, "y_labels.npy"), le.classes_)

    print(f"Saved NumPy arrays to: {output_dir}")
    return X_all, y_all, le.classes_



# ======================
# Training Loop
# ======================
def train_model(model, train_loader, device, epochs=50, lr=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss:.4f}")


# ======================
# Evaluation
# ======================
def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y_batch.numpy())

    return np.array(all_labels), np.array(all_preds)


# ======================
# Save Model
# ======================
def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Model weights saved to: {path}")


# ======================
# Diagnostic Plots
# ======================
def plot_classification_metrics(y_true, y_pred, labels):

    from collections import Counter

    # ==============================
    # 1. Prediction Counts per Class
    # ==============================
    print("\nTotal Predictions Per Class:")
    pred_counter = Counter(y_pred)
    for class_index, count in pred_counter.items():
        print(f"{labels[class_index]}: {count} frames")

    
    # ==============================
    # 2. Confusion Matrix
    # ==============================
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    plt.title("Confusion Matrix")
    plt.show()

    # ==============================
    # 3. Classification Report
    # ==============================
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=labels))

    # ==============================
    # 4. Basic Summary Metrics
    # ==============================
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted")
    print(f"\nAccuracy: {acc:.4f}")
    print(f"Weighted F1 Score: {f1:.4f}")

    # ==============================
    # 5. ROC Curve (for binary or multi-class)
    # ==============================
    y_true_bin = label_binarize(y_true, classes=np.arange(len(labels)))
    y_pred_bin = label_binarize(y_pred, classes=np.arange(len(labels)))

    if y_true_bin.shape[1] == 1:
        y_true_bin = np.hstack([1 - y_true_bin, y_true_bin])
        y_pred_bin = np.hstack([1 - y_pred_bin, y_pred_bin])

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(y_true_bin.shape[1]):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_bin[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(8, 6))
    for i, label in enumerate(labels):
        plt.plot(fpr[i], tpr[i], label=f"{label} (AUC = {roc_auc[i]:.2f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()


# ======================
# Prediction on unseen data
# ======================
def evaluate_on_new_csv(csv_path, model_path, y_labels, device):
    import torch
    from torch.utils.data import DataLoader

    # Load cleaned CSV
    df = pd.read_csv(csv_path, header=[0, 1])

    # Extract features
    X = df.loc[:, df.columns.get_level_values(0) != "meta"].to_numpy(dtype=np.float32)
    y = df[("meta", "bond_type")].values

    # Encode labels
    le = LabelEncoder()
    le.fit(y_labels)  # Use label space from training
    y_encoded = le.transform(y)

    # Dataset + Dataloader
    dataset = FingerprintDataset(X, y_encoded)
    loader = DataLoader(dataset, batch_size=32)

    # Load model
    model = CNN1D(input_length=X.shape[1], num_classes=len(y_labels))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Predict
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y_batch.numpy())

    # Show metrics
    plot_classification_metrics(np.array(all_labels), np.array(all_preds), y_labels)


# ======================
# LIME function 
# ======================
def explain_prediction_with_lime(model, dataset, index, y_labels, feature_names, device):
    model.eval()

    X_sample, y_sample = dataset[index]
    X_np = X_sample.squeeze(0).numpy()  # shape: (features,)

    # Define prediction function for LIME
    def predict_fn(inputs):
        inputs_tensor = torch.tensor(inputs[:, np.newaxis, :], dtype=torch.float32).to(device)
        outputs = model(inputs_tensor)
        probs = F.softmax(outputs, dim=1).detach().cpu().numpy()
        return probs


    # Setup LIME
    explainer = LimeTabularExplainer(
        training_data=np.array([sample[0].squeeze(0).numpy() for sample in dataset]),
        mode="classification",
        class_names=y_labels,
        feature_names=feature_names,
        discretize_continuous=False
    )


    # Explain one instance
    explanation = explainer.explain_instance(
        data_row=X_np,
        predict_fn=predict_fn,
       num_features=12,
        top_labels=1
    )

    explanation.show_in_notebook(show_table=True)
    explanation.save_to_file("lime_explanation.html")


# ======================
# SHAP function 
# ======================
def explain_prediction_shap_deep(model, X_train, X_mdma, real_feature_names, frame_index, device):
    model.eval()

    # Convert background and sample to tensors
    background = torch.tensor(X_train[:200]).unsqueeze(1).float().to(device)  # shape: (N, 1, features)
    mdma_tensor = torch.tensor(X_mdma).unsqueeze(1).float().to(device)

    # Create DeepExplainer
    explainer = shap.DeepExplainer(model, background)

    # Predict and get SHAP values for selected frame
    i = frame_index
    shap_values = explainer.shap_values(mdma_tensor[i:i+1], check_additivity=False)

    # Get predicted class
    pred_class = torch.argmax(model(mdma_tensor[i:i+1])).item()

    # Extract SHAP vector for that class
    shap_vector = shap_values[pred_class][0]
    if shap_vector.ndim == 2:
        shap_vector = shap_vector[:, pred_class]

    # Create SHAP Explanation object
    explanation_obj = shap.Explanation(
        values=shap_vector,
        base_values=explainer.expected_value[pred_class],
        data=X_mdma[i],
        feature_names=real_feature_names
    )

    # Plot waterfall chart
    shap.plots.waterfall(explanation_obj)

