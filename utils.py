import os
import lime
import torch
import shap
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import streamlit.components.v1 as components
from lime.lime_tabular import LimeTabularExplainer
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder,label_binarize
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, 
                             precision_recall_curve, average_precision_score,precision_recall_fscore_support)


# ============================
# 1. Data Preprocessing
# ============================

def clean_and_save_drug_csv(drug_files: dict, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    for drug, config in drug_files.items():
        df = pd.read_csv(config["path"])
        df_cleaned = df.drop(index=2).reset_index(drop=True)
        header_residues = df_cleaned.iloc[0]
        header_types = df_cleaned.iloc[1]
        multi_index = pd.MultiIndex.from_arrays([header_residues, header_types])
        df_cleaned = df_cleaned.drop(index=[0, 1]).reset_index(drop=True)
        df_cleaned.columns = multi_index
        frame_col = df.iloc[3:, 0].reset_index(drop=True)
        df_cleaned.insert(0, ("meta", "frame"), frame_col.astype(int))
        if ("protein", "interaction") in df_cleaned.columns:
            df_cleaned = df_cleaned.drop(columns=[("protein", "interaction")])
        df_cleaned = df_cleaned.apply(pd.to_numeric, errors='ignore').fillna(0)
        df_cleaned[("meta", "bond_type")] = config["bond_type"]
        df_cleaned[("meta", "drug_name")] = config["drug_name"]
        df_cleaned = df_cleaned[
            [col for col in df_cleaned.columns if col[0] != "meta" or col[1] == "frame"]
            + [("meta", "bond_type"), ("meta", "drug_name")]
        ]
        output_path = os.path.join(output_dir, f"cleaned_{config['drug_name']}.csv")
        df_cleaned.to_csv(output_path, index=False)
        print(f"✅ Saved cleaned CSV for {drug} at {output_path}")


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
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(ResidualBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = nn.Sequential()
        if in_channels != out_channels:
            self.downsample = nn.Conv1d(in_channels, out_channels, 1)

    def forward(self, x):
        identity = self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class ResNet1D(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(ResNet1D, self).__init__()
        self.layer1 = ResidualBlock1D(input_channels, 32)
        self.layer2 = ResidualBlock1D(32, 64)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.pool(x)
        x = self.flatten(x)
        return self.fc(x)


# ============================
# 3. Model Evaluation
# ============================
def evaluate_on_new_csv(csv_path, model_path, y_labels, device):
    import torch
    from torch.utils.data import DataLoader
    import torch.nn.functional as F
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    # Load cleaned CSV
    df = pd.read_csv(csv_path, header=[0, 1])
    X = df.loc[:, df.columns.get_level_values(0) != "meta"].to_numpy(dtype=np.float32)
    y = df[("meta", "bond_type")].values

    # Encode labels
    le = LabelEncoder()
    le.fit(y_labels)
    y_encoded = le.transform(y)

    dataset = FingerprintDataset(X, y_encoded)
    loader = DataLoader(dataset, batch_size=32)

    model = ResNet1D(input_channels=1, num_classes=len(y_labels))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    all_preds, all_labels = [], []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            probs = F.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y_batch.numpy())

    acc = accuracy_score(all_labels, all_preds)
    print(f"\n✅ Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=y_labels))

    # Confusion matrix figure
    cm = confusion_matrix(all_labels, all_preds)
    fig_cm, ax_cm = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=y_labels)
    disp.plot(ax=ax_cm, cmap="Blues", colorbar=False)
    ax_cm.set_title("Confusion Matrix")

    result_df = pd.DataFrame({
        "True Label": [y_labels[i] for i in all_labels],
        "Predicted": [y_labels[i] for i in all_preds]
    })
    print("\nSample Predictions:")
    print(result_df.head(10))

    return result_df, fig_cm


# ============================
# 4. LIME
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
        num_features=12,
        top_labels=1
    )
    # Return the explanation HTML (to embed in Streamlit)
    return explanation.as_html(show_table=True)

# ============================
# 5. SHAP
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

    # Plot to a matplotlib figure and return it
    fig = plt.figure(figsize=(10,6))
    shap.plots.waterfall(explanation, max_display=20, show=False)
    plt.tight_layout()
    return fig, pred_label, confidence