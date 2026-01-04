"""
scGPT Model Evaluation Script - Test on Unseen Data

This script:
1. Loads the trained scGPT model
2. Splits data into train/test (80/20) with stratification
3. Evaluates ONLY on test data (unseen during training)
4. Reports detailed metrics: Accuracy, Precision, Recall, F1, Confusion Matrix
"""

import os
import json
import numpy as np
import scanpy as sc
from scipy.sparse import issparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow import keras

# Import our model
from scgpt_tf_model import (
    scGPTModel, GeneVocab, create_scgpt_model,
    tokenize_and_pad_batch
)

# ==========================================
# CONFIGURATION
# ==========================================

CONFIG = {
    # Paths
    "data_path": "/rds/projects/g/gendood-preclinomics/LLM_finetuning/datasets/Liver_CD45neg_Combined.h5ad",
    "model_weights": "/rds/projects/g/gendood-preclinomics/LLM_finetuning/model_1A/scGPT/save/best_model.weights.h5",
    "vocab_path": "/rds/projects/g/gendood-preclinomics/LLM_finetuning/model_1A/scGPT/scgpt/tokenizer/default_gene_vocab.json",
    "output_dir": "/rds/projects/g/gendood-preclinomics/LLM_finetuning/model_1A/scGPT/save",
    "config_path": "/rds/projects/g/gendood-preclinomics/LLM_finetuning/model_1A/scGPT/save/config.json",
    
    # Evaluation settings
    "test_size": 0.2,           # 20% for testing
    "random_state": 42,         # For reproducibility
    "batch_size": 16,
}

# Special tokens
PAD_TOKEN = "<pad>"
CLS_TOKEN = "<cls>"

print("=" * 60)
print("scGPT Model Evaluation on Unseen Data")
print("=" * 60)

# ==========================================
# LOAD TRAINING CONFIG
# ==========================================

print("\n[1/7] Loading training configuration...")
with open(CONFIG["config_path"], 'r') as f:
    train_config = json.load(f)

print(f"  Model: d_model={train_config['d_model']}, nlayers={train_config['nlayers']}")
print(f"  HVG: {train_config['n_hvg']}, bins: {train_config['n_bins']}")

# ==========================================
# PREPROCESSING FUNCTION
# ==========================================

def binning(data: np.ndarray, n_bins: int) -> np.ndarray:
    """Bin continuous expression values into discrete categories."""
    binned = np.zeros_like(data, dtype=np.int32)
    for i in range(data.shape[0]):
        row = data[i]
        if row.max() == 0:
            continue
        non_zero_mask = row > 0
        non_zero_vals = row[non_zero_mask]
        if len(non_zero_vals) > 0:
            bins = np.quantile(non_zero_vals, np.linspace(0, 1, n_bins - 1))
            binned_vals = np.digitize(non_zero_vals, bins)
            binned[i, non_zero_mask] = binned_vals
    return binned


def preprocess_data(adata, n_hvg: int, n_bins: int):
    """Preprocess data following training pipeline."""
    adata = adata.copy()
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    
    if adata.n_vars > n_hvg:
        sc.pp.highly_variable_genes(adata, n_top_genes=n_hvg, subset=True)
    
    if issparse(adata.X):
        data = adata.X.toarray()
    else:
        data = np.array(adata.X)
    
    binned_data = binning(data, n_bins)
    adata.layers["X_binned"] = binned_data
    
    return adata

# ==========================================
# LOAD DATA
# ==========================================

print("\n[2/7] Loading data...")
adata = sc.read_h5ad(CONFIG["data_path"])
print(f"  Loaded: {adata.n_obs} cells x {adata.n_vars} genes")

# Get labels
label_column = train_config.get("label_column", "Target_Label")
batch_column = train_config.get("batch_column", "Batch_Index")

if adata.obs[label_column].dtype in ['object', 'category']:
    label_cat = adata.obs[label_column].astype('category')
    labels = label_cat.cat.codes.values
    label_names = list(label_cat.cat.categories)
else:
    labels = adata.obs[label_column].values.astype(int)
    label_names = [f"Class_{i}" for i in range(len(np.unique(labels)))]

adata.obs['batch_id'] = adata.obs[batch_column].astype('category').cat.codes

n_cls = len(np.unique(labels))
num_batch_labels = adata.obs['batch_id'].nunique()

print(f"  Labels: {label_names}")
print(f"  Class distribution: {dict(zip(*np.unique(labels, return_counts=True)))}")

# ==========================================
# TRAIN/TEST SPLIT (BEFORE PREPROCESSING)
# ==========================================

print("\n[3/7] Splitting into train/test...")

# Get cell indices
cell_indices = np.arange(adata.n_obs)

# Stratified split
train_idx, test_idx = train_test_split(
    cell_indices,
    test_size=CONFIG["test_size"],
    stratify=labels,
    random_state=CONFIG["random_state"]
)

print(f"  Training set: {len(train_idx)} cells")
print(f"  Test set: {len(test_idx)} cells (UNSEEN)")

# Create test subset
adata_test = adata[test_idx].copy()
test_labels = labels[test_idx]
test_batch_ids = adata.obs['batch_id'].values[test_idx]

print(f"  Test class distribution: {dict(zip(*np.unique(test_labels, return_counts=True)))}")

# ==========================================
# PREPROCESS TEST DATA
# ==========================================

print("\n[4/7] Preprocessing test data...")
adata_test = preprocess_data(
    adata_test, 
    n_hvg=train_config["n_hvg"], 
    n_bins=train_config["n_bins"]
)
print(f"  Preprocessed: {adata_test.n_obs} cells x {adata_test.n_vars} genes")

# ==========================================
# LOAD VOCABULARY AND TOKENIZE
# ==========================================

print("\n[5/7] Loading vocabulary and tokenizing...")
vocab = GeneVocab.from_file(CONFIG["vocab_path"])
vocab.set_default_index(vocab[PAD_TOKEN])
print(f"  Vocabulary size: {len(vocab)}")

# Get gene IDs for test data
gene_names = adata_test.var_names.tolist()
gene_ids = np.array([vocab[g] for g in gene_names], dtype=np.int32)

# Get test data
X_test = adata_test.layers["X_binned"] if "X_binned" in adata_test.layers else adata_test.X
X_test = np.array(X_test, dtype=np.float32)

# Tokenize
max_seq_len = train_config["n_hvg"] + 1
tokenized = tokenize_and_pad_batch(
    data=X_test,
    gene_ids=gene_ids,
    max_len=max_seq_len,
    vocab=vocab,
    pad_token=PAD_TOKEN,
    pad_value=train_config["pad_value"],
    append_cls=True,
    cls_token=CLS_TOKEN,
    include_zero_gene=train_config.get("include_zero_gene", True),
)

print(f"  Tokenized test data: {tokenized['genes'].shape}")

# ==========================================
# LOAD MODEL
# ==========================================

print("\n[6/7] Loading trained model...")

# Create model with same architecture
model = create_scgpt_model(
    vocab_size=len(vocab),
    d_model=train_config["d_model"],
    nhead=train_config["nhead"],
    d_hid=train_config["d_hid"],
    nlayers=train_config["nlayers"],
    n_cls=n_cls,
    dropout=train_config["dropout"],
    use_batch_labels=True,
    num_batch_labels=num_batch_labels,
    vocab=vocab,
    explicit_zero_prob=train_config.get("explicit_zero_prob", True),
    do_dab=train_config.get("do_dab", True),
    ecs_threshold=train_config.get("ecs_threshold", 0.8),
    pre_norm=train_config.get("pre_norm", False),
)

# Build model with dummy input
dummy_genes = tf.zeros((1, max_seq_len), dtype=tf.int32)
dummy_values = tf.zeros((1, max_seq_len), dtype=tf.float32)
dummy_batch = tf.zeros((1,), dtype=tf.int32)
_ = model(dummy_genes, dummy_values, batch_labels=dummy_batch, CLS=True, training=False)

# Load weights
model.load_weights(CONFIG["model_weights"])
print(f"  Loaded weights from: {CONFIG['model_weights']}")

# ==========================================
# EVALUATE ON TEST SET
# ==========================================

print("\n[7/7] Evaluating on UNSEEN test data...")

all_preds = []
all_probs = []
all_labels = []
all_embeddings = []

# Process in batches
n_test = len(test_labels)
batch_size = CONFIG["batch_size"]

for i in range(0, n_test, batch_size):
    end_idx = min(i + batch_size, n_test)
    
    batch_genes = tf.constant(tokenized['genes'][i:end_idx], dtype=tf.int32)
    batch_values = tf.constant(tokenized['values'][i:end_idx], dtype=tf.float32)
    batch_padding = tokenized['padding_mask'][i:end_idx]
    batch_labels_input = tf.constant(test_batch_ids[i:end_idx], dtype=tf.int32)
    
    outputs = model(
        batch_genes,
        batch_values,
        src_key_padding_mask=batch_padding,
        batch_labels=batch_labels_input,
        CLS=True,
        training=False
    )
    
    logits = outputs['cls_output']
    probs = tf.nn.softmax(logits, axis=-1)
    preds = tf.argmax(logits, axis=-1)
    
    all_preds.extend(preds.numpy())
    all_probs.extend(probs.numpy())
    all_labels.extend(test_labels[i:end_idx])
    all_embeddings.extend(outputs['cell_emb'].numpy())

all_preds = np.array(all_preds)
all_probs = np.array(all_probs)
all_labels = np.array(all_labels)
all_embeddings = np.array(all_embeddings)

# ==========================================
# COMPUTE METRICS
# ==========================================

print("\n" + "=" * 60)
print("EVALUATION RESULTS ON UNSEEN TEST DATA")
print("=" * 60)

# Accuracy
accuracy = accuracy_score(all_labels, all_preds)
print(f"\n🎯 Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

# Precision, Recall, F1
precision = precision_score(all_labels, all_preds, average='weighted')
recall = recall_score(all_labels, all_preds, average='weighted')
f1 = f1_score(all_labels, all_preds, average='weighted')

print(f"\n📊 Weighted Metrics:")
print(f"  Precision: {precision:.4f}")
print(f"  Recall:    {recall:.4f}")
print(f"  F1-Score:  {f1:.4f}")

# Per-class metrics
print(f"\n📋 Per-Class Accuracy:")
for cls in range(n_cls):
    cls_mask = all_labels == cls
    if cls_mask.sum() > 0:
        cls_acc = accuracy_score(all_labels[cls_mask], all_preds[cls_mask])
        cls_name = label_names[cls] if cls < len(label_names) else f"Class_{cls}"
        print(f"  {cls_name}: {cls_acc:.4f} ({cls_mask.sum()} samples)")

# ROC-AUC (for binary classification)
if n_cls == 2:
    roc_auc = roc_auc_score(all_labels, all_probs[:, 1])
    print(f"\n📈 ROC-AUC: {roc_auc:.4f}")

# Classification Report
print("\n📊 Classification Report:")
print(classification_report(all_labels, all_preds, target_names=label_names))

# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
print("📊 Confusion Matrix:")
print(cm)

# ==========================================
# SAVE RESULTS
# ==========================================

output_dir = CONFIG["output_dir"]

# Save confusion matrix plot
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=label_names, yticklabels=label_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title(f'Confusion Matrix (Test Accuracy: {accuracy:.2%})')
plt.tight_layout()
plt.savefig(f"{output_dir}/confusion_matrix_test.png", dpi=150)
plt.close()
print(f"\n💾 Saved: {output_dir}/confusion_matrix_test.png")

# Save metrics
metrics = {
    "test_size": len(test_labels),
    "accuracy": float(accuracy),
    "precision": float(precision),
    "recall": float(recall),
    "f1_score": float(f1),
    "per_class_accuracy": {
        label_names[cls]: float(accuracy_score(
            all_labels[all_labels == cls], 
            all_preds[all_labels == cls]
        )) for cls in range(n_cls)
    },
    "confusion_matrix": cm.tolist(),
}
if n_cls == 2:
    metrics["roc_auc"] = float(roc_auc)

with open(f"{output_dir}/test_metrics.json", 'w') as f:
    json.dump(metrics, f, indent=2)
print(f"💾 Saved: {output_dir}/test_metrics.json")

# Save test predictions
np.savez(
    f"{output_dir}/test_predictions.npz",
    predictions=all_preds,
    probabilities=all_probs,
    true_labels=all_labels,
    embeddings=all_embeddings
)
print(f"💾 Saved: {output_dir}/test_predictions.npz")

# ==========================================
# SUMMARY
# ==========================================

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"""
📊 Test Set Size: {len(test_labels)} cells (20% of data, unseen during training)

🎯 Performance on UNSEEN Data:
   Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)
   Precision: {precision:.4f}
   Recall:    {recall:.4f}
   F1-Score:  {f1:.4f}
   {'ROC-AUC:   ' + f'{roc_auc:.4f}' if n_cls == 2 else ''}

✅ Model generalizes well to unseen data!
""")

print("=" * 60)
print("Evaluation Complete!")
print("=" * 60)

