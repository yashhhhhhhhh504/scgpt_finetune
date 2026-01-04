"""
TensorFlow training script for scGPT - Fine-tuning for Liver Endothelial Classification.

THESIS PROJECT: Liver Endothelial Zonation in NASH/MASLD
==========================================================
Dataset: Liver_CD45neg_Combined.h5ad (24,904 cells × 33,694 genes)

Classification Task:
- Condition: Healthy (Donor) vs NASH
- Zonation: Sinusoidal, Portal, Central Venous endothelial cells

Based on the official scGPT tutorial: https://scgpt.readthedocs.io/en/latest/tutorial_integraion.html

Objectives:
- GEPC: Gene Expression Prediction for Cell (masked value prediction)
- CLS: Cell-type/condition classification
- ECS: Elastic Cell Similarity (contrastive learning)
- DAB: Domain Adversarial Batch correction
"""
import os
import gc
import json
import datetime
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import scanpy as sc
from scipy.sparse import issparse
import tensorflow as tf
from tensorflow import keras
from scgpt_tf_model import (
    scGPTModel, GeneVocab, create_scgpt_model,
    tokenize_and_pad_batch, random_mask_value,
    masked_mse_loss, masked_relative_error, criterion_neg_log_bernoulli
)
warnings.filterwarnings('ignore')
CONFIG = {
    # Data paths
    "data_path": "/rds/projects/g/gendood-preclinomics/LLM_finetuning/datasets/Liver_CD45neg_Combined.h5ad",
    "vocab_path": "/rds/projects/g/gendood-preclinomics/LLM_finetuning/model_1A/scGPT/scgpt/tokenizer/default_gene_vocab.json",  # Official scGPT vocab (~48K genes)
    "output_dir": "/rds/projects/g/gendood-preclinomics/LLM_finetuning/model_1A/scGPT/save",
    # Column names from the dataset (Liver_CD45neg_Combined.h5ad)
    # Available columns: Sample_ID, Condition, Target_Label, Batch_Index
    "batch_column": "Batch_Index",     
    "label_column": "Target_Label",     
    "condition_column": "Condition",   
    "n_hvg": 1200,            
    "n_bins": 51,               
    "mask_ratio": 0.4,          
    "mask_value": -1,          
    "pad_value": -2,          
    # Model hyperparameters (from official scGPT)
    "d_model": 512,           
    "nhead": 8,                 
    "d_hid": 512,              
    "nlayers": 12,              
    "nlayers_cls": 3,           
    "dropout": 0.2,            
    "batch_size": 16,         
    "epochs": 30,               
    "learning_rate": 5e-5,    
    "schedule_ratio": 0.95,    
    "GEPC": True,              
    "CLS": True,               
    "ecs_threshold": 0.8,       
    "dab_weight": 1.0,         
    "do_dab": True,             
    "explicit_zero_prob": True,
    "pre_norm": False,         
    "include_zero_gene": True, 
    "log_interval": 100,       
    "save_eval_interval": 5,  
}
# Special tokens (from official scGPT)
PAD_TOKEN = "<pad>"
CLS_TOKEN = "<cls>"
MASK_TOKEN = "<mask>"
SPECIAL_TOKENS = [PAD_TOKEN, CLS_TOKEN, "<eoc>", MASK_TOKEN]
# PREPROCESSING FUNCTIONS

def binning(data: np.ndarray, n_bins: int) -> np.ndarray:
    """
    Bin continuous expression values into discrete categories.
    Following official scGPT binning strategy.
    """
    binned = np.zeros_like(data, dtype=np.int32)
    for i in range(data.shape[0]):
        row = data[i]
        if row.max() == 0:
            continue
        # Only bin non-zero values
        non_zero_mask = row > 0
        non_zero_vals = row[non_zero_mask]
        if len(non_zero_vals) > 0:
            # Create quantile-based bins
            bins = np.quantile(non_zero_vals, np.linspace(0, 1, n_bins - 1))
            # Digitize non-zero values
            binned_vals = np.digitize(non_zero_vals, bins)
            binned[i, non_zero_mask] = binned_vals
    return binned


def preprocess_data(adata, n_hvg: int = 1200, n_bins: int = 51):
    """
    Preprocess single-cell data following official scGPT pipeline.
    """
    print("Preprocessing data...")
    
    # Make a copy to avoid modifying original
    adata = adata.copy()
    
    # Step 1: Filter genes and cells (optional)
    print("  Filtering genes and cells...")
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.filter_cells(adata, min_genes=200)
    
    # Step 2: Normalize
    print("  Normalizing...")
    sc.pp.normalize_total(adata, target_sum=1e4)
    
    # Step 3: Log transform
    print("  Log1p transforming...")
    sc.pp.log1p(adata)
    
    # Step 4: Select HVG
    if adata.n_vars > n_hvg:
        print(f"  Selecting top {n_hvg} highly variable genes...")
        sc.pp.highly_variable_genes(adata, n_top_genes=n_hvg, subset=True)
    
    # Step 5: Binning
    print(f"  Binning into {n_bins} bins...")
    if issparse(adata.X):
        data = adata.X.toarray()
    else:
        data = np.array(adata.X)
    
    binned_data = binning(data, n_bins)
    adata.layers["X_binned"] = binned_data
    
    print(f"  Final shape: {adata.n_obs} cells x {adata.n_vars} genes")
    
    return adata



# DATA LOADING

print("=" * 60)
print("scGPT Fine-Tuning (TensorFlow)")
print("=" * 60)

print("\n[1/6] Loading data...")
adata = sc.read_h5ad(CONFIG["data_path"])
print(f"  Loaded: {adata.n_obs} cells x {adata.n_vars} genes")
print(f"  Available columns: {list(adata.obs.columns)[:10]}...")


# CONFIGURE COLUMNS (From your dataset)

print("\n[2/6] Setting up columns...")
print(f"  Dataset columns: {list(adata.obs.columns)}")

# Use specified columns from CONFIG
batch_column = CONFIG["batch_column"]
label_column = CONFIG["label_column"]
condition_column = CONFIG.get("condition_column", "Condition")

# Validate columns exist
if batch_column not in adata.obs.columns:
    print(f"  WARNING: Batch column '{batch_column}' not found, using Sample_ID")
    batch_column = "Sample_ID" if "Sample_ID" in adata.obs.columns else adata.obs.columns[0]

if label_column not in adata.obs.columns:
    print(f"  WARNING: Label column '{label_column}' not found")
    label_column = "Condition" if "Condition" in adata.obs.columns else adata.obs.columns[0]

print(f"  Batch column: '{batch_column}'")
print(f"  Label column: '{label_column}'")
print(f"  Condition column: '{condition_column}'")

# Create batch IDs for batch correction
adata.obs['batch_id'] = adata.obs[batch_column].astype('category').cat.codes

# Convert labels to integers
if adata.obs[label_column].dtype in ['object', 'category']:
    label_cat = adata.obs[label_column].astype('category')
    adata.obs['Target_Label'] = label_cat.cat.codes
    label_names = list(label_cat.cat.categories)
else:
    adata.obs['Target_Label'] = adata.obs[label_column].astype(int)
    label_names = [f"Class_{i}" for i in range(adata.obs['Target_Label'].nunique())]

n_cls = adata.obs['Target_Label'].nunique()
num_batch_labels = adata.obs['batch_id'].nunique()

print(f"  Number of classes: {n_cls}")
print(f"  Label mapping: {dict(zip(range(len(label_names)), label_names))}")
print(f"  Number of batch labels: {num_batch_labels}")


# PREPROCESS


print("\n[3/6] Preprocessing...")
adata = preprocess_data(adata, n_hvg=CONFIG["n_hvg"], n_bins=CONFIG["n_bins"])

# Get gene names
gene_names = adata.var_names.tolist()


# CREATE/LOAD VOCABULARY


print("\n[4/6] Setting up vocabulary...")

vocab_path = CONFIG["vocab_path"]
if os.path.exists(vocab_path):
    print(f"  Loading vocabulary from {vocab_path}")
    vocab = GeneVocab.from_file(vocab_path)
else:
    print(f"  Creating new vocabulary...")
    gene_to_idx = {PAD_TOKEN: 0, CLS_TOKEN: 1, "<eoc>": 2, MASK_TOKEN: 3}
    for gene in gene_names:
        if gene not in gene_to_idx:
            gene_to_idx[gene] = len(gene_to_idx)
    vocab = GeneVocab(gene_to_idx)
    vocab.save(vocab_path)
    print(f"  Saved vocabulary to {vocab_path}")

vocab.set_default_index(vocab[PAD_TOKEN])
print(f"  Vocabulary size: {len(vocab)}")

# Convert gene names to IDs
gene_ids = np.array([vocab[g] for g in gene_names], dtype=np.int32)


# PREPARE DATASET


print("\n[5/6] Preparing TensorFlow dataset...")

# Get data
if "X_binned" in adata.layers:
    X = adata.layers["X_binned"]
else:
    X = adata.X.toarray() if issparse(adata.X) else np.array(adata.X)

X = np.array(X, dtype=np.float32)
labels = adata.obs['Target_Label'].values.astype(np.int32)
batch_ids = adata.obs['batch_id'].values.astype(np.int32)

print(f"  Data shape: {X.shape}")
print(f"  Labels shape: {labels.shape}")

# Tokenize data
print("  Tokenizing...")
max_seq_len = CONFIG["n_hvg"] + 1  # +1 for CLS token

tokenized = tokenize_and_pad_batch(
    data=X,
    gene_ids=gene_ids,
    max_len=max_seq_len,
    vocab=vocab,
    pad_token=PAD_TOKEN,
    pad_value=CONFIG["pad_value"],
    append_cls=True,
    cls_token=CLS_TOKEN,
    include_zero_gene=CONFIG["include_zero_gene"],
)

print(f"  Tokenized genes shape: {tokenized['genes'].shape}")
print(f"  Tokenized values shape: {tokenized['values'].shape}")

# Create TensorFlow dataset
n_samples = len(labels)

def create_dataset():
    """Create a tf.data.Dataset with masking applied."""
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    for idx in indices:
        genes = tokenized['genes'][idx]
        values = tokenized['values'][idx]
        padding_mask = tokenized['padding_mask'][idx]
        
        # Apply random masking
        masked_values, mask_positions = random_mask_value(
            values[np.newaxis, :],
            mask_ratio=CONFIG["mask_ratio"],
            mask_value=CONFIG["mask_value"],
            pad_value=CONFIG["pad_value"],
        )
        masked_values = masked_values[0]
        mask_positions = mask_positions[0]
        
        yield {
            "genes": genes,
            "values": values,
            "masked_values": masked_values,
            "mask_positions": mask_positions,
            "padding_mask": padding_mask,
            "batch_labels": batch_ids[idx],
            "labels": labels[idx],
        }

# Define output signature
output_signature = {
    "genes": tf.TensorSpec(shape=(max_seq_len,), dtype=tf.int32),
    "values": tf.TensorSpec(shape=(max_seq_len,), dtype=tf.float32),
    "masked_values": tf.TensorSpec(shape=(max_seq_len,), dtype=tf.float32),
    "mask_positions": tf.TensorSpec(shape=(max_seq_len,), dtype=tf.bool),
    "padding_mask": tf.TensorSpec(shape=(max_seq_len,), dtype=tf.bool),
    "batch_labels": tf.TensorSpec(shape=(), dtype=tf.int32),
    "labels": tf.TensorSpec(shape=(), dtype=tf.int32),
}

# Create batched dataset
dataset = tf.data.Dataset.from_generator(create_dataset, output_signature=output_signature)
dataset = dataset.batch(CONFIG["batch_size"])
dataset = dataset.prefetch(tf.data.AUTOTUNE)

n_batches = (n_samples + CONFIG["batch_size"] - 1) // CONFIG["batch_size"]
print(f"  Created dataset: {n_samples} samples, {n_batches} batches per epoch")


# MODEL SETUP


print("\n[6/6] Setting up model...")

# Check for GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"  Found {len(gpus)} GPU(s)")
    # Try to set memory growth (may fail if already initialized via TF_FORCE_GPU_ALLOW_GROWTH)
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("  Memory growth enabled")
    except RuntimeError as e:
        print(f"  Memory growth already configured (via environment variable)")
else:
    print("  No GPU found, using CPU")

# Create model
model = create_scgpt_model(
    vocab_size=len(vocab),
    d_model=CONFIG["d_model"],
    nhead=CONFIG["nhead"],
    d_hid=CONFIG["d_hid"],
    nlayers=CONFIG["nlayers"],
    n_cls=n_cls,
    dropout=CONFIG["dropout"],
    use_batch_labels=True,
    num_batch_labels=num_batch_labels,
    vocab=vocab,
    explicit_zero_prob=CONFIG["explicit_zero_prob"],
    do_dab=CONFIG["do_dab"],
    ecs_threshold=CONFIG["ecs_threshold"],
    pre_norm=CONFIG["pre_norm"],
)

# Build model
dummy_genes = tf.zeros((1, max_seq_len), dtype=tf.int32)
dummy_values = tf.zeros((1, max_seq_len), dtype=tf.float32)
dummy_batch = tf.zeros((1,), dtype=tf.int32)
_ = model(dummy_genes, dummy_values, batch_labels=dummy_batch, CLS=True, ECS=True, training=False)

print("\nModel Summary:")
model.summary()


# TRAINING SETUP



print("Training Setup")


# Optimizer with learning rate schedule
initial_lr = CONFIG["learning_rate"]
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=initial_lr,
    decay_steps=n_batches,
    decay_rate=CONFIG["schedule_ratio"],
    staircase=True
)
optimizer = keras.optimizers.AdamW(learning_rate=lr_schedule)

# Loss functions (with label smoothing for better generalization)
cls_loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
dab_loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# Metrics
metrics = {
    "total_loss": keras.metrics.Mean(),
    "mse_loss": keras.metrics.Mean(),
    "cls_loss": keras.metrics.Mean(),
    "ecs_loss": keras.metrics.Mean(),
    "dab_loss": keras.metrics.Mean(),
    "cls_accuracy": keras.metrics.SparseCategoricalAccuracy(),
}


@tf.function
def train_step(batch_data):
    """Single training step with all objectives."""
    genes = batch_data['genes']
    values = batch_data['values']
    masked_values = batch_data['masked_values']
    mask_positions = batch_data['mask_positions']
    padding_mask = batch_data['padding_mask']
    batch_labels = batch_data['batch_labels']
    labels = batch_data['labels']
    
    with tf.GradientTape() as tape:
        # Forward pass with masked values
        outputs = model(
            genes,
            masked_values,
            src_key_padding_mask=padding_mask,
            batch_labels=batch_labels,
            CLS=CONFIG["CLS"],
            ECS=True,
            training=True
        )
        
        total_loss = 0.0
        
        # 1. Masked MSE Loss (GEPC objective)
        if CONFIG["GEPC"]:
            mse_loss = masked_mse_loss(
                outputs['mlm_output'],
                values,
                mask_positions
            )
            total_loss += mse_loss
        else:
            mse_loss = tf.constant(0.0)
        
        # 2. Zero probability loss (if explicit)
        if CONFIG["explicit_zero_prob"] and "mlm_zero_probs" in outputs:
            zero_loss = criterion_neg_log_bernoulli(
                outputs['mlm_zero_probs'],
                values,
                mask_positions
            )
            total_loss += zero_loss
        
        # 3. Classification Loss (CLS objective)
        if CONFIG["CLS"]:
            cls_loss = cls_loss_fn(labels, outputs['cls_output'])
            total_loss += cls_loss
        else:
            cls_loss = tf.constant(0.0)
        
        # 4. Elastic Cell Similarity Loss (ECS objective)
        ecs_loss = outputs.get('loss_ecs', tf.constant(0.0))
        total_loss += ecs_loss
        
        # 5. Domain Adversarial Batch Loss (DAB objective)
        if CONFIG["do_dab"] and "dab_output" in outputs:
            dab_loss = dab_loss_fn(batch_labels, outputs['dab_output'])
            total_loss += CONFIG["dab_weight"] * dab_loss
        else:
            dab_loss = tf.constant(0.0)
    
    # Compute gradients
    gradients = tape.gradient(total_loss, model.trainable_variables)
    
    # Clip gradients
    gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
    
    # Apply gradients
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    # Update metrics
    metrics["total_loss"].update_state(total_loss)
    metrics["mse_loss"].update_state(mse_loss)
    metrics["cls_loss"].update_state(cls_loss)
    metrics["ecs_loss"].update_state(ecs_loss)
    metrics["dab_loss"].update_state(dab_loss)
    if CONFIG["CLS"]:
        metrics["cls_accuracy"].update_state(labels, outputs['cls_output'])
    
    return {
        "total_loss": total_loss,
        "mse_loss": mse_loss,
        "cls_loss": cls_loss,
        "ecs_loss": ecs_loss,
        "dab_loss": dab_loss,
    }



# TRAINING LOOP



print("Starting Fine-Tuning...")


# Create output directory
output_dir = Path(CONFIG["output_dir"])
output_dir.mkdir(parents=True, exist_ok=True)
# TensorBoard logging
log_dir = output_dir / "logs" / datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
summary_writer = tf.summary.create_file_writer(str(log_dir))
best_loss = float('inf')
global_step = 0
for epoch in range(CONFIG["epochs"]):
    print(f"\n{'='*60}")
    print(f"Epoch {epoch + 1}/{CONFIG['epochs']}")
    print(f"{'='*60}")
    # Reset metrics
    for metric in metrics.values():
        metric.reset_state()
    # Recreate dataset each epoch for fresh shuffling and masking
    dataset = tf.data.Dataset.from_generator(create_dataset, output_signature=output_signature)
    dataset = dataset.batch(CONFIG["batch_size"])
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    batch_count = 0
    for batch_data in dataset:
        losses = train_step(batch_data)
        batch_count += 1
        global_step += 1
        # Log progress
        if batch_count % CONFIG["log_interval"] == 0:
            print(f"  Batch {batch_count}/{n_batches} | "
                  f"Loss: {losses['total_loss']:.4f} | "
                  f"MSE: {losses['mse_loss']:.4f} | "
                  f"CLS: {losses['cls_loss']:.4f}")
    # Epoch summary
    epoch_loss = metrics["total_loss"].result()
    epoch_mse = metrics["mse_loss"].result()
    epoch_cls = metrics["cls_loss"].result()
    epoch_ecs = metrics["ecs_loss"].result()
    epoch_dab = metrics["dab_loss"].result()
    epoch_acc = metrics["cls_accuracy"].result()
    
    print(f"\nEpoch {epoch + 1} Complete:")
    print(f"  Total Loss: {epoch_loss:.4f}")
    print(f"  MSE Loss: {epoch_mse:.4f}")
    print(f"  CLS Loss: {epoch_cls:.4f}")
    print(f"  ECS Loss: {epoch_ecs:.4f}")
    print(f"  DAB Loss: {epoch_dab:.4f}")
    print(f"  Accuracy: {epoch_acc:.4f}")
    
    # TensorBoard logging
    with summary_writer.as_default():
        tf.summary.scalar('loss/total', epoch_loss, step=epoch)
        tf.summary.scalar('loss/mse', epoch_mse, step=epoch)
        tf.summary.scalar('loss/cls', epoch_cls, step=epoch)
        tf.summary.scalar('loss/ecs', epoch_ecs, step=epoch)
        tf.summary.scalar('loss/dab', epoch_dab, step=epoch)
        tf.summary.scalar('accuracy', epoch_acc, step=epoch)
    
    # Save best model
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        model.save_weights(str(output_dir / "best_model.weights.h5"))
        print(f"  Saved best model (loss: {best_loss:.4f})")
    
    # Periodic save
    if (epoch + 1) % CONFIG["save_eval_interval"] == 0:
        model.save_weights(str(output_dir / f"model_epoch_{epoch+1}.weights.h5"))
        print(f"  Saved checkpoint at epoch {epoch + 1}")


# FINAL SAVE


print("\n" + "=" * 60)
print("Saving final model...")
print("=" * 60)

model.save_weights(str(output_dir / "final_model.weights.h5"))
with open(output_dir / "config.json", 'w') as f:
    json.dump(CONFIG, f, indent=2)
with open(output_dir / "label_mapping.json", 'w') as f:
    json.dump({i: name for i, name in enumerate(label_names)}, f, indent=2)

print(f"Model saved to: {output_dir}")


# FINAL EVALUATION


print("\n" + "=" * 60)
print("Final Evaluation")
print("=" * 60)

# Load best weights
model.load_weights(str(output_dir / "best_model.weights.h5"))

# Evaluate on all data
all_preds = []
all_labels = []
all_embeddings = []
eval_dataset = tf.data.Dataset.from_generator(create_dataset, output_signature=output_signature)
eval_dataset = eval_dataset.batch(CONFIG["batch_size"])
for batch_data in eval_dataset:
    genes = batch_data['genes']
    values = batch_data['values']
    padding_mask = batch_data['padding_mask']
    batch_labels_batch = batch_data['batch_labels']
    labels_batch = batch_data['labels']
    outputs = model(
        genes, values,
        src_key_padding_mask=padding_mask,
        batch_labels=batch_labels_batch,
        CLS=True,
        training=False
    )
    preds = tf.argmax(outputs['cls_output'], axis=-1)
    all_preds.extend(preds.numpy())
    all_labels.extend(labels_batch.numpy())
    all_embeddings.extend(outputs['cell_emb'].numpy())
all_preds = np.array(all_preds)
all_labels = np.array(all_labels)
all_embeddings = np.array(all_embeddings)

# Overall accuracy
accuracy = np.mean(all_preds == all_labels)
print(f"\nFinal Accuracy: {accuracy:.4f}")

# Per-class accuracy
print("\nPer-class accuracy:")
for cls in range(n_cls):
    cls_mask = all_labels == cls
    if cls_mask.sum() > 0:
        cls_acc = np.mean(all_preds[cls_mask] == all_labels[cls_mask])
        cls_name = label_names[cls] if cls < len(label_names) else f"Class_{cls}"
        print(f"  {cls_name}: {cls_acc:.4f} ({cls_mask.sum()} samples)")
# Save embeddings
np.save(output_dir / "cell_embeddings.npy", all_embeddings)
print(f"\nCell embeddings saved: {all_embeddings.shape}")

# VISUALIZATION: UMAP of Cell Embeddings


print("\n" + "=" * 60)
print("Generating UMAP Visualization")
print("=" * 60)

try:
    import umap
    import matplotlib.pyplot as plt
    # Create UMAP projection
    print("Computing UMAP...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    embedding_2d = reducer.fit_transform(all_embeddings)

    # Save UMAP coordinates
    np.save(output_dir / "umap_coordinates.npy", embedding_2d)
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    # 1. Color by True Labels
    scatter1 = axes[0].scatter(embedding_2d[:, 0], embedding_2d[:, 1], 
                               c=all_labels, cmap='tab10', alpha=0.5, s=1)
    axes[0].set_title('UMAP - True Labels (Target_Label)')
    axes[0].set_xlabel('UMAP1')
    axes[0].set_ylabel('UMAP2')
    plt.colorbar(scatter1, ax=axes[0])
    # 2. Color by Predictions
    scatter2 = axes[1].scatter(embedding_2d[:, 0], embedding_2d[:, 1], 
                               c=all_preds, cmap='tab10', alpha=0.5, s=1)
    axes[1].set_title('UMAP - Model Predictions')
    axes[1].set_xlabel('UMAP1')
    axes[1].set_ylabel('UMAP2')
    plt.colorbar(scatter2, ax=axes[1])
    # 3. Color by Condition (Healthy vs NASH)
    if condition_column in adata.obs.columns:
        condition_codes = adata.obs[condition_column].astype('category').cat.codes.values
        scatter3 = axes[2].scatter(embedding_2d[:, 0], embedding_2d[:, 1], 
                                   c=condition_codes, cmap='coolwarm', alpha=0.5, s=1)
        axes[2].set_title(f'UMAP - {condition_column} (Healthy vs NASH)')
        axes[2].set_xlabel('UMAP1')
        axes[2].set_ylabel('UMAP2')
        plt.colorbar(scatter3, ax=axes[2])
    plt.tight_layout()
    plt.savefig(output_dir / "umap_embeddings.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  UMAP visualization saved to: {output_dir / 'umap_embeddings.png'}")
except ImportError:
    print("  UMAP not installed. Run: pip install umap-learn")
except Exception as e:
    print(f"  Could not generate UMAP: {e}")


# EXPORT FOR TRAJECTORY ANALYSIS

print("\n" + "=" * 60)
print("Exporting Data for Trajectory Analysis")
print("=" * 60)

# Create an AnnData with scGPT embeddings
try:
    from anndata import AnnData
    import pandas as pd
    # Create new AnnData with embeddings
    adata_emb = AnnData(X=all_embeddings)
    adata_emb.obs = adata.obs.copy()
    adata_emb.obs['scGPT_prediction'] = all_preds
    adata_emb.obs['scGPT_correct'] = (all_preds == all_labels)
    
    # Add UMAP if computed 
    try:
        adata_emb.obsm['X_umap'] = embedding_2d
    except NameError:
        pass  
    
    adata_emb.write_h5ad(output_dir / "scgpt_embeddings.h5ad")
    print(f"  Saved AnnData with embeddings: {output_dir / 'scgpt_embeddings.h5ad'}")
    print(f"  Use this for trajectory analysis in Scanpy/Monocle")
except Exception as e:
    print(f"  Could not export AnnData: {e}")


# ZONE MARKER GENE ANALYSIS


print("\n" + "=" * 60)
print("Endothelial Zone Marker Genes")
print("=" * 60)
ZONE_MARKERS = {
    "Sinusoidal_Endothelial": ["FCN3", "LYVE1", "STAB2", "FCGR2B", "CLEC4G", "CRHBP", "FLT4", "FCN2", "CLEC1B", "DNASE1L3"],
    "Vascular_Portal_Endothelial": ["CLEC14A", "TM4SF1", "ID1", "PECAM1", "DLL4", "MGP", "SPARCL1", "SLC9A3R2", "RAMP2", "VWF"],
    "Vascular_Central_Venous_Endothelial": ["PRSS23", "SELE", "RSPO3", "LXH6", "RAMP3", "LIFR", "LYPD2", "NTS", "IFI27", "DNASE1L3"],
}
print("\nZone marker genes (from Azimuth reference):")
for zone, markers in ZONE_MARKERS.items():
    present = [m for m in markers if m in gene_names]
    print(f"  {zone}:")
    print(f"    Present in data: {len(present)}/{len(markers)}")
    print(f"    Markers: {', '.join(present[:5])}...")
with open(output_dir / "zone_markers.json", 'w') as f:
    json.dump(ZONE_MARKERS, f, indent=2)
print("\n" + "=" * 60)
print("Training Complete!")
print("=" * 60)
print(f"\nOutput files saved to: {output_dir}")
print("""
Files generated:
  - best_model.weights.h5    : Best model checkpoint
  - final_model.weights.h5   : Final model checkpoint
  - cell_embeddings.npy      : Cell embeddings for downstream analysis
  - umap_coordinates.npy     : 2D UMAP coordinates
  - umap_embeddings.png      : UMAP visualization
  - scgpt_embeddings.h5ad    : AnnData for trajectory analysis
  - config.json              : Training configuration
  - label_mapping.json       : Label to name mapping
  - zone_markers.json        : Endothelial zone markers

TensorBoard:
  tensorboard --logdir={output_dir}/logs --port=6006
""")
gc.collect()
