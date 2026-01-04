

import json
import math
from typing import Dict, Optional, Any, Union, List
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# ==========================================
# VOCABULARY
# ==========================================

class GeneVocab:
    """
    Gene vocabulary class compatible with TensorFlow workflow.
    Matches the scGPT vocabulary format.
    """
    
    def __init__(self, gene_to_idx: Dict[str, int]):
        self.gene_to_idx = gene_to_idx
        self.idx_to_gene = {v: k for k, v in gene_to_idx.items()}
        self.default_idx = 0
        
    def __getitem__(self, gene: str) -> int:
        return self.gene_to_idx.get(gene, self.default_idx)
    
    def __len__(self) -> int:
        return len(self.gene_to_idx)
    
    def __contains__(self, gene: str) -> bool:
        return gene in self.gene_to_idx
    
    def set_default_index(self, idx: int):
        self.default_idx = idx
        
    @classmethod
    def from_file(cls, filepath: str) -> "GeneVocab":
        with open(filepath, 'r') as f:
            gene_to_idx = json.load(f)
        return cls(gene_to_idx)
    
    def save(self, filepath: str):
        with open(filepath, 'w') as f:
            json.dump(self.gene_to_idx, f, indent=2)


# ==========================================
# TOKENIZATION UTILITIES
# ==========================================

def tokenize_and_pad_batch(
    data: np.ndarray,
    gene_ids: np.ndarray,
    max_len: int,
    vocab: GeneVocab,
    pad_token: str = "<pad>",
    pad_value: int = -2,
    append_cls: bool = True,
    cls_token: str = "<cls>",
    include_zero_gene: bool = False,
) -> Dict[str, np.ndarray]:
    """
    Tokenize and pad a batch of gene expression data.
    
    Args:
        data: Expression values, shape (batch_size, n_genes)
        gene_ids: Gene token IDs, shape (n_genes,) or (batch_size, n_genes)
        max_len: Maximum sequence length
        vocab: Gene vocabulary
        pad_token: Padding token string
        pad_value: Value to use for padding
        append_cls: Whether to append CLS token at the beginning
        cls_token: CLS token string
        include_zero_gene: Whether to include genes with zero expression
        
    Returns:
        Dictionary with 'genes', 'values', and 'padding_mask'
    """
    batch_size = data.shape[0]
    pad_id = vocab[pad_token]
    cls_id = vocab[cls_token] if append_cls else None
    
    # Handle gene_ids shape
    if gene_ids.ndim == 1:
        gene_ids = np.tile(gene_ids, (batch_size, 1))
    
    genes_list = []
    values_list = []
    
    for i in range(batch_size):
        if include_zero_gene:
            # Include all genes
            genes = gene_ids[i]
            values = data[i]
        else:
            # Only include non-zero genes
            nonzero_mask = data[i] != 0
            genes = gene_ids[i][nonzero_mask]
            values = data[i][nonzero_mask]
        
        # Truncate if too long
        if len(genes) > max_len - (1 if append_cls else 0):
            genes = genes[:max_len - (1 if append_cls else 0)]
            values = values[:max_len - (1 if append_cls else 0)]
        
        # Append CLS token
        if append_cls:
            genes = np.concatenate([[cls_id], genes])
            values = np.concatenate([[0], values])  # CLS token has value 0
        
        genes_list.append(genes)
        values_list.append(values)
    
    # Pad sequences
    max_seq_len = max(len(g) for g in genes_list)
    max_seq_len = min(max_seq_len, max_len)
    
    genes_padded = np.full((batch_size, max_seq_len), pad_id, dtype=np.int32)
    values_padded = np.full((batch_size, max_seq_len), pad_value, dtype=np.float32)
    padding_mask = np.ones((batch_size, max_seq_len), dtype=bool)  # True = padded
    
    for i in range(batch_size):
        seq_len = min(len(genes_list[i]), max_seq_len)
        genes_padded[i, :seq_len] = genes_list[i][:seq_len]
        values_padded[i, :seq_len] = values_list[i][:seq_len]
        padding_mask[i, :seq_len] = False  # Not padded
    
    return {
        "genes": genes_padded,
        "values": values_padded,
        "padding_mask": padding_mask,
    }


def random_mask_value(
    values: np.ndarray,
    mask_ratio: float = 0.15,
    mask_value: int = -1,
    pad_value: int = -2,
) -> tuple:
    """
    Randomly mask values for masked value prediction task.
    
    Args:
        values: Expression values, shape (batch_size, seq_len)
        mask_ratio: Ratio of values to mask
        mask_value: Value to use for masked positions
        pad_value: Padding value (will not be masked)
        
    Returns:
        Tuple of (masked_values, mask_positions)
    """
    masked_values = values.copy()
    mask_positions = np.zeros_like(values, dtype=bool)
    
    for i in range(values.shape[0]):
        # Find non-padding positions
        non_pad_mask = values[i] != pad_value
        non_pad_indices = np.where(non_pad_mask)[0]
        
        if len(non_pad_indices) > 0:
            # Randomly select positions to mask
            n_mask = max(1, int(len(non_pad_indices) * mask_ratio))
            mask_indices = np.random.choice(non_pad_indices, n_mask, replace=False)
            
            masked_values[i, mask_indices] = mask_value
            mask_positions[i, mask_indices] = True
    
    return masked_values, mask_positions


# ==========================================
# MODEL LAYERS
# ==========================================

class GeneEncoder(layers.Layer):
    """Encode gene tokens into embeddings."""
    
    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None, **kwargs):
        super().__init__(**kwargs)
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        
    def build(self, input_shape):
        self.embedding = layers.Embedding(
            input_dim=self.num_embeddings,
            output_dim=self.embedding_dim,
            mask_zero=(self.padding_idx == 0) if self.padding_idx is not None else False,
        )
        self.enc_norm = layers.LayerNormalization(epsilon=1e-5)
        super().build(input_shape)
        
    def call(self, x: tf.Tensor) -> tf.Tensor:
        x = self.embedding(x)
        x = self.enc_norm(x)
        return x


class ContinuousValueEncoder(layers.Layer):
    """Encode continuous expression values into embeddings."""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_value: int = 512, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.dropout_rate = dropout
        self.max_value = max_value
        
    def build(self, input_shape):
        self.linear1 = layers.Dense(self.d_model)
        self.activation = layers.ReLU()
        self.linear2 = layers.Dense(self.d_model)
        self.norm = layers.LayerNormalization(epsilon=1e-5)
        self.dropout = layers.Dropout(self.dropout_rate)
        super().build(input_shape)
        
    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        x = tf.expand_dims(x, -1)
        x = tf.clip_by_value(x, clip_value_min=-float('inf'), clip_value_max=float(self.max_value))
        x = self.activation(self.linear1(x))
        x = self.linear2(x)
        x = self.norm(x)
        return self.dropout(x, training=training)


class CategoryValueEncoder(layers.Layer):
    """Encode categorical (binned) values into embeddings."""
    
    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None, **kwargs):
        super().__init__(**kwargs)
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        
    def build(self, input_shape):
        self.embedding = layers.Embedding(input_dim=self.num_embeddings, output_dim=self.embedding_dim)
        self.enc_norm = layers.LayerNormalization(epsilon=1e-5)
        super().build(input_shape)
        
    def call(self, x: tf.Tensor) -> tf.Tensor:
        x = tf.cast(x, tf.int32)
        x = self.embedding(x)
        x = self.enc_norm(x)
        return x


class BatchLabelEncoder(layers.Layer):
    """Encode batch labels into embeddings for batch correction."""
    
    def __init__(self, num_embeddings: int, embedding_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        
    def build(self, input_shape):
        self.embedding = layers.Embedding(input_dim=self.num_embeddings, output_dim=self.embedding_dim)
        self.enc_norm = layers.LayerNormalization(epsilon=1e-5)
        super().build(input_shape)
        
    def call(self, x: tf.Tensor) -> tf.Tensor:
        x = self.embedding(x)
        x = self.enc_norm(x)
        return x


class TransformerEncoderLayer(layers.Layer):
    """A single Transformer encoder layer with pre-norm or post-norm."""
    
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, 
                 dropout: float = 0.1, pre_norm: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.dropout_rate = dropout
        self.pre_norm = pre_norm
        
    def build(self, input_shape):
        self.self_attn = layers.MultiHeadAttention(
            num_heads=self.nhead,
            key_dim=self.d_model // self.nhead,
            dropout=self.dropout_rate,
        )
        self.ffn = keras.Sequential([
            layers.Dense(self.dim_feedforward, activation='gelu'),
            layers.Dropout(self.dropout_rate),
            layers.Dense(self.d_model),
            layers.Dropout(self.dropout_rate),
        ])
        self.norm1 = layers.LayerNormalization(epsilon=1e-5)
        self.norm2 = layers.LayerNormalization(epsilon=1e-5)
        self.dropout1 = layers.Dropout(self.dropout_rate)
        super().build(input_shape)
        
    def call(self, src: tf.Tensor, attention_mask: Optional[tf.Tensor] = None, training: bool = False) -> tf.Tensor:
        if self.pre_norm:
            # Pre-norm: norm -> attention -> residual
            src2 = self.norm1(src)
            src2 = self.self_attn(src2, src2, src2, attention_mask=attention_mask, training=training)
            src = src + self.dropout1(src2, training=training)
            src2 = self.norm2(src)
            src = src + self.ffn(src2, training=training)
        else:
            # Post-norm: attention -> residual -> norm
            src2 = self.self_attn(src, src, src, attention_mask=attention_mask, training=training)
            src = self.norm1(src + self.dropout1(src2, training=training))
            src = self.norm2(src + self.ffn(src, training=training))
        return src


class TransformerEncoder(layers.Layer):
    """Stack of Transformer encoder layers."""
    
    def __init__(self, d_model: int, nhead: int, num_layers: int, dim_feedforward: int = 2048,
                 dropout: float = 0.1, pre_norm: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout_rate = dropout
        self.pre_norm = pre_norm
        
    def build(self, input_shape):
        self.layers_list = [
            TransformerEncoderLayer(
                d_model=self.d_model, nhead=self.nhead, dim_feedforward=self.dim_feedforward,
                dropout=self.dropout_rate, pre_norm=self.pre_norm, name=f"layer_{i}"
            ) for i in range(self.num_layers)
        ]
        super().build(input_shape)
        
    def call(self, src: tf.Tensor, attention_mask: Optional[tf.Tensor] = None, training: bool = False) -> tf.Tensor:
        output = src
        for layer in self.layers_list:
            output = layer(output, attention_mask=attention_mask, training=training)
        return output


class ExprDecoder(layers.Layer):
    """Decoder for masked expression value prediction (GEPC objective)."""
    
    def __init__(self, d_model: int, explicit_zero_prob: bool = False, use_batch_labels: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.explicit_zero_prob = explicit_zero_prob
        self.use_batch_labels = use_batch_labels
        self.d_in = d_model * 2 if use_batch_labels else d_model
        
    def build(self, input_shape):
        self.fc = keras.Sequential([
            layers.Dense(self.d_model),
            layers.LeakyReLU(negative_slope=0.01),
            layers.Dense(self.d_model),
            layers.LeakyReLU(negative_slope=0.01),
            layers.Dense(1),
        ])
        if self.explicit_zero_prob:
            self.zero_logit = keras.Sequential([
                layers.Dense(self.d_model),
                layers.LeakyReLU(negative_slope=0.01),
                layers.Dense(self.d_model),
                layers.LeakyReLU(negative_slope=0.01),
                layers.Dense(1),
            ])
        super().build(input_shape)
        
    def call(self, x: tf.Tensor) -> Dict[str, tf.Tensor]:
        pred_value = tf.squeeze(self.fc(x), axis=-1)
        if not self.explicit_zero_prob:
            return {"pred": pred_value}
        zero_logits = tf.squeeze(self.zero_logit(x), axis=-1)
        zero_probs = tf.sigmoid(zero_logits)
        return {"pred": pred_value, "zero_probs": zero_probs}


class ClsDecoder(layers.Layer):
    """Decoder for cell-type classification."""
    
    def __init__(self, d_model: int, n_cls: int, nlayers: int = 3, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.n_cls = n_cls
        self.nlayers = nlayers
        
    def build(self, input_shape):
        self.hidden_layers = []
        for i in range(self.nlayers - 1):
            self.hidden_layers.append(layers.Dense(self.d_model))
            self.hidden_layers.append(layers.ReLU())
            self.hidden_layers.append(layers.LayerNormalization(epsilon=1e-5))
        self.out_layer = layers.Dense(self.n_cls)
        super().build(input_shape)
        
    def call(self, x: tf.Tensor) -> tf.Tensor:
        for layer in self.hidden_layers:
            x = layer(x)
        return self.out_layer(x)


class AdversarialDiscriminator(layers.Layer):
    """Discriminator for domain adversarial batch correction (DAB objective)."""
    
    def __init__(self, d_model: int, n_cls: int, nlayers: int = 3, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.n_cls = n_cls
        self.nlayers = nlayers
        
    def build(self, input_shape):
        self.hidden_layers = []
        for i in range(self.nlayers - 1):
            self.hidden_layers.append(layers.Dense(self.d_model))
            self.hidden_layers.append(layers.LeakyReLU(negative_slope=0.01))
            self.hidden_layers.append(layers.LayerNormalization(epsilon=1e-5))
        self.out_layer = layers.Dense(self.n_cls)
        super().build(input_shape)
        
    def call(self, x: tf.Tensor, reverse_grad: bool = True) -> tf.Tensor:
        # Gradient reversal for adversarial training
        if reverse_grad:
            x = gradient_reverse(x)
        for layer in self.hidden_layers:
            x = layer(x)
        return self.out_layer(x)


@tf.custom_gradient
def gradient_reverse(x):
    """Gradient reversal layer for adversarial training."""
    def grad(dy):
        return -dy
    return x, grad


# ==========================================
# MAIN MODEL
# ==========================================

class scGPTModel(keras.Model):
    """
    TensorFlow implementation of scGPT Transformer Model.
    Based on the official scGPT architecture for single-cell analysis.
    
    Supports objectives:
    - GEPC: Gene Expression Prediction for Cell (masked value prediction)
    - CLS: Cell-type classification
    - ECS: Elastic Cell Similarity
    - DAB: Domain Adversarial Batch correction
    """
    
    def __init__(
        self,
        ntoken: int,
        d_model: int = 512,
        nhead: int = 8,
        d_hid: int = 512,
        nlayers: int = 12,
        nlayers_cls: int = 3,
        n_cls: int = 2,
        vocab: Any = None,
        dropout: float = 0.2,
        pad_token: str = "<pad>",
        pad_value: int = -2,
        use_batch_labels: bool = False,
        num_batch_labels: Optional[int] = None,
        input_emb_style: str = "continuous",
        n_input_bins: Optional[int] = None,
        cell_emb_style: str = "cls",
        explicit_zero_prob: bool = False,
        do_dab: bool = False,
        ecs_threshold: float = 0.8,
        pre_norm: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.model_type = "Transformer"
        self.d_model = d_model
        self.nhead = nhead
        self.use_batch_labels = use_batch_labels
        self.input_emb_style = input_emb_style
        self.cell_emb_style = cell_emb_style
        self.n_cls = n_cls
        self.vocab = vocab
        self.pad_value = pad_value
        self.explicit_zero_prob = explicit_zero_prob
        self.do_dab = do_dab
        self.ecs_threshold = ecs_threshold
        
        # Gene Encoder
        padding_idx = vocab[pad_token] if vocab is not None else 0
        self.encoder = GeneEncoder(num_embeddings=ntoken, embedding_dim=d_model, padding_idx=padding_idx)
        
        # Value Encoder
        if input_emb_style == "continuous":
            self.value_encoder = ContinuousValueEncoder(d_model=d_model, dropout=dropout)
        elif input_emb_style == "category":
            assert n_input_bins is not None and n_input_bins > 0
            self.value_encoder = CategoryValueEncoder(num_embeddings=n_input_bins, embedding_dim=d_model)
        else:
            self.value_encoder = None
            
        # Batch Label Encoder
        if use_batch_labels:
            self.batch_encoder = BatchLabelEncoder(num_embeddings=num_batch_labels, embedding_dim=d_model)
        
        # Transformer Encoder
        self.transformer_encoder = TransformerEncoder(
            d_model=d_model, nhead=nhead, num_layers=nlayers,
            dim_feedforward=d_hid, dropout=dropout, pre_norm=pre_norm
        )
        
        # Decoders
        self.expr_decoder = ExprDecoder(
            d_model=d_model, explicit_zero_prob=explicit_zero_prob, use_batch_labels=use_batch_labels
        )
        self.cls_decoder = ClsDecoder(d_model=d_model, n_cls=n_cls, nlayers=nlayers_cls)
        
        # Domain Adversarial Discriminator
        if do_dab and num_batch_labels is not None:
            self.dab_discriminator = AdversarialDiscriminator(d_model=d_model, n_cls=num_batch_labels)
        
        self._config = {
            "ntoken": ntoken, "d_model": d_model, "nhead": nhead, "d_hid": d_hid,
            "nlayers": nlayers, "nlayers_cls": nlayers_cls, "n_cls": n_cls,
            "dropout": dropout, "pad_token": pad_token, "pad_value": pad_value,
            "use_batch_labels": use_batch_labels, "num_batch_labels": num_batch_labels,
            "input_emb_style": input_emb_style, "n_input_bins": n_input_bins,
            "cell_emb_style": cell_emb_style, "explicit_zero_prob": explicit_zero_prob,
            "do_dab": do_dab, "ecs_threshold": ecs_threshold, "pre_norm": pre_norm,
        }
    
    def _encode(self, src: tf.Tensor, values: tf.Tensor, src_key_padding_mask: Optional[tf.Tensor] = None,
                batch_labels: Optional[tf.Tensor] = None, training: bool = False) -> tf.Tensor:
        """Encode input through transformer."""
        src_emb = self.encoder(src)
        
        if self.value_encoder is not None:
            values_emb = self.value_encoder(values, training=training)
            total_embs = src_emb + values_emb
        else:
            values_expanded = tf.expand_dims(values, -1)
            total_embs = src_emb * values_expanded
        
        # Convert padding mask for attention
        attention_mask = None
        if src_key_padding_mask is not None:
            attention_mask = tf.cast(~src_key_padding_mask, tf.float32)
            attention_mask = attention_mask[:, tf.newaxis, tf.newaxis, :]
        
        output = self.transformer_encoder(total_embs, attention_mask=attention_mask, training=training)
        return output
    
    def _get_cell_emb(self, layer_output: tf.Tensor, weights: Optional[tf.Tensor] = None) -> tf.Tensor:
        """Extract cell embedding from transformer output."""
        if self.cell_emb_style == "cls":
            return layer_output[:, 0, :]
        elif self.cell_emb_style == "avg-pool":
            return tf.reduce_mean(layer_output, axis=1)
        elif self.cell_emb_style == "w-pool":
            if weights is None:
                raise ValueError("weights required for w-pool")
            weights = tf.expand_dims(weights, -1)
            cell_emb = tf.reduce_sum(layer_output * weights, axis=1)
            return tf.math.l2_normalize(cell_emb, axis=1)
        return layer_output[:, 0, :]
    
    def call(self, src: tf.Tensor, values: tf.Tensor, src_key_padding_mask: Optional[tf.Tensor] = None,
             batch_labels: Optional[tf.Tensor] = None, CLS: bool = False, MVC: bool = False,
             ECS: bool = False, do_sample: bool = False, training: bool = False) -> Dict[str, tf.Tensor]:
        """
        Forward pass with multiple objectives.
        
        Args:
            src: Gene token IDs, shape (batch, seq_len)
            values: Expression values, shape (batch, seq_len)
            src_key_padding_mask: Padding mask, shape (batch, seq_len)
            batch_labels: Batch labels for batch correction
            CLS: Compute classification output
            MVC: Compute masked value prediction (cell embedding based)
            ECS: Compute elastic cell similarity loss
            training: Training mode
        """
        transformer_output = self._encode(src, values, src_key_padding_mask, batch_labels, training=training)
        
        output = {}
        
        # Batch embedding for decoders
        if self.use_batch_labels and batch_labels is not None:
            batch_emb = self.batch_encoder(batch_labels)
            batch_emb_expanded = tf.tile(tf.expand_dims(batch_emb, 1), [1, tf.shape(transformer_output)[1], 1])
            decoder_input = tf.concat([transformer_output, batch_emb_expanded], axis=-1)
        else:
            decoder_input = transformer_output
            
        # Expression prediction (GEPC)
        mlm_output = self.expr_decoder(decoder_input)
        output["mlm_output"] = mlm_output["pred"]
        if self.explicit_zero_prob:
            output["mlm_zero_probs"] = mlm_output.get("zero_probs")
        
        # Cell embedding
        cell_emb = self._get_cell_emb(transformer_output, values)
        output["cell_emb"] = cell_emb
        
        # Classification (CLS)
        if CLS:
            output["cls_output"] = self.cls_decoder(cell_emb)
        
        # Elastic Cell Similarity (ECS)
        if ECS:
            cell_emb_normed = tf.math.l2_normalize(cell_emb, axis=1)
            cos_sim = tf.matmul(cell_emb_normed, cell_emb_normed, transpose_b=True)
            # Mask diagonal
            mask = tf.eye(tf.shape(cos_sim)[0])
            cos_sim = cos_sim * (1 - mask)
            cos_sim = tf.nn.relu(cos_sim)
            output["loss_ecs"] = tf.reduce_mean(1 - (cos_sim - self.ecs_threshold) ** 2)
        
        # Domain Adversarial Batch (DAB)
        if self.do_dab and hasattr(self, 'dab_discriminator'):
            output["dab_output"] = self.dab_discriminator(cell_emb, reverse_grad=True)
            
        return output
    
    def get_config(self):
        config = super().get_config()
        config.update(self._config)
        return config


# ==========================================
# LOSS FUNCTIONS
# ==========================================

def masked_mse_loss(pred: tf.Tensor, target: tf.Tensor, mask: tf.Tensor) -> tf.Tensor:
    """
    Masked MSE loss for expression prediction.
    Only compute loss on masked positions.
    """
    mask = tf.cast(mask, tf.float32)
    loss = tf.square(pred - target) * mask
    return tf.reduce_sum(loss) / (tf.reduce_sum(mask) + 1e-8)


def masked_relative_error(pred: tf.Tensor, target: tf.Tensor, mask: tf.Tensor) -> tf.Tensor:
    """Masked relative error."""
    mask = tf.cast(mask, tf.float32)
    error = tf.abs(pred - target) / (tf.abs(target) + 1e-8) * mask
    return tf.reduce_sum(error) / (tf.reduce_sum(mask) + 1e-8)


def criterion_neg_log_bernoulli(pred_probs: tf.Tensor, target: tf.Tensor, mask: tf.Tensor) -> tf.Tensor:
    """Negative log Bernoulli loss for zero probability prediction."""
    mask = tf.cast(mask, tf.float32)
    target_binary = tf.cast(target == 0, tf.float32)
    loss = -target_binary * tf.math.log(pred_probs + 1e-8) - (1 - target_binary) * tf.math.log(1 - pred_probs + 1e-8)
    loss = loss * mask
    return tf.reduce_sum(loss) / (tf.reduce_sum(mask) + 1e-8)


# ==========================================
# FACTORY FUNCTION
# ==========================================

def create_scgpt_model(
    vocab_size: int,
    d_model: int = 512,
    nhead: int = 8,
    d_hid: int = 512,
    nlayers: int = 12,
    n_cls: int = 2,
    dropout: float = 0.2,
    use_batch_labels: bool = False,
    num_batch_labels: Optional[int] = None,
    vocab: Any = None,
    explicit_zero_prob: bool = False,
    do_dab: bool = False,
    ecs_threshold: float = 0.8,
    pre_norm: bool = False,
) -> scGPTModel:
    """Factory function to create an scGPT model."""
    return scGPTModel(
        ntoken=vocab_size, d_model=d_model, nhead=nhead, d_hid=d_hid,
        nlayers=nlayers, n_cls=n_cls, dropout=dropout,
        use_batch_labels=use_batch_labels, num_batch_labels=num_batch_labels,
        vocab=vocab, explicit_zero_prob=explicit_zero_prob,
        do_dab=do_dab, ecs_threshold=ecs_threshold, pre_norm=pre_norm,
    )
