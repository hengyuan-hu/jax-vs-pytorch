from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Configuration for the language models."""

    seq_len: int
    n_layers: int
    d_model: int
    num_heads: int
    ff_dim: int
    dropout: float

    batch_size: int
    learning_rate: float
