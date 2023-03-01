import math
import torch
import torch.nn as nn

import einops
from config import ModelConfig


class TorchLM(nn.Module):
    """LM that uses pytorch's Transformer classes"""
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        self.positional_encoding = nn.Parameter(torch.empty(cfg.seq_len, 1, cfg.d_model))
        nn.init.normal_(self.positional_encoding)
        self.byte_embedding = nn.Embedding(
            num_embeddings=256, embedding_dim=cfg.d_model
        )
        t_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.num_heads,
            dim_feedforward=cfg.ff_dim,
            batch_first=False,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer=t_layer, num_layers=cfg.n_layers
        )
        self.prob_decoder = nn.Linear(in_features=cfg.d_model, out_features=256)

    def forward(self, text_batch):
        batch_size = text_batch.shape[1]
        # Shift input right so causality isn't violated
        embeddings = self.byte_embedding(text_batch.int())

        zeros = torch.zeros(1, batch_size, self.cfg.d_model, device=text_batch.device)
        embeddings = torch.cat([zeros, embeddings[:-1, :, :]], axis=0)  # type: ignore
        embeddings = embeddings + self.positional_encoding
        embeddings = nn.functional.dropout(embeddings, p=self.cfg.dropout)

        mask = nn.Transformer.generate_square_subsequent_mask(self.cfg.seq_len).to(embeddings)
        output_probabilities = self.prob_decoder(self.transformer(embeddings, mask=mask))
        return output_probabilities


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_head, causal, flash):
        super().__init__()
        assert d_model % num_head == 0

        self.d_model = d_model
        self.num_head = num_head
        self.d_head = d_model // num_head
        self.causal = causal
        self.flash = flash

        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        """
        x: [seq_len, batch, d_model]
        """
        qkv = self.qkv_proj(x)
        q, k, v = einops.rearrange(
            qkv, "t b (k h d) -> b k h t d", k=3, h=self.num_head
        ).unbind(1)
        # q, k, v: (batch, num_head, seq, d_head)
        if self.flash:
            # force flash attention, it will raise error if flash cannot be applied
            with torch.backends.cuda.sdp_kernel(enable_math=False, enable_mem_efficient=False):
                attn_v = torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, dropout_p=0.0, is_causal=True
                )
        else:
            softmax_scale = 1.0 / math.sqrt(self.d_head)
            score = torch.einsum("bhtd,bhsd->bhts", q, k) * softmax_scale
            # score: (batch, num_head, seq, seq)

            if self.causal:
                seq_len = q.size(2)
                causal_mask = torch.triu(
                    torch.full((seq_len, seq_len), -1e6, device=score.device), diagonal=1
                )  # set lower_tri & diagnal = 0
                score = score + causal_mask

            attn = score.softmax(dim=-1)
            attn_v = torch.einsum("bhts,bhsd->bhtd", attn, v)

        attn_v = einops.rearrange(attn_v, "b h t d -> t b (h d)")
        return self.out_proj(attn_v)


class TransformerLayer(nn.Module):
    def __init__(self, d_model, num_head, ff_dim, dropout, flash):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, num_head, causal=True, flash=flash)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, ff_dim)
        self.linear2 = nn.Linear(ff_dim, d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)

        self.dropout = dropout

    def forward(self, x):
        x = x + nn.functional.dropout(self.mha(self.layer_norm1(x)), p=self.dropout)
        x = x + nn.functional.dropout(self._ff_block(self.layer_norm2(x)), p=self.dropout)
        return x

    def _ff_block(self, x):
        x = nn.functional.dropout(nn.functional.relu(self.linear1(x)), self.dropout)
        x = self.linear2(x)
        return x


class HandCraftLM(nn.Module):
    def __init__(self, cfg: ModelConfig, use_flash):
        super().__init__()
        self.cfg = cfg

        self.positional_encoding = nn.Parameter(torch.empty(cfg.seq_len, 1, cfg.d_model))
        nn.init.normal_(self.positional_encoding)
        self.byte_embedding = nn.Embedding(num_embeddings=256, embedding_dim=cfg.d_model)

        self.transformer = nn.Sequential(*[
            TransformerLayer(cfg.d_model, cfg.num_heads, cfg.ff_dim, cfg.dropout, use_flash)
            for _ in range(cfg.n_layers)
        ])
        self.prob_decoder = nn.Linear(in_features=cfg.d_model, out_features=256)

    def forward(self, text_batch):
        batch_size = text_batch.shape[1]
        # shift input right so causality isn't violated
        embeddings = self.byte_embedding(text_batch.int())

        zeros = torch.zeros(1, batch_size, self.cfg.d_model, device=text_batch.device)
        embeddings = torch.cat([zeros, embeddings[:-1, :, :]], axis=0)  # type: ignore
        embeddings = embeddings + self.positional_encoding
        embeddings = torch.nn.functional.dropout(embeddings, p=self.cfg.dropout)

        x = embeddings
        for layer in self.transformer:
            x = layer(x)
        return self.prob_decoder(x)
