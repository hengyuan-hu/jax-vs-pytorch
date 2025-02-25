import math
import torch
import torch.nn as nn
import einops
from config import ModelConfig

from torch import Tensor
# from xformers.ops import memory_efficient_attention, unbind, fmha
# import xformers.ops as xops


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


from torch.nn.functional import scaled_dot_product_attention
from torch.nn.attention import SDPBackend, sdpa_kernel


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
            # with torch.backends.cuda.sdp_kernel(enable_math=False, enable_mem_efficient=True):
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                attn_v = torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, dropout_p=0.0, is_causal=True
                )
            # attn_v = memory_efficient_attention(
            #     q, k, v, #dropout_p=0.0, is_causal=True
            # )
        else:
            softmax_scale = 1.0 / math.sqrt(self.d_head)
            score = torch.einsum("bhtd,bhsd->bhts", q, k) * softmax_scale
            # score: (batch, num_head, seq, seq)

            if self.causal:
                seq_len = q.size(2)
                causal_mask = torch.triu(
                    torch.full((seq_len, seq_len), float("-inf"), device=score.device), diagonal=1
                )  # set lower_tri & diagnal = 0
                score = score + causal_mask

            attn = score.softmax(dim=-1)
            attn_v = torch.einsum("bhts,bhsd->bhtd", attn, v)

        attn_v = einops.rearrange(attn_v, "b h t d -> t b (h d)")
        return self.out_proj(attn_v)


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = (
            self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        )

        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        attn = q @ k.transpose(-2, -1)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MemEffAttention(Attention):
    def forward(self, x: Tensor, attn_bias=None) -> Tensor:
        # print("here")
        # return super().forward(x)
        # if not XFORMERS_AVAILABLE:
        #     assert attn_bias is None, "xFormers is required for nested tensors usage"
        #     return super().forward(x)
        # print(x.size())
        # seq_len = 1
        # seq_len = x.size(1)
        # attn_bias = torch.triu(
        #     torch.full((seq_len, seq_len), -1e6, device=x.device, dtype=x.dtype), diagonal=1
        # )
        attn_bias=xops.LowerTriangularMask()

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        q, k, v = unbind(qkv, 2)

        # if attn_bias is not None:
        # self_att_op = fmha.MemoryEfficientAttentionFlashAttentionOp
        self_att_op = fmha.MemoryEfficientAttentionFlashAttentionOp
        # else:
        # self_att_op = None
        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias, op=self_att_op)
        x = x.reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class TransformerLayer(nn.Module):
    def __init__(self, d_model, num_head, ff_dim, dropout, flash, use_xformer):
        super().__init__()
        if use_xformer:
            self.mha = MemEffAttention(d_model, num_head)
        else:
            self.mha = MultiHeadAttention(d_model, num_head, causal=True, flash=flash)

        self.layer_norm1 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, ff_dim)
        self.linear2 = nn.Linear(ff_dim, d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)

        self.dropout = dropout

    def forward(self, x):
        # x = x + nn.functional.dropout(self.mha(x), p=self.dropout)
        # x = x + nn.functional.dropout(self._ff_block(x), p=self.dropout)
        x = x + nn.functional.dropout(self.mha(self.layer_norm1(x)), p=self.dropout)
        x = x + nn.functional.dropout(self._ff_block(self.layer_norm2(x)), p=self.dropout)
        return x

    def _ff_block(self, x):
        x = nn.functional.dropout(nn.functional.relu(self.linear1(x)), self.dropout)
        x = self.linear2(x)
        return x


class HandCraftLM(nn.Module):
    def __init__(self, cfg: ModelConfig, use_flash, use_xformer):
        super().__init__()
        self.cfg = cfg
        self.use_flash = use_flash
        self.use_xformer = use_xformer

        self.positional_encoding = nn.Parameter(torch.empty(cfg.seq_len, 1, cfg.d_model))
        nn.init.normal_(self.positional_encoding)
        self.byte_embedding = nn.Embedding(num_embeddings=256, embedding_dim=cfg.d_model)

        self.transformer = nn.Sequential(*[
            TransformerLayer(cfg.d_model, cfg.num_heads, cfg.ff_dim, cfg.dropout, use_flash, use_xformer)
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
        if self.use_xformer:
            x = x.transpose(0,1)

        for layer in self.transformer:
            x = layer(x)

        if self.use_xformer:
            x = x.transpose(0,1)
        return self.prob_decoder(x)
