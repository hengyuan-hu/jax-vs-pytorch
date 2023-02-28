import time
import numpy as np
import torch
import torch.nn as nn
from functools import partial

from config import ModelConfig
from data import Enwik9Loader

from flash_attn.modules.block import Block
from flash_attn.modules.mha import MHA
from flash_attn.modules.mlp import Mlp


class FlashLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        self.positional_encoding = nn.Parameter(
            torch.empty(cfg.seq_len, 1, cfg.d_model)
        )
        nn.init.normal_(self.positional_encoding)

        self.byte_embedding = nn.Embedding(
            num_embeddings=256, embedding_dim=cfg.d_model
        )

        self.transformer = Block(
            dim=cfg.d_model,
            prenorm=False,
            mixer_cls=partial(MHA, num_heads=cfg.num_heads, causal=True),
            mlp_cls=partial(Mlp, hidden_features=cfg.ff_dim),
            resid_dropout1=cfg.dropout,
            resid_dropout2=cfg.dropout,
            # residual_in_fp32=True,
        )
        self.prob_decoder = nn.Linear(in_features=cfg.d_model, out_features=256)

    def forward(self, text_batch):
        batch_size = text_batch.shape[0]
        # Shift input right so causality isn't violated
        embeddings = self.byte_embedding(text_batch.int())
        embeddings = torch.cat(
            [
                torch.zeros(batch_size, 1, self.cfg.d_model, device=text_batch.device),
                embeddings[:, :-1, :],
            ],
            axis=1,
        )  # type: ignore
        embeddings = nn.Dropout(p=self.cfg.dropout)(
            embeddings + self.positional_encoding
        )
        # mask = nn.Transformer.generate_square_subsequent_mask(self.cfg.seq_len).to(embeddings)

        # residual = None
        # for layer in self.transformer:
        #     embeddings, residual = layer(embeddings, residual)
        layer = self.transformer
        embeddings  = layer(embeddings)

        output_probabilities = self.prob_decoder(embeddings)
        return output_probabilities


def compute_loss(lm, batch):
    probs = lm(batch)
    probs = probs.reshape(-1, 256)
    targets = batch.reshape(-1).long()
    return nn.CrossEntropyLoss()(probs, targets)


def train_epoch(lm, cfg: ModelConfig, datapath: str) -> None:
    optimizer = torch.optim.Adam(lm.parameters(), cfg.learning_rate)
    losses = []
    t = time.time()
    for i, batch in enumerate(Enwik9Loader(cfg.batch_size, cfg.seq_len, datapath)):
        data = torch.tensor(batch, device="cuda")
        data = data.transpose(0, 1).contiguous()
        loss = compute_loss(lm, data)
        losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        log_per = 20
        if (i + 1) % log_per == 0:
            time_elps = time.time() - t
            print(
                f"At iter {i+1}, loss {np.mean(losses):.4f}, "
                f"recent loss {np.mean(losses[-20:]):.4f}, "
                f"Speed: {log_per * cfg.batch_size / time_elps:.2f}, time: {time_elps:.1f}"
            )
            t = time.time()
            losses = []


if __name__ == "__main__":
    from config import ModelConfig
    import torch

    enwik9 = "./enwik9"

    # This is the highest batch size PyTorch can handle, the JAX model can do 79
    cfg = ModelConfig(
        seq_len=256,
        n_layers=1,
        d_model=512,
        num_heads=8,
        ff_dim=3072,
        dropout=0.1,
        batch_size=100,
        learning_rate=1e-3,
    )

    lm = FlashLM(cfg).cuda()
    print(lm)
    for v in lm.parameters():
        print(f"{v.size()}".ljust(30), f"{abs(v.mean().item()):.2e}", f"{v.std().item():.2e}")

    train_epoch(lm, cfg, enwik9)
