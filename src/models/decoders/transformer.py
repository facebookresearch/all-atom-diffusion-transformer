"""Copyright (c) Meta Platforms, Inc. and affiliates."""

import math
from typing import Any, Dict, Tuple

import torch
from torch import nn
from torch_geometric.utils import to_dense_batch
from torch_scatter import scatter


def get_index_embedding(indices, emb_dim, max_len=2048):
    """Creates sine / cosine positional embeddings from a prespecified indices.

    Args:
        indices: offsets of size [..., num_tokens] of type integer
        emb_dim: dimension of the embeddings to create
        max_len: maximum length

    Returns:
        positional embedding of shape [..., num_tokens, emb_dim]
    """
    K = torch.arange(emb_dim // 2, device=indices.device)
    pos_embedding_sin = torch.sin(
        indices[..., None] * math.pi / (max_len ** (2 * K[None] / emb_dim))
    ).to(indices.device)
    pos_embedding_cos = torch.cos(
        indices[..., None] * math.pi / (max_len ** (2 * K[None] / emb_dim))
    ).to(indices.device)
    pos_embedding = torch.cat([pos_embedding_sin, pos_embedding_cos], axis=-1)
    return pos_embedding


class TransformerDecoder(nn.Module):
    """Transformer decoder as part of pure Transformer-based VAEs.

    See src/models/encoders/transformer.py for documentation.
    """

    def __init__(
        self,
        max_num_elements=100,
        d_model: int = 1024,
        nhead: int = 8,
        dim_feedforward: int = 2048,
        activation: str = "gelu",
        dropout: float = 0.0,
        norm_first: bool = True,
        bias: bool = True,
        num_layers: int = 6,
    ):
        super().__init__()

        self.max_num_elements = max_num_elements
        self.d_model = d_model
        self.num_layers = num_layers

        activation = {
            "gelu": nn.GELU(approximate="tanh"),
            "relu": nn.ReLU(),
        }[activation]
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                activation=activation,
                dropout=dropout,
                batch_first=True,
                norm_first=norm_first,
                bias=bias,
            ),
            norm=nn.LayerNorm(d_model),
            num_layers=num_layers,
        )

        self.atom_types_head = nn.Linear(d_model, max_num_elements, bias=True)
        self.pos_head = nn.Linear(d_model, 3, bias=False)
        self.frac_coords_head = nn.Linear(d_model, 3, bias=False)
        self.lattice_head = nn.Linear(d_model, 6, bias=False)

    def forward(self, encoded_batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Args:
            encoded_batch: Dict with the following attributes:
                x (torch.Tensor): Encoded batch of atomic environments
                num_atoms (torch.Tensor): Number of atoms in each molecular environment
                batch (torch.Tensor): Batch index for each atom
                token_idx (torch.Tensor): Token index for each atom
        """
        x = encoded_batch["x"]

        # Positional embedding
        x += get_index_embedding(encoded_batch["token_idx"], self.d_model)

        # Convert from PyG batch to dense batch with padding
        x, token_mask = to_dense_batch(x, encoded_batch["batch"])

        # Transformer forward pass
        x = self.transformer.forward(x, src_key_padding_mask=(~token_mask))
        x = x[token_mask]

        # Global pooling: (n, d) -> (bsz, d)
        x_global = scatter(x, encoded_batch["batch"], dim=0, reduce="mean")

        # Atomic type prediction head
        atom_types_out = self.atom_types_head(x)

        # Lattice lengths and angles prediction head
        lattices_out = self.lattice_head(x_global)

        # Fractional coordinates prediction head
        frac_coords_out = self.frac_coords_head(x)

        # Cartesian coordinates prediction head
        pos_out = self.pos_head(x)

        return {
            "atom_types": atom_types_out,
            "lattices": lattices_out,
            "lengths": lattices_out[:, :3],
            "angles": lattices_out[:, 3:],
            "frac_coords": frac_coords_out,
            "pos": pos_out,
        }
