"""Copyright (c) Meta Platforms, Inc. and affiliates."""

import math
from typing import Any, Dict, Tuple

import torch
from torch import nn
from torch_scatter import scatter

from src.models.components.equiformer_v2.layer_norm import (
    EquivariantLayerNormArray,
    EquivariantLayerNormArraySphericalHarmonics,
    EquivariantRMSNormArraySphericalHarmonics,
    EquivariantRMSNormArraySphericalHarmonicsV2,
)
from src.models.components.equiformer_v2.module_list import ModuleListInfo
from src.models.components.equiformer_v2.radial_function import RadialFunction
from src.models.components.equiformer_v2.so3 import (
    SO3_Embedding,
    SO3_Grid,
    SO3_LinearV2,
)
from src.models.components.equiformer_v2.transformer_block import FeedForwardNetwork


class FeedForwardDecoder(nn.Module):
    """Equivariant feedforward decoder as part of Equiformer-based VAEs.

    See src/models/encoders/equiformer.py for documentation.
    """

    def __init__(
        self,
        max_num_elements=90,
        sphere_channels=128,
        ffn_hidden_channels=512,
        lmax_list=[6],
        mmax_list=[2],
        grid_resolution=None,
        ffn_activation="scaled_silu",
        use_gate_act=False,
        use_grid_mlp=False,
        use_sep_s2_act=True,
        weight_init="normal",
    ):
        super().__init__()

        ###########################################
        # Initializations from EquiformerV2 code
        ###########################################

        self.max_num_elements = max_num_elements

        self.sphere_channels = sphere_channels
        self.ffn_hidden_channels = ffn_hidden_channels

        self.lmax_list = lmax_list
        self.mmax_list = mmax_list
        self.grid_resolution = grid_resolution

        self.ffn_activation = ffn_activation
        self.use_gate_act = use_gate_act
        self.use_grid_mlp = use_grid_mlp
        self.use_sep_s2_act = use_sep_s2_act

        self.weight_init = weight_init
        assert self.weight_init in ["normal", "uniform"]

        assert len(self.lmax_list) == 1, "Only one resolution is supported for now."
        self.d_model = self.sphere_channels * (self.lmax_list[0] + 1) ** 2

        # Initialize the transformations between spherical and grid representations
        self.SO3_grid = ModuleListInfo(f"({max(self.lmax_list)}, {max(self.lmax_list)})")
        for l in range(max(self.lmax_list) + 1):
            SO3_m_grid = nn.ModuleList()
            for m in range(max(self.lmax_list) + 1):
                SO3_m_grid.append(
                    SO3_Grid(l, m, resolution=self.grid_resolution, normalization="component")
                )
            self.SO3_grid.append(SO3_m_grid)

        # Atomic type prediction head
        self.atom_types_head = FeedForwardNetwork(
            self.sphere_channels,
            self.ffn_hidden_channels,
            self.max_num_elements,
            self.lmax_list,
            self.mmax_list,
            self.SO3_grid,
            self.ffn_activation,
            self.use_gate_act,
            self.use_grid_mlp,
            self.use_sep_s2_act,
        )

        # Cartesian coordinates prediction head
        self.pos_head = FeedForwardNetwork(
            self.sphere_channels,
            self.ffn_hidden_channels,
            1,  # single rank-1 tensor
            self.lmax_list,
            self.mmax_list,
            self.SO3_grid,
            self.ffn_activation,
            self.use_gate_act,
            self.use_grid_mlp,
            self.use_sep_s2_act,
        )

        # Fractional coordinates prediction head
        self.frac_coords_head = FeedForwardNetwork(
            self.sphere_channels,
            self.ffn_hidden_channels,
            3,  # three scalar values
            self.lmax_list,
            self.mmax_list,
            self.SO3_grid,
            self.ffn_activation,
            self.use_gate_act,
            self.use_grid_mlp,
            self.use_sep_s2_act,
        )

        # Lattice parameters prediction head
        self.lattice_head = FeedForwardNetwork(
            self.sphere_channels,
            self.ffn_hidden_channels,
            6,  # a, b, c, alpha, beta, gamma
            self.lmax_list,
            self.mmax_list,
            self.SO3_grid,
            self.ffn_activation,
            self.use_gate_act,
            self.use_grid_mlp,
            self.use_sep_s2_act,
        )

        self.apply(self._init_weights)
        self.apply(self._uniform_init_rad_func_linear_weights)

    def forward(self, encoded_batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Args:
            encoded_batch: Dict with the following attributes:
                x (torch.Tensor): Encoded batch of atomic environments
                num_atoms (torch.Tensor): Number of atoms in each molecular environment
                batch (torch.Tensor): Batch index for each atom
        """
        x = encoded_batch["x"]

        _x = SO3_Embedding(
            len(encoded_batch["num_atoms"]),
            self.lmax_list,
            self.sphere_channels,
            x.device,
            x.dtype,
        )
        _x.embedding = x.view(-1, (self.lmax_list[0] + 1) ** 2, self.sphere_channels)
        x = _x

        # Global pooling
        x_global = SO3_Embedding(
            len(encoded_batch["num_atoms"]),
            self.lmax_list,
            self.sphere_channels,
            x.embedding.device,
            x.embedding.dtype,
        )
        x_global.embedding = scatter(
            x.embedding, encoded_batch["batch"], dim=0, reduce="mean"
        )  # (n, d) -> (bsz, d)

        # Atomic type prediction head
        atom_types_out = self.atom_types_head(x).embedding.narrow(1, 0, 1).squeeze(1)

        # Lattice lengths and angles prediction head
        lattices_out = self.lattice_head(x_global).embedding.narrow(1, 0, 1).squeeze(1)

        # Fractional coordinates prediction head
        frac_coords_out = self.frac_coords_head(x).embedding.narrow(1, 0, 1).squeeze(1)

        # Cartesian coordinates prediction head
        pos_out = self.pos_head(x).embedding.narrow(1, 1, 3).squeeze(2)

        return {
            "atom_types": atom_types_out,
            "lattices": lattices_out,
            "lengths": lattices_out[:, :3],
            "angles": lattices_out[:, 3:],
            "frac_coords": frac_coords_out,
            "pos": pos_out,
        }

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def _init_weights(self, m):
        if isinstance(m, torch.nn.Linear) or isinstance(m, SO3_LinearV2):
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
            if self.weight_init == "normal":
                std = 1 / math.sqrt(m.in_features)
                torch.nn.init.normal_(m.weight, 0, std)

        elif isinstance(m, torch.nn.LayerNorm):
            torch.nn.init.constant_(m.bias, 0)
            torch.nn.init.constant_(m.weight, 1.0)

    def _uniform_init_rad_func_linear_weights(self, m):
        if isinstance(m, RadialFunction):
            m.apply(self._uniform_init_linear_weights)

    def _uniform_init_linear_weights(self, m):
        if isinstance(m, torch.nn.Linear):
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
            std = 1 / math.sqrt(m.in_features)
            torch.nn.init.uniform_(m.weight, -std, std)

    @torch.jit.ignore
    def no_weight_decay(self):
        no_wd_list = []
        named_parameters_list = [name for name, _ in self.named_parameters()]
        for module_name, module in self.named_modules():
            if (
                isinstance(module, torch.nn.Linear)
                or isinstance(module, SO3_LinearV2)
                or isinstance(module, torch.nn.LayerNorm)
                or isinstance(module, EquivariantLayerNormArray)
                or isinstance(module, EquivariantLayerNormArraySphericalHarmonics)
                or isinstance(module, EquivariantRMSNormArraySphericalHarmonics)
                or isinstance(module, EquivariantRMSNormArraySphericalHarmonicsV2)
            ):
                for parameter_name, _ in module.named_parameters():
                    if isinstance(module, torch.nn.Linear) or isinstance(module, SO3_LinearV2):
                        if "weight" in parameter_name:
                            continue
                    global_parameter_name = module_name + "." + parameter_name
                    assert global_parameter_name in named_parameters_list
                    no_wd_list.append(global_parameter_name)
        return set(no_wd_list)
