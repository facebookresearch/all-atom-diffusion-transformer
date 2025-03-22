"""Copyright (c) Meta Platforms, Inc. and affiliates."""

import os
import warnings

import hydra
import numpy as np
import omegaconf
import pandas as pd
import torch
from omegaconf import ValueNode
from torch.utils.data import Dataset
from torch_geometric.data import Data

from src.data.components.preprocessing_utils import preprocess

warnings.filterwarnings("ignore", category=DeprecationWarning)


class CrystalDataset(Dataset):
    """Crystal dataset class for periodic 3D crystal structures.

    Adapted from CDVAE: https://github.com/txie-93/cdvae

    Args:
        name: Name of the dataset.
        path: Path to the dataset CSV file.
        save_path: Path to save the preprocessed data.
        prop: Name of property attribute to predict.
        niggli: Whether to use Niggli cell reduction.
        primitive: Whether to use primitive cell reduction.
        graph_method: Method to construct graph.
        tolerance: Tolerance for symmetry finding.
        use_space_group: Whether to include space group label.
        use_pos_index: Whether to include position index.
        preprocess_workers: Number of workers for data preprocessing.
    """

    def __init__(
        self,
        name: ValueNode,
        path: ValueNode,
        save_path: ValueNode,
        prop: ValueNode,
        niggli: ValueNode,
        primitive: ValueNode,
        graph_method: ValueNode,
        tolerance: ValueNode,
        use_space_group: ValueNode,
        use_pos_index: ValueNode,
        preprocess_workers: ValueNode,
    ) -> None:
        super().__init__()

        self.path = path
        self.name = name
        self.df = pd.read_csv(path)
        self.prop = prop
        self.niggli = niggli
        self.primitive = primitive
        self.graph_method = graph_method
        self.tolerance = tolerance
        self.use_space_group = use_space_group
        self.use_pos_index = use_pos_index

        self._preprocess(save_path, preprocess_workers, prop)
        self._add_scaled_lattices_attr()

    def _preprocess(self, save_path, preprocess_workers, prop):
        if os.path.exists(save_path):
            # Load cached data into memory
            self.cached_data = torch.load(save_path)
        else:
            import warnings

            warnings.simplefilter("ignore", UserWarning)
            print(f"Preprocessing {self.name} dataset...")
            cached_data = preprocess(
                self.path,
                preprocess_workers,
                niggli=self.niggli,
                primitive=self.primitive,
                graph_method=self.graph_method,
                prop_list=[prop],
                use_space_group=self.use_space_group,
                tol=self.tolerance,
            )
            # Save preprocessed data as pt object
            torch.save(cached_data, save_path)
            self.cached_data = cached_data

    def _add_scaled_lattices_attr(self):
        for data_dict in self.cached_data:
            graph_arrays = data_dict["graph_arrays"]
            lengths = graph_arrays["lengths"]
            angles = graph_arrays["angles"]
            num_atoms = graph_arrays["num_atoms"]

            # normalize the lengths of lattice vectors, which makes
            # lengths for materials of different sizes at same scale
            _lengths = lengths / float(num_atoms) ** (1 / 3)

            # convert angles of lattice vectors to be in radians
            _angles = np.radians(angles)

            # add scaled lengths and angles to graph arrays
            graph_arrays["length_scaled"] = _lengths
            graph_arrays["angles_radians"] = _angles
            graph_arrays["lattices_scaled"] = np.concatenate([_lengths, _angles])

    def __len__(self) -> int:
        return len(self.cached_data)

    def __getitem__(self, index):
        data_dict = self.cached_data[index]

        atom_types = torch.LongTensor(data_dict["graph_arrays"]["atom_types"])
        frac_coords = torch.Tensor(data_dict["graph_arrays"]["frac_coords"])
        cell = torch.Tensor(data_dict["graph_arrays"]["cell"]).unsqueeze(0)
        lattices = torch.Tensor(data_dict["graph_arrays"]["lattices"]).unsqueeze(0)
        lattices_scaled = torch.Tensor(data_dict["graph_arrays"]["lattices_scaled"]).unsqueeze(0)
        lengths = torch.Tensor(data_dict["graph_arrays"]["lengths"]).view(1, -1)
        lengths_scaled = torch.Tensor(data_dict["graph_arrays"]["length_scaled"]).view(1, -1)
        angles = torch.Tensor(data_dict["graph_arrays"]["angles"]).view(1, -1)
        angles_radians = torch.Tensor(data_dict["graph_arrays"]["angles_radians"]).view(1, -1)
        num_atoms = torch.LongTensor([data_dict["graph_arrays"]["num_atoms"]])
        token_idx = torch.arange(data_dict["graph_arrays"]["num_atoms"])
        # edge_index = torch.LongTensor(data_dict['graph_arrays']['edge_indices'].T).contiguous()
        # to_jimages = torch.LongTensor(data_dict['graph_arrays']['to_jimages'])

        # Cartesian coordinates (NOTE do not zero-center prior to graph construction)
        pos = torch.einsum(
            "bi,bij->bj", frac_coords, torch.repeat_interleave(cell, num_atoms, dim=0)
        )

        data = Data(
            atom_types=atom_types,
            pos=pos,
            frac_coords=frac_coords,
            cell=cell,
            lattices=lattices,
            lattices_scaled=lattices_scaled,
            lengths=lengths,
            lengths_scaled=lengths_scaled,
            angles=angles,
            angles_radians=angles_radians,
            num_atoms=num_atoms,
            num_nodes=num_atoms,  # special attribute used for PyG batching
            token_idx=token_idx,
            dataset_idx=torch.tensor([0], dtype=torch.long),
        )

        if self.use_space_group:
            data.spacegroup = torch.LongTensor([data_dict["spacegroup"]])
            # data.ops = torch.Tensor(data_dict["wyckoff_ops"])
            # data.anchor_index = torch.LongTensor(data_dict["anchors"])

        # if self.use_pos_index:
        #     pos_dic = {}
        #     indexes = []
        #     for atom in atom_types:
        #         pos_dic[atom] = pos_dic.get(atom, 0) + 1
        #         indexes.append(pos_dic[atom] - 1)
        #     data.index = torch.LongTensor(indexes)

        return data

    def __repr__(self) -> str:
        return f"CrystalDataset({self.name=}, {self.path=})"
