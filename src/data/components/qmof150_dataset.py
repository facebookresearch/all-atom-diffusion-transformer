"""Copyright (c) Meta Platforms, Inc. and affiliates."""

import os
import warnings
from typing import Callable, List, Optional

import numpy as np
import torch
from mofchecker import MOFChecker
from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm

from src.data.components.preprocessing_utils import build_crystal, build_crystal_graph

warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", FutureWarning)


class QMOF150(InMemoryDataset):
    """The QMOF150 dataset from QMOF, as a PyG InMemoryDataset.

    In order to create a torch_geometric.data.InMemoryDataset, you need to implement four fundamental methods:
    - InMemoryDataset.raw_file_names(): A list of files in the raw_dir which needs to be found in order to skip the download.
    - InMemoryDataset.processed_file_names(): A list of files in the processed_dir which needs to be found in order to skip the processing.
    - InMemoryDataset.download(): Downloads raw data into raw_dir.
    - InMemoryDataset.process(): Processes raw data and saves it into the processed_dir.

    Args:
        root (str): Root directory where the dataset should be saved.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
        force_reload (bool, optional): Whether to re-process the dataset.
            (default: :obj:`False`)
    """

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        force_reload: bool = False,
    ) -> None:
        super().__init__(root, transform, pre_transform, pre_filter, force_reload=force_reload)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        return ["relaxed_structures"]

    @property
    def processed_file_names(self) -> List[str]:
        return ["qmof150.pt"]

    def download(self) -> None:
        import zipfile

        from huggingface_hub import hf_hub_download

        hf_hub_download(
            repo_id="chaitjo/QMOF150_ADiT",
            filename="raw/relaxed_structures.zip",
            repo_type="dataset",
            local_dir=self.root,
        )
        with zipfile.ZipFile(
            os.path.join(self.root, "raw/relaxed_structures.zip"), "r"
        ) as zip_ref:
            zip_ref.extractall(os.path.join(self.root, "raw/"))

    def process(self) -> None:
        if os.path.exists(os.path.join(self.root, "raw/all.pt")):
            cached_data = torch.load(os.path.join(self.root, "raw/all.pt"))
        else:
            data_dir = os.path.join(self.root, "raw/relaxed_structures")
            filenames = os.listdir(data_dir)
            cached_data = []
            t = tqdm(filenames)
            for filename in t:
                t.set_description(filename)
                filepath = os.path.join(data_dir, filename)
                crystal_str = open(filepath).read()
                crystal = build_crystal(crystal_str, niggli=True, primitive=False)
                result_dict = {}
                result_dict["spacegroup"] = 1  # currently do not compute the spacegroup...
                graph_arrays = build_crystal_graph(crystal, "none")
                result_dict.update(
                    {"qmof_id": filename, "cif": crystal_str, "graph_arrays": graph_arrays}
                )
                cached_data.append(result_dict)

            torch.save(cached_data, os.path.join(self.root, "raw/all.pt"))

        data_list = []
        for data_dict in cached_data:
            # extract attributes from data_dict
            graph_arrays = data_dict["graph_arrays"]
            atom_types = graph_arrays["atom_types"]
            frac_coords = graph_arrays["frac_coords"]
            cell = graph_arrays["cell"]
            lattices = graph_arrays["lattices"]
            lengths = graph_arrays["lengths"]
            angles = graph_arrays["angles"]
            num_atoms = graph_arrays["num_atoms"]

            # only retain MOFs with under 150 atoms in unit cell
            if num_atoms <= 150:
                # normalize the lengths of lattice vectors, which makes
                # lengths for materials of different sizes at same scale
                _lengths = lengths / float(num_atoms) ** (1 / 3)
                # convert angles of lattice vectors to be in radians
                _angles = np.radians(angles)
                # add scaled lengths and angles to graph arrays
                graph_arrays["length_scaled"] = _lengths
                graph_arrays["angles_radians"] = _angles
                graph_arrays["lattices_scaled"] = np.concatenate([_lengths, _angles])

                data = Data(
                    id=data_dict["qmof_id"],
                    atom_types=torch.LongTensor(atom_types),
                    frac_coords=torch.Tensor(frac_coords),
                    cell=torch.Tensor(cell).unsqueeze(0),
                    lattices=torch.Tensor(lattices).unsqueeze(0),
                    lattices_scaled=torch.Tensor(graph_arrays["lattices_scaled"]).unsqueeze(0),
                    lengths=torch.Tensor(lengths).view(1, -1),
                    lengths_scaled=torch.Tensor(graph_arrays["length_scaled"]).view(1, -1),
                    angles=torch.Tensor(angles).view(1, -1),
                    angles_radians=torch.Tensor(graph_arrays["angles_radians"]).view(1, -1),
                    num_atoms=torch.LongTensor([num_atoms]),
                    num_nodes=torch.LongTensor(
                        [num_atoms]
                    ),  # special attribute used for PyG batching
                    token_idx=torch.arange(num_atoms),
                    dataset_idx=torch.tensor(
                        [0], dtype=torch.long
                    ),  # 0 --> indicates periodic/crystal
                )
                # 3D coordinates (NOTE do not zero-center prior to graph construction)
                data.pos = torch.einsum(
                    "bi,bij->bj",
                    data.frac_coords,
                    torch.repeat_interleave(data.cell, data.num_atoms, dim=0),
                )
                # space group number
                data.spacegroup = torch.LongTensor([data_dict["spacegroup"]])

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue
                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                data_list.append(data)

        self.save(data_list, os.path.join(self.root, "processed/qmof150.pt"))


def process_one(data_dir, filename):
    filepath = os.path.join(data_dir, filename)
    crystal_str = open(filepath).read()
    crystal = build_crystal(crystal_str, niggli=True, primitive=False)
    result_dict = {}
    result_dict["spacegroup"] = 1  # currently do not compute the spacegroup...
    graph_arrays = build_crystal_graph(crystal, "none")
    result_dict.update({"qmof_id": filename, "cif": crystal_str, "graph_arrays": graph_arrays})

    checker = MOFChecker(structure=crystal)  # .from_cif()
    desc = checker.get_mof_descriptors()
    all_check = []
    for k, v in desc.items():
        if type(v) == bool:
            if k == "has_3d_connected_graph":
                # NOTE
                # <has_3d_connected_graph> is introduced in a newer version of MOFCHecker.
                # MOFCheckerv0.9.5 does not have this check. When using a newer version of MOFChecker,
                # this check is skipped to be consistent with paper-reported results.
                continue
            if k in [
                "has_carbon",
                "has_hydrogen",
                "has_metal",
                "is_porous",
            ]:
                all_check.append(int(v))
            else:
                all_check.append(int(not v))
    # only save MOFs which pass all checks
    if np.all(all_check):
        return result_dict
    else:
        return None
