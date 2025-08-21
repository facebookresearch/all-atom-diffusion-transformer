"""Copyright (c) Meta Platforms, Inc. and affiliates."""

import os
import warnings
from functools import partial
from typing import Any, Dict, List

import numpy as np
import torch
import wandb
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure
from tqdm import tqdm

from src.tools.ase_notebook import AseView
from src.utils import joblib_map

warnings.simplefilter("ignore", FutureWarning)

ase_view = AseView(
    rotations="45x,45y,45z",
    atom_font_size=16,
    axes_length=30,
    canvas_size=(400, 400),
    zoom=1.2,
    show_bonds=False,
    # uc_dash_pattern=(.6, .4),
    atom_show_label=True,
    canvas_background_opacity=0.0,
)
# ase_view.add_miller_plane(1, 0, 0, color="green")


class MOFReconstructionEvaluator:
    """Evaluator for Metal Organic Framework reconstruction tasks. Can be used within a Lightning
    module, appending predictions and ground truths during training and computing metrics at the
    end of an epoch, or can be used as a standalone object to evaluate predictions on a dataset.

    Args:
        stol (float): StructureMatcher tolerance for matching sites.
        angle_tol (float): StructureMatcher tolerance for matching angles.
        ltol (float): StructureMatcher tolerance for matching lengths.
    """

    def __init__(self, stol=0.5, angle_tol=10, ltol=0.3, device="cpu"):
        self.matcher = StructureMatcher(stol=stol, angle_tol=angle_tol, ltol=ltol)
        self.pred_arrays_list = []  # list of Dict[str, np.array] predictions
        self.gt_arrays_list = []  # list of Dict[str, np.array] ground truths
        self.pred_mof_list = []  # list of MOF predictions
        self.gt_mof_list = []  # list of MOF ground truths
        self.device = device

    def append_pred_array(self, pred: Dict[str, np.array]):
        """Append a prediction to the evaluator."""
        self.pred_arrays_list.append(pred)

    def append_gt_array(self, gt: Dict[str, np.array]):
        """Append a ground truth to the evaluator."""
        self.gt_arrays_list.append(gt)

    def clear(self):
        """Clear the stored predictions and ground truths, to be used at the end of an epoch."""
        self.pred_arrays_list = []
        self.gt_arrays_list = []
        self.pred_mof_list = []
        self.gt_mof_list = []

    def _arrays_to_structures(self, save: bool = False, save_dir: str = ""):
        """Convert stored predictions and ground truths to PyMatGen Structure objects for
        evaluation."""
        self.pred_mof_list = joblib_map(
            partial(
                array_dict_to_structure,
                save=save,
                save_dir_name=f"{save_dir}/pred",
            ),
            self.pred_arrays_list,
            n_jobs=-4,
            inner_max_num_threads=1,
            desc=f"    Pred to Structure",
            total=len(self.pred_arrays_list),
        )
        self.gt_mof_list = joblib_map(
            partial(
                array_dict_to_structure,
                save=save,
                save_dir_name=f"{save_dir}/gt",
            ),
            self.gt_arrays_list,
            n_jobs=-4,
            inner_max_num_threads=1,
            desc=f"    G.T. to Structure",
            total=len(self.gt_arrays_list),
        )

    def _get_metrics(self, pred, gt, is_valid):
        if not is_valid:
            return float("inf")
        try:
            rms_dist = self.matcher.get_rms_dist(pred, gt)
            rms_dist = float("inf") if rms_dist is None else rms_dist[0]
            return rms_dist
        except Exception:
            return float("inf")

    def get_metrics(self, save: bool = False, save_dir: str = "") -> Dict[str, Any]:
        """Compute the match rate and avg. RMS distance between predictions and ground truths.

        Note: self.rms_dists can be used to access RMSD per sample but is not returned.

        Returns:
            Dict: Dictionary of metrics, including match rate and avg. RMSD.
        """
        assert len(self.pred_arrays_list) == len(
            self.gt_arrays_list
        ), "Number of predictions and ground truths must match."

        # Convert predictions and ground truths to MOF objects
        self._arrays_to_structures(save, save_dir)

        # Check validity of predictions and ground truths
        validity = [
            c1.properties["valid"] and c2.properties["valid"]
            for c1, c2 in zip(self.pred_mof_list, self.gt_mof_list)
        ]
        self.rms_dists = []
        for i in tqdm(range(len(self.pred_mof_list)), desc="   Reconstruction eval"):
            self.rms_dists.append(
                self._get_metrics(self.pred_mof_list[i], self.gt_mof_list[i], validity[i])
            )
        self.rms_dists = torch.tensor(self.rms_dists, device=self.device)
        match_rate = (~torch.isinf(self.rms_dists)).long()
        if len(self.rms_dists[~torch.isinf(self.rms_dists)]) == 0:
            # No valid predictions --> return large RMSD for logging purposes
            return {
                "match_rate": match_rate,
                "rms_dist": torch.tensor([10.0] * len(match_rate), device=self.device),
            }
        else:
            return {
                "match_rate": match_rate,
                "rms_dist": self.rms_dists[~torch.isinf(self.rms_dists)],
            }

    def get_wandb_table(self, current_epoch: int = 0, save_dir: str = "") -> wandb.Table:
        """Create a wandb.Table object with the results of the evaluation."""
        pred_table = wandb.Table(
            columns=[
                "Epoch",
                "Sample idx",
                "Num atoms",
                "RMSD",
                "Match?",
                "Valid?",
                "True atom types",
                "Pred atom types",
                "True lengths",
                "Pred lengths",
                "True angles",
                "Pred angles",
                "True 2D",
                "Pred 2D",
            ]
        )
        for idx in range(len(self.pred_mof_list)):
            sample_idx = self.gt_mof_list[idx].properties["sample_idx"]
            assert sample_idx == self.pred_mof_list[idx].properties["sample_idx"]

            num_atoms = len(self.pred_arrays_list[idx]["atom_types"])

            rmsd = self.rms_dists[idx]

            match = rmsd != float("inf")

            true_atom_types = " ".join(
                [str(int(t)) for t in self.gt_arrays_list[idx]["atom_types"]]
            )

            pred_atom_types = " ".join(
                [str(int(t)) for t in self.pred_arrays_list[idx]["atom_types"]]
            )

            true_lengths = " ".join(
                [f"{l:.2f}" for l in self.gt_arrays_list[idx]["lengths"].squeeze()]
            )

            true_angles = " ".join(
                [f"{a:.2f}" for a in self.gt_arrays_list[idx]["angles"].squeeze()]
            )

            pred_lengths = " ".join(
                [f"{l:.2f}" for l in self.pred_arrays_list[idx]["lengths"].squeeze()]
            )

            pred_angles = " ".join(
                [f"{a:.2f}" for a in self.pred_arrays_list[idx]["angles"].squeeze()]
            )

            # 2D structures
            try:
                true_2d = ase_view.make_wandb_image(
                    self.gt_mof_list[idx],
                    center_in_uc=False,
                )
            except Exception as e:
                # log.error(f"Failed to load 2D structure for true sample {sample_idx}.")
                true_2d = None
            try:
                pred_2d = ase_view.make_wandb_image(
                    self.pred_mof_list[idx],
                    center_in_uc=False,
                )
            except Exception as e:
                # log.error(f"Failed to load 2D structure for pred sample {sample_idx}.")
                pred_2d = None

            # Update table
            pred_table.add_data(
                current_epoch,
                sample_idx,
                num_atoms,
                rmsd,
                match,
                self.pred_mof_list[idx].properties["valid"],
                true_atom_types,
                pred_atom_types,
                true_lengths,
                pred_lengths,
                true_angles,
                pred_angles,
                true_2d,
                pred_2d,
            )

        return pred_table


def array_dict_to_structure(
    x: dict[str, np.ndarray],
    save: bool = False,
    save_dir_name: str = "",
) -> Structure:
    """Method to convert a dictionary of numpy arrays to a Structure object which is compatible
    with StructureMatcher (used for evaluations).

    Args:
        x: Dictionary of numpy arrays with keys:
            - 'frac_coords': Fractional coordinates of atoms.
            - 'atom_types': Atomic numbers of atoms.
            - 'lengths': Lengths of the lattice vectors.
            - 'angles': Angles between the lattice vectors.
            - 'sample_idx': Index of the sample in the dataset.
        save: Whether to save the MOF as a CIF file.
        save_dir_name: Directory to save the CIF file.

    Returns:
        Structure: PyMatGen Structure object, optionally saved as a CIF file.
    """
    try:
        frac_coords = x["frac_coords"]
        atom_types = x["atom_types"]
        lengths = x["lengths"].squeeze().tolist()
        angles = x["angles"].squeeze().tolist()
        sample_idx = x["sample_idx"]

        struct = Structure(
            lattice=Lattice.from_parameters(*(lengths + angles)),
            species=atom_types,
            coords=frac_coords,
            coords_are_cartesian=False,
        )
        struct.properties["sample_idx"] = sample_idx
        struct.properties["valid"] = True
        if save:
            os.makedirs(save_dir_name, exist_ok=True)
            struct.to(os.path.join(save_dir_name, f"mof_{x['sample_idx']}.cif"))

    except:
        # returns an absurd MOF
        frac_coords = np.zeros_like(x["frac_coords"])
        atom_types = np.ones_like(x["atom_types"])
        lengths = (100 * np.ones_like(x["lengths"])).squeeze().tolist()
        angles = (90 * np.ones_like(x["angles"])).squeeze().tolist()
        sample_idx = x["sample_idx"]

        struct = Structure(
            lattice=Lattice.from_parameters(*(lengths + angles)),
            species=atom_types,
            coords=frac_coords,
            coords_are_cartesian=False,
        )
        struct.properties["sample_idx"] = sample_idx
        struct.properties["valid"] = False

    return struct
