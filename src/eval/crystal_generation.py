"""Copyright (c) Meta Platforms, Inc. and affiliates."""

import os
import warnings
from functools import partial
from typing import Any, Dict, Literal, Tuple

import numpy as np
import torch
import wandb
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core.structure import Structure
from tqdm import tqdm

from src.eval.crystal import Crystal
from src.tools.ase_notebook import AseView
from src.utils import joblib_map, pylogger

warnings.filterwarnings("ignore", category=UserWarning)

log = pylogger.RankedLogger(__name__)

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


class CrystalGenerationEvaluator:
    """Evaluator for crystal generation tasks.

    Can be used within a Lightning module by appending sampled structures and computing metrics at
    the end of an epoch.
    """

    def __init__(
        self,
        dataset_cif_list,
        stol=0.5,
        angle_tol=10,
        ltol=0.3,
        device="cpu",
        compute_novelty=False,
    ):
        self.dataset_cif_list = dataset_cif_list
        self.dataset_struct_list = None  # loader first time it is required
        self.matcher = StructureMatcher(stol=stol, angle_tol=angle_tol, ltol=ltol)
        self.pred_arrays_list = []
        self.pred_crys_list = []
        self.device = device
        self.compute_novelty = compute_novelty

    def append_pred_array(self, pred: Dict):
        """Append a prediction to the evaluator."""
        self.pred_arrays_list.append(pred)

    def clear(self):
        """Clear the stored predictions, to be used at the end of an epoch."""
        self.pred_arrays_list = []
        self.pred_crys_list = []

    def _arrays_to_crystals(self, save: bool = False, save_dir: str = ""):
        """Convert stored predictions and ground truths to Crystal objects for evaluation."""
        self.pred_crys_list = joblib_map(
            partial(
                array_dict_to_crystal,
                save=save,
                save_dir_name=save_dir,
            ),
            self.pred_arrays_list,
            n_jobs=-4,
            inner_max_num_threads=1,
            desc=f"    Pred to Crystal",
            total=len(self.pred_arrays_list),
        )

    def _dataset_cif_to_struct(self):
        """Convert dataset CIFs to Structure objects for novelty evaluation."""
        if self.dataset_struct_list is None:
            self.dataset_struct_list = joblib_map(
                partial(Structure.from_str, fmt="cif"),
                self.dataset_cif_list,
                n_jobs=-4,
                inner_max_num_threads=1,
                desc="    Load dataset CIFs (one time)",
                total=len(self.dataset_cif_list),
            )

    def _get_novelty(self, struct):
        # matcher = StructureMatcher(stol=0.5, angle_tol=10, ltol=0.3)
        for other_struct in self.dataset_struct_list:
            if self.matcher.fit(struct, other_struct, skip_structure_reduction=True):
                return False
        return True

    def get_metrics(self, save: bool = False, save_dir: str = ""):
        assert len(self.pred_arrays_list) > 0, "No predictions to evaluate."

        # Convert predictions and ground truths to Crystal objects
        self._arrays_to_crystals(save, save_dir)

        # Compute validity metrics
        metrics_dict = {
            "valid_rate": torch.tensor([c.valid for c in self.pred_crys_list], device=self.device),
            "comp_valid_rate": torch.tensor(
                [c.comp_valid for c in self.pred_crys_list], device=self.device
            ),
            "struct_valid_rate": torch.tensor(
                [c.struct_valid for c in self.pred_crys_list], device=self.device
            ),
        }

        # Compute uniqueness
        valid_structs = [c.structure for c in self.pred_crys_list if c.valid]
        unique_struct_groups = self.matcher.group_structures(valid_structs)
        if len(valid_structs) > 0:
            metrics_dict["unique_rate"] = torch.tensor(
                len(unique_struct_groups) / len(valid_structs), device=self.device
            )
        else:
            metrics_dict["unique_rate"] = torch.tensor(0.0, device=self.device)

        # Compute novelty (slow to compute)
        if self.compute_novelty:
            self._dataset_cif_to_struct()
            struct_is_novel = []
            for struct in tqdm(
                [group[0] for group in unique_struct_groups],
                desc="    Novelty",
                total=len(unique_struct_groups),
            ):
                struct_is_novel.append(self._get_novelty(struct))
            # # This is slower...
            # struct_is_novel = joblib_map(
            #     self._get_novelty,
            #     [group[0] for group in unique_struct_groups],
            #     n_jobs=-4,
            #     inner_max_num_threads=1,
            #     desc=f"    Novelty",
            #     total=len(unique_struct_groups),
            # )
            metrics_dict["novel_rate"] = torch.tensor(
                sum(struct_is_novel) / len(struct_is_novel), device=self.device
            )
        else:
            metrics_dict["novel_rate"] = torch.tensor(-1.0, device=self.device)

        return metrics_dict

    def get_wandb_table(self, current_epoch: int = 0, save_dir: str = ""):
        # Log crystal structures and metrics to wandb
        pred_table = wandb.Table(
            columns=[
                "Global step",
                "Sample idx",
                "Num atoms",
                "Valid?",
                "Comp valid?",
                "Struct valid?",
                "Pred atom types",
                "Pred lengths",
                "Pred angles",
                "Pred 2D",
            ]
        )

        for idx in range(len(self.pred_crys_list)):
            sample_idx = self.pred_crys_list[idx].sample_idx

            num_atoms = len(self.pred_crys_list[idx].atom_types)

            pred_atom_types = " ".join([str(int(t)) for t in self.pred_crys_list[idx].atom_types])

            pred_lengths = " ".join([f"{l:.2f}" for l in self.pred_crys_list[idx].lengths])

            pred_angles = " ".join([f"{a:.2f}" for a in self.pred_crys_list[idx].angles])

            try:
                pred_2d = ase_view.make_wandb_image(
                    self.pred_crys_list[idx].structure,
                    center_in_uc=False,
                )
            except Exception as e:
                log.error(f"Failed to load 2D structure for pred sample {sample_idx}.")
                pred_2d = None

            # Update table
            pred_table.add_data(
                current_epoch,
                sample_idx,
                num_atoms,
                self.pred_crys_list[idx].valid,
                self.pred_crys_list[idx].comp_valid,
                self.pred_crys_list[idx].struct_valid,
                pred_atom_types,
                pred_lengths,
                pred_angles,
                pred_2d,
            )

        return pred_table


def array_dict_to_crystal(
    x: dict[str, np.ndarray],
    save: bool = False,
    save_dir_name: str = "",
) -> Crystal:
    """Method to convert a dictionary of numpy arrays to a Crystal object which is compatible with
    StructureMatcher (used for evaluations). Previously called 'safe_crystal', as it return a
    generic crystal if the input is invalid.

    Adapted from: https://github.com/facebookresearch/flowmm

    Args:
        x: Dictionary of numpy arrays with keys:
            - 'frac_coords': Fractional coordinates of atoms.
            - 'atom_types': Atomic numbers of atoms.
            - 'lengths': Lengths of the lattice vectors.
            - 'angles': Angles between the lattice vectors.
            - 'sample_idx': Index of the sample in the dataset.
        save: Whether to save the crystal as a CIF file.
        save_dir_name: Directory to save the CIF file.

    Returns:
        Crystal: Crystal object, optionally saved as a CIF file.
    """
    # Check if the lattice angles are in a valid range
    if np.all(50 < x["angles"]) and np.all(x["angles"] < 130):
        crys = Crystal(x)
        if save:
            os.makedirs(save_dir_name, exist_ok=True)
            crys.structure.to(os.path.join(save_dir_name, f"crystal_{x['sample_idx']}.cif"))
    else:
        # returns an absurd crystal
        crys = Crystal(
            {
                "frac_coords": np.zeros_like(x["frac_coords"]),
                "atom_types": np.zeros_like(x["atom_types"]),
                "lengths": 100 * np.ones_like(x["lengths"]),
                "angles": np.ones_like(x["angles"]) * 90,
                "sample_idx": x["sample_idx"],
            }
        )
    return crys
