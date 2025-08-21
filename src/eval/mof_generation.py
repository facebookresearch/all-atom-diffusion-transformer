"""Copyright (c) Meta Platforms, Inc. and affiliates."""

import os
import warnings
from functools import partial
from typing import Any, Dict, Literal, Tuple

import numpy as np
import pandas as pd
import torch
import wandb
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SymmetryUndeterminedError
from tqdm import tqdm

from src.tools.ase_notebook import AseView
from src.utils import joblib_map, pylogger

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

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


class MOFGenerationEvaluator:
    """Evaluator for Metal Organic Framework generation tasks.

    Can be used within a Lightning module by appending sampled structures and computing metrics at
    the end of an epoch.
    """

    def __init__(
        self,
        stol=0.5,
        angle_tol=10,
        ltol=0.3,
        device="cpu",
    ):
        self.matcher = StructureMatcher(stol=stol, angle_tol=angle_tol, ltol=ltol)
        self.pred_arrays_list = []
        self.pred_mof_list = []
        self.device = device

    def append_pred_array(self, pred: Dict):
        """Append a prediction to the evaluator."""
        self.pred_arrays_list.append(pred)

    def clear(self):
        """Clear the stored predictions, to be used at the end of an epoch."""
        self.pred_arrays_list = []
        self.pred_mof_list = []

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

    def get_metrics(self, save: bool = False, save_dir: str = ""):
        assert len(self.pred_arrays_list) > 0, "No predictions to evaluate."

        # Convert predictions and ground truths to Crystal objects
        self._arrays_to_structures(save, save_dir)

        # Compute validity metrics
        metrics_dict = {
            "valid_rate": torch.tensor(
                [c.properties["valid"] for c in self.pred_mof_list], device=self.device
            ),
        }

        # NOTE MOFChecker seems to cause segfaults when used in our slurm environment
        # in an iterative manner, but works all fine when used in a notebook iteratively.

        # from mofchecker import MOFChecker

        # mofchecker_dict = []
        # for s in tqdm(valid_structs, desc="    MOFChecker"):
        #     try:
        #         mofchecker = MOFChecker(
        #             structure=s,
        #             symprec=None,
        #             angle_tolerance=None,
        #             primitive=False
        #         )
        #         desc = mofchecker.get_mof_descriptors()
        #         all_checks = []
        #         for k, v in desc.items():
        #             if type(v) == bool:
        #                 if k == "has_3d_connected_graph":
        #                     continue
        #                 if k in ["has_carbon", "has_hydrogen", "has_metal", "is_porous"]:
        #                     all_checks.append(int(v))
        #                 else:
        #                     all_checks.append(int(not v))
        #         desc["all_checks"] = np.all(all_checks)
        #     except: # SymmetryUndeterminedError or IndexError:
        #         # import ipdb; ipdb.set_trace()
        #         # all checks failed if PyMatGen leads to an error
        #         desc = {
        #             "has_carbon": False,
        #             "has_hydrogen": False,
        #             "has_atomic_overlaps": True,
        #             "has_overcoordinated_c": True,
        #             "has_overcoordinated_n": True,
        #             "has_overcoordinated_h": True,
        #             "has_undercoordinated_c": True,
        #             "has_undercoordinated_n": True,
        #             "has_undercoordinated_rare_earth": True,
        #             "has_metal": False,
        #             "has_lone_molecule": True,
        #             "has_high_charges": True,
        #             # "is_porous",  # always nan for all QMOF CIF files
        #             "has_suspicicious_terminal_oxo": True,
        #             "has_undercoordinated_alkali_alkaline": True,
        #             "has_geometrically_exposed_metal": True,
        #             # 'has_3d_connected_graph',
        #             "all_checks": False,
        #         }
        #     mofchecker_dict.append(dict(desc))

        # mofchecker_dict = (
        #     pd.DataFrame(
        #         mofchecker_dict,
        #         columns=[
        #             "has_carbon",
        #             "has_hydrogen",
        #             "has_atomic_overlaps",
        #             "has_overcoordinated_c",
        #             "has_overcoordinated_n",
        #             "has_overcoordinated_h",
        #             "has_undercoordinated_c",
        #             "has_undercoordinated_n",
        #             "has_undercoordinated_rare_earth",
        #             "has_metal",
        #             "has_lone_molecule",
        #             "has_high_charges",
        #             # "is_porous",  # always nan for all QMOF CIF files
        #             "has_suspicicious_terminal_oxo",
        #             "has_undercoordinated_alkali_alkaline",
        #             "has_geometrically_exposed_metal",
        #             # 'has_3d_connected_graph',
        #             "all_checks",
        #         ],
        #     )
        #     .mean()
        #     .to_dict()
        # )

        # metrics_dict = {**metrics_dict, **mofchecker_dict}

        return metrics_dict

    def get_wandb_table(self, current_epoch: int = 0, save_dir: str = ""):
        # Log crystal structures and metrics to wandb
        pred_table = wandb.Table(
            columns=[
                "Global step",
                "Sample idx",
                "Num atoms",
                "Valid?",
                "Pred atom types",
                "Pred lengths",
                "Pred angles",
                "Pred 2D",
            ]
        )

        for idx in range(len(self.pred_mof_list)):
            sample_idx = self.pred_mof_list[idx].properties["sample_idx"]

            num_atoms = len(self.pred_arrays_list[idx]["atom_types"])

            pred_atom_types = " ".join(
                [str(int(t)) for t in self.pred_arrays_list[idx]["atom_types"]]
            )

            pred_lengths = " ".join(
                [f"{l:.2f}" for l in self.pred_arrays_list[idx]["lengths"].squeeze()]
            )

            pred_angles = " ".join(
                [f"{a:.2f}" for a in self.pred_arrays_list[idx]["angles"].squeeze()]
            )

            try:
                pred_2d = ase_view.make_wandb_image(
                    self.pred_mof_list[idx],
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
                self.pred_mof_list[idx].properties["valid"],
                pred_atom_types,
                pred_lengths,
                pred_angles,
                pred_2d,
            )

        return pred_table


def array_dict_to_structure(
    x: dict[str, np.ndarray], save: bool = False, save_dir_name: str = "", cutoff=0.5
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

        # structural validity
        dist_mat = struct.distance_matrix
        dist_mat = dist_mat + np.diag(np.ones(dist_mat.shape[0]) * (cutoff + 10.0))
        if dist_mat.min() < cutoff or struct.volume < 0.1:
            struct.properties["valid"] = False
        else:
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
