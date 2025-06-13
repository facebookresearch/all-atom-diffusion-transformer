"""Copyright (c) Meta Platforms, Inc. and affiliates."""

import copy
import os
import random
import time
from typing import Any, Dict, Literal, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import wandb
from lightning import LightningModule
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig
from torch.nn import ModuleDict
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_batch
from torchmetrics import MeanMetric
from tqdm import tqdm

from src.eval.crystal_generation import CrystalGenerationEvaluator
from src.eval.mof_generation import MOFGenerationEvaluator
from src.eval.molecule_generation import MoleculeGenerationEvaluator
from src.models.components.kabsch_utils import random_rotation_matrix
from src.models.vae_module import VariationalAutoencoderLitModule
from src.utils import pylogger

log = pylogger.RankedLogger(__name__)


# IDX_TO_DATASET = {
#     0: "mp20",
#     1: "qm9",
#     2: "qmof150",
# }
# DATASET_TO_IDX = {
#     "mp20": 0,  # periodic
#     "qm9": 1,  # non-periodic
#     "qmof150": 0,  # periodic
# }


class LatentDiffusionLitModule(LightningModule):
    """LightningModule for latent diffusion generative modellling of 3D atomic systems.

    A `LightningModule` implements 8 key methods:

    ```python
    def __init__(self):
    # Define initialization code here.

    def setup(self, stage):
    # Things to setup before each stage, 'fit', 'validate', 'test', 'predict'.
    # This hook is called on every process when using DDP.

    def training_step(self, batch, batch_idx):
    # The complete training step.

    def validation_step(self, batch, batch_idx):
    # The complete validation step.

    def test_step(self, batch, batch_idx):
    # The complete test step.

    def predict_step(self, batch, batch_idx):
    # The complete predict step.

    def configure_optimizers(self):
    # Define and configure optimizers and LR schedulers.
    ```

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        autoencoder_ckpt: str,
        denoiser: torch.nn.Module,
        interpolant: DictConfig,
        augmentations: DictConfig,
        sampling: DictConfig,
        conditioning: DictConfig,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        scheduler_frequency: str,
        compile: bool,
        lambda_energy: float = 0.2,  # weight for energy loss
        energy_head_hidden_dim: int = 128,  # hidden dimension for energy head
        disable_evaluators=False,  # add this
    ) -> None:
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # autoencoder models (first-stage model)
        self.autoencoder_ckpt = autoencoder_ckpt
        log.info(f"Loading Autoencoder ckpt: {autoencoder_ckpt}")
        self.autoencoder = VariationalAutoencoderLitModule.load_from_checkpoint(
            autoencoder_ckpt, map_location="cpu"
        )
        # freeze autoencoder
        self.autoencoder.requires_grad_(False)
        self.autoencoder.eval()

        # denoiser model (second-stage model)
        self.denoiser = denoiser

        # interpolant for diffusion or flow matching training/sampling
        self.interpolant = interpolant

        # evaluator objects for computing metrics
        if not getattr(self.hparams, "disable_evaluators", False):
            self.val_generation_evaluators = {
                # "mp20": CrystalGenerationEvaluator(
                #     dataset_cif_list=pd.read_csv(
                #         os.path.join(self.hparams.sampling.data_dir, f"mp_20/raw/all.csv")
                #     )["cif"].tolist()
                # ),
                "qm9": MoleculeGenerationEvaluator(
                    dataset_smiles_list=torch.load(
                        os.path.join(self.hparams.sampling.data_dir, f"qm9/smiles.pt"),
                    ),
                    removeHs=self.hparams.sampling.removeHs,
                ),
                # "qmof150": MOFGenerationEvaluator(),
            }
            self.test_generation_evaluators = copy.deepcopy(self.val_generation_evaluators)
        else:
            log.warning("Evaluator loading disabled for testing.")
            self.val_generation_evaluators = {}
            self.test_generation_evaluators = {}

        # metric objects for calculating and averaging across batches
        self.train_metrics = ModuleDict(
            {
                "loss": MeanMetric(),
                "x_loss": MeanMetric(),
                "energy_loss": MeanMetric(),  # energy-aware loss
                "x_loss t=[0,25)": MeanMetric(),
                "x_loss t=[25,50)": MeanMetric(),
                "x_loss t=[50,75)": MeanMetric(),
                "x_loss t=[75,100)": MeanMetric(),
                "t_avg": MeanMetric(),
                "dataset_idx": MeanMetric(),
            }
        )
        self.val_metrics = ModuleDict(
            {
                # "mp20": ModuleDict(
                #     {
                #         "loss": MeanMetric(),
                #         "x_loss": MeanMetric(),
                #         "energy_loss": MeanMetric(),  # energy-aware loss
                #         "x_loss t=[0,25)": MeanMetric(),
                #         "x_loss t=[25,50)": MeanMetric(),
                #         "x_loss t=[50,75)": MeanMetric(),
                #         "x_loss t=[75,100)": MeanMetric(),
                #         "t_avg": MeanMetric(),
                #         "valid_rate": MeanMetric(),
                #         "struct_valid_rate": MeanMetric(),
                #         "comp_valid_rate": MeanMetric(),
                #         "unique_rate": MeanMetric(),
                #         "novel_rate": MeanMetric(),
                #         "sampling_time": MeanMetric(),
                #     }
                # ),
                "qm9": ModuleDict(
                    {
                        "loss": MeanMetric(),
                        "x_loss": MeanMetric(),
                        "energy_loss": MeanMetric(),  # energy-aware loss
                        "x_loss t=[0,25)": MeanMetric(),
                        "x_loss t=[25,50)": MeanMetric(),
                        "x_loss t=[50,75)": MeanMetric(),
                        "x_loss t=[75,100)": MeanMetric(),
                        "t_avg": MeanMetric(),
                        "valid_rate": MeanMetric(),
                        "unique_rate": MeanMetric(),
                        "novel_rate": MeanMetric(),
                        "mol_pred_loaded": MeanMetric(),
                        "sanitization": MeanMetric(),
                        "inchi_convertible": MeanMetric(),
                        "all_atoms_connected": MeanMetric(),
                        "bond_lengths": MeanMetric(),
                        "bond_angles": MeanMetric(),
                        "internal_steric_clash": MeanMetric(),
                        "aromatic_ring_flatness": MeanMetric(),
                        "double_bond_flatness": MeanMetric(),
                        "internal_energy": MeanMetric(),
                        "sampling_time": MeanMetric(),
                    }
                ),
                # "qmof150": ModuleDict(
                #     {
                #         "loss": MeanMetric(),
                #         "x_loss": MeanMetric(),
                #         "energy_loss": MeanMetric(),  # energy-aware loss
                #         "x_loss t=[0,25)": MeanMetric(),
                #         "x_loss t=[25,50)": MeanMetric(),
                #         "x_loss t=[50,75)": MeanMetric(),
                #         "x_loss t=[75,100)": MeanMetric(),
                #         "t_avg": MeanMetric(),
                #         "valid_rate": MeanMetric(),
                #         "unique_rate": MeanMetric(),
                #         "has_carbon": MeanMetric(),
                #         "has_hydrogen": MeanMetric(),
                #         "has_atomic_overlaps": MeanMetric(),
                #         "has_overcoordinated_c": MeanMetric(),
                #         "has_overcoordinated_n": MeanMetric(),
                #         "has_overcoordinated_h": MeanMetric(),
                #         "has_undercoordinated_c": MeanMetric(),
                #         "has_undercoordinated_n": MeanMetric(),
                #         "has_undercoordinated_rare_earth": MeanMetric(),
                #         "has_metal": MeanMetric(),
                #         "has_lone_molecule": MeanMetric(),
                #         "has_high_charges": MeanMetric(),
                #         # "is_porous": MeanMetric(),
                #         "has_suspicicious_terminal_oxo": MeanMetric(),
                #         "has_undercoordinated_alkali_alkaline": MeanMetric(),
                #         "has_geometrically_exposed_metal": MeanMetric(),
                #         # 'has_3d_connected_graph': MeanMetric(),
                #         "all_checks": MeanMetric(),
                #         "sampling_time": MeanMetric(),
                #     }
                # ),
            }
        )
        self.test_metrics = copy.deepcopy(self.val_metrics)

        # load bincounts for sampling
        self.num_nodes_bincount = {
            # "mp20": torch.nn.Parameter(
            #     torch.load(
            #         os.path.join(self.hparams.sampling.data_dir, f"mp_20/num_nodes_bincount.pt"),
            #         map_location="cpu",
            #     ),
            #     requires_grad=False,
            # ),
            "qm9": torch.nn.Parameter(
                torch.load(
                    os.path.join(self.hparams.sampling.data_dir, f"qm9/num_nodes_bincount.pt"),
                    map_location="cpu",
                ),
                requires_grad=False,
            ),
            # "qmof150": torch.nn.Parameter(
            #     torch.load(
            #         os.path.join(self.hparams.sampling.data_dir, f"qmof/num_nodes_bincount.pt"),
            #         map_location="cpu",
            #     ),
            #     requires_grad=False,
            # ),
        }
        self.spacegroups_bincount = {
            # "mp20": torch.nn.Parameter(
            #     torch.load(
            #         os.path.join(self.hparams.sampling.data_dir, f"mp_20/spacegroups_bincount.pt"),
            #         map_location="cpu",
            #     ),
            #     requires_grad=False,
            # ),
            "qm9": None,
            # "qmof150": None,
        }

        # Dynamically detect active datasets based on metrics
        self.datasets = list(self.val_metrics.keys())
        self.dataset_to_idx = {d: i for i, d in enumerate(self.datasets)}

        # Energy-aware extension
        # Energy prediction head maps latent vector -> scalar energy
        if hasattr(self.autoencoder, "latent_dim"):
            latent_dim = self.autoencoder.latent_dim
        else:
            raise AttributeError(
                "autoencoder is missing latent_dim attribute. Please add `self.latent_dim = latent_dim` to the VAE."
            )

        self.energy_head = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, self.hparams.energy_head_hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hparams.energy_head_hidden_dim, 1),
        )

    def forward(self, batch: Data, sample_posterior: bool = True):
        # Encode batch to latent space
        with torch.no_grad():
            encoded_batch = self.autoencoder.encode(batch)
            if sample_posterior:
                encoded_batch["x"] = encoded_batch["posterior"].sample()
            else:
                encoded_batch["x"] = encoded_batch["posterior"].mode()
            x_1 = encoded_batch["x"]

            # Convert from PyG batch to dense batch with padding
            x_1, mask = to_dense_batch(x_1, encoded_batch["batch"])
            dense_encoded_batch = {"x_1": x_1, "token_mask": mask, "diffuse_mask": mask}

        # Corrupt batch using the interpolant
        self.interpolant.device = dense_encoded_batch["x_1"].device
        noisy_dense_encoded_batch = self.interpolant.corrupt_batch(dense_encoded_batch)

        # Prepare conditioning inputs to forward pass
        dataset_idx = batch.dataset_idx + 1  # 0 -> null class
        # if not self.hparams.conditioning.dataset_idx:
        #     dataset_idx = torch.zeros_like(dataset_idx)
        spacegroup = batch.spacegroup
        if not self.hparams.conditioning.spacegroup:
            spacegroup = torch.zeros_like(batch.spacegroup)

        # Use self-conditioning for ~half training batches
        if (
            self.interpolant.self_condition
            and random.random() < self.interpolant.self_condition_prob
        ):
            with torch.no_grad():
                x_sc = self.denoiser(
                    x=noisy_dense_encoded_batch["x_t"],
                    t=noisy_dense_encoded_batch["t"],
                    dataset_idx=dataset_idx,
                    spacegroup=spacegroup,
                    mask=mask,
                    x_sc=None,
                )
        else:
            x_sc = None

        # Run denoiser model
        pred_x = self.denoiser(
            x=noisy_dense_encoded_batch["x_t"],
            t=noisy_dense_encoded_batch["t"],
            dataset_idx=dataset_idx,
            spacegroup=spacegroup,
            mask=mask,
            x_sc=x_sc,
        )

        return pred_x, noisy_dense_encoded_batch

    def criterion(
        self,
        noisy_dense_encoded_batch: Dict[str, torch.Tensor],
        pred_x: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        # Compute MSE loss w/ masking for padded tokens
        gt_x_1 = noisy_dense_encoded_batch["x_1"]
        # norm_scale = 1 - torch.min(noisy_dense_encoded_batch["t"].unsqueeze(-1), torch.tensor(0.9))  # causes bradcasting issue
        norm_scale = 1 - torch.min(
            noisy_dense_encoded_batch["t"].view(-1, 1, 1), torch.tensor(0.9)
        )
        x_error = (gt_x_1 - pred_x) / norm_scale
        loss_mask = (
            noisy_dense_encoded_batch["token_mask"] * noisy_dense_encoded_batch["diffuse_mask"]
        )
        loss_denom = torch.sum(loss_mask, dim=-1) * pred_x.size(-1)
        x_loss = torch.sum(x_error**2 * loss_mask[..., None], dim=(-1, -2)) / loss_denom
        loss_dict = {"loss": x_loss.mean(), "x_loss": x_loss}

        # add diffusion loss stratified across t
        num_bins = 4
        flat_losses = x_loss.detach().cpu().numpy().flatten()
        flat_t = noisy_dense_encoded_batch["t"].detach().cpu().numpy().flatten()
        bin_edges = np.linspace(0.0, 1.0 + 1e-3, num_bins + 1)
        bin_idx = np.sum(bin_edges[:, None] <= flat_t[None, :], axis=0) - 1
        t_binned_loss = np.bincount(bin_idx, weights=flat_losses)
        t_binned_n = np.bincount(bin_idx)
        for t_bin in np.unique(bin_idx).tolist():
            bin_start = bin_edges[t_bin]
            bin_end = bin_edges[t_bin + 1]
            t_range = f"x_loss t=[{int(bin_start*100)},{int(bin_end*100)})"
            range_loss = t_binned_loss[t_bin] / t_binned_n[t_bin]
            loss_dict[t_range] = range_loss
        loss_dict["t_avg"] = np.mean(flat_t)

        return loss_dict

    #####################################################################################################

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        for dataset in self.val_metrics.keys():
            for metric in self.val_metrics[dataset].values():
                metric.reset()

    def on_train_epoch_start(self) -> None:
        """Lightning hook that is called when a training epoch starts."""
        for metric in self.train_metrics.values():
            metric.reset()

    def training_step(self, batch: Data, batch_idx: int) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        with torch.no_grad():
            # save masks used to apply augmentations
            # sample_is_periodic = batch.dataset_idx != DATASET_TO_IDX["qm9"]
            # node_is_periodic = sample_is_periodic[batch.batch]
            sample_is_qm9 = batch.dataset_idx == self.dataset_to_idx["qm9"]
            node_is_qm9 = sample_is_qm9[batch.batch]
            node_is_periodic = ~node_is_qm9  # only periodic atoms need frac_coords recomputation

            # if self.hparams.augmentations.frac_coords == True:
            #     if node_is_periodic.any():
            #         # sample random translation vector from batch length distribution / 2
            #         random_translation = (
            #             torch.normal(
            #                 torch.abs(batch.lengths.mean(dim=0)),
            #                 torch.abs(batch.lengths.std(dim=0)) + 1e-8,
            #             )
            #             / 2
            #         )
            #         # apply same random translation to all Cartesian coordinates
            #         pos_aug = batch.pos + random_translation
            #         batch.pos = pos_aug
            #         # compute new fractional coordinates for samples which are periodic
            #         cell_per_node_inv = torch.linalg.inv(batch.cell[batch.batch][node_is_periodic])
            #         frac_coords_aug = torch.einsum(
            #             "bi,bij->bj", batch.pos[node_is_periodic], cell_per_node_inv
            #         )
            #         frac_coords_aug = frac_coords_aug % 1.0
            #         batch.frac_coords[node_is_periodic] = frac_coords_aug

            if self.hparams.augmentations.pos == True:
                rot_mat = random_rotation_matrix(validate=True, device=self.device)
                pos_aug = batch.pos @ rot_mat.T
                batch.pos = pos_aug
                cell_aug = batch.cell @ rot_mat.T
                batch.cell = cell_aug
                # fractional coordinates are rotation invariant
                # assert torch.allclose(
                #     batch.frac_coords,
                #     torch.einsum("bi,bij->bj", pos_aug, torch.linalg.inv(cell_aug)[batch.batch]) % 1.0,
                #     rtol=1e-3,
                #     atol=1e-3,
                # )

        # forward pass
        pred_x, noisy_dense_encoded_batch = self.forward(batch)

        # calculate diffusion loss
        loss_dict = self.criterion(noisy_dense_encoded_batch, pred_x)

        # calculate energy-aware loss
        if self.hparams.lambda_energy > 0:
            z_latent = noisy_dense_encoded_batch["x_1"]  # shape: (B, T, D)
            z_mask = noisy_dense_encoded_batch["token_mask"]  # shape: (B, T)
            z_sum = (z_latent * z_mask.unsqueeze(-1)).sum(dim=1)  # (B, D)
            z_count = z_mask.sum(dim=1, keepdim=True).clamp(min=1)
            z_mean = z_sum / z_count  # (B, D)
            energy_pred = self.energy_head(z_mean).squeeze(-1)  # (B,)

            if hasattr(batch, "energy"):
                energy_gt = batch.energy.to(energy_pred.device)  # (B,)
                # scale energy loss to avoid exploding gradients
                energy_loss = 1e-08 * F.mse_loss(energy_pred, energy_gt)
                weighted_energy_loss = self.hparams.lambda_energy * energy_loss
                loss_dict["energy_loss"] = energy_loss
                loss_dict["loss"] = loss_dict["loss"] + weighted_energy_loss
            else:
                log.warning("Energy labels not found in batch. Skipping energy loss.")

        # log relative proportions of datasets in batch
        loss_dict["dataset_idx"] = batch.dataset_idx.detach().flatten()

        # update and log train metrics
        for k, v in loss_dict.items():
            self.train_metrics[k](v)
            self.log(
                f"train/{k}",
                self.train_metrics[k],
                on_step=True,
                on_epoch=False,
                prog_bar=False if k != "loss" else True,
            )

        # return loss or backpropagation will fail
        return loss_dict["loss"]

    #####################################################################################################

    def on_validation_epoch_start(self) -> None:
        self.on_evaluation_epoch_start(stage="val")

    def validation_step(self, batch: Data, batch_idx: int, dataloader_idx: int = 0) -> None:
        self.evaluation_step(batch, batch_idx, dataloader_idx, stage="val")

    def on_validation_epoch_end(self) -> None:
        self.on_evaluation_epoch_end(stage="val")

    #####################################################################################################

    def on_test_epoch_start(self) -> None:
        self.on_evaluation_epoch_start(stage="test")

    def test_step(self, batch: Data, batch_idx: int, dataloader_idx: int = 0) -> None:
        self.evaluation_step(batch, batch_idx, dataloader_idx, stage="test")

    def on_test_epoch_end(self) -> None:
        self.on_evaluation_epoch_end(stage="test")

    #####################################################################################################

    def on_evaluation_epoch_start(self, stage: Literal["val", "test"]) -> None:
        "Lightning hook that is called when a validation/test epoch starts."
        if stage not in ["val", "test"]:
            raise ValueError("stage must be 'val' or 'test'.")
        metrics = getattr(self, f"{stage}_metrics")
        for dataset in metrics.keys():
            for metric in metrics[dataset].values():
                metric.reset()
        generation_evaluators = getattr(self, f"{stage}_generation_evaluators")
        for dataset in generation_evaluators.keys():
            generation_evaluators[dataset].clear()  # clear lists for next epoch

    def evaluation_step(
        self,
        batch: Data,
        batch_idx: int,
        dataloader_idx: int,
        stage: Literal["val", "test"],
    ) -> None:
        """Perform a single evaluation step on a batch of data from the validation/test set."""

        if stage not in ["val", "test"]:
            raise ValueError("stage must be 'val' or 'test'.")
        # metrics = getattr(self, f"{stage}_metrics")[IDX_TO_DATASET[dataloader_idx]]
        # generation_evaluator = getattr(self, f"{stage}_generation_evaluators")[
        #     IDX_TO_DATASET[dataloader_idx]
        # ]
        dataset = self.datasets[dataloader_idx]
        metrics = getattr(self, f"{stage}_metrics")[dataset]
        generation_evaluator = getattr(self, f"{stage}_generation_evaluators")[dataset]
        generation_evaluator.device = metrics["loss"].device

        # forward pass
        pred_x, noisy_dense_encoded_batch = self.forward(batch)

        # calculate diffusion loss
        loss_dict = self.criterion(noisy_dense_encoded_batch, pred_x)

        # calculate energy-aware loss ===
        if self.hparams.lambda_energy > 0:
            z_latent = noisy_dense_encoded_batch["x_1"]  # shape: (B, T, D)
            z_mask = noisy_dense_encoded_batch["token_mask"]  # shape: (B, T)
            z_sum = (z_latent * z_mask.unsqueeze(-1)).sum(dim=1)  # (B, D)
            z_count = z_mask.sum(dim=1, keepdim=True).clamp(min=1)
            z_mean = z_sum / z_count  # (B, D)
            energy_pred = self.energy_head(z_mean).squeeze(-1)  # (B,)

            if hasattr(batch, "energy"):
                energy_gt = batch.energy.to(energy_pred.device)  # (B,)
                energy_loss = F.mse_loss(energy_pred, energy_gt)
                weighted_energy_loss = self.hparams.lambda_energy * energy_loss
                loss_dict["energy_loss"] = energy_loss
                loss_dict["loss"] = loss_dict["loss"] + weighted_energy_loss
            else:
                log.warning(
                    f"Energy labels not found in batch during {stage}. Skipping energy loss."
                )

        # update and log per-step val metrics
        for k, v in loss_dict.items():
            metrics[k](v)
            self.log(
                # f"{stage}_{IDX_TO_DATASET[dataloader_idx]}/{k}",
                f"{stage}_{dataset}/{k}",
                metrics[k],
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
                add_dataloader_idx=False,
            )

    def on_evaluation_epoch_end(self, stage: Literal["val", "test"]) -> None:
        """Lightning hook that is called when a validation/test epoch ends."""

        if stage not in ["val", "test"]:
            raise ValueError("stage must be 'val' or 'test'.")
        metrics = getattr(self, f"{stage}_metrics")
        generation_evaluators = getattr(self, f"{stage}_generation_evaluators")

        # for dataset in metrics.keys():
        for dataset in self.datasets:
            generation_evaluators[dataset].device = metrics[dataset]["loss"].device
            t_start = time.time()
            for samples_so_far in tqdm(
                range(0, self.hparams.sampling.num_samples, self.hparams.sampling.batch_size),
                desc=f"    Sampling",
            ):
                # Perform sampling and decoding to crystal structures
                out, batch, samples = self.sample_and_decode(
                    num_nodes_bincount=self.num_nodes_bincount[dataset],
                    spacegroups_bincount=self.spacegroups_bincount[dataset],
                    batch_size=self.hparams.sampling.batch_size,
                    cfg_scale=self.hparams.sampling.cfg_scale,
                    # dataset_idx=DATASET_TO_IDX[dataset],
                    dataset_idx=self.dataset_to_idx[dataset],
                )
                # Save predictions for metrics and visualisation
                start_idx = 0
                for idx_in_batch, num_atom in enumerate(batch["num_atoms"].tolist()):
                    _atom_types = (
                        out["atom_types"].narrow(0, start_idx, num_atom).argmax(dim=1)
                    )  # take argmax
                    _atom_types[_atom_types == 0] = 1  # atom type 0 -> 1 (H) to prevent crash
                    _pos = out["pos"].narrow(0, start_idx, num_atom) * 10.0  # nm to A
                    _frac_coords = out["frac_coords"].narrow(0, start_idx, num_atom)
                    _lengths = out["lengths"][idx_in_batch] * float(num_atom) ** (
                        1 / 3
                    )  # unscale lengths
                    _angles = torch.rad2deg(out["angles"][idx_in_batch])  # convert to degrees
                    generation_evaluators[dataset].append_pred_array(
                        {
                            "atom_types": _atom_types.detach().cpu().numpy(),
                            "pos": _pos.detach().cpu().numpy(),
                            "frac_coords": _frac_coords.detach().cpu().numpy(),
                            "lengths": _lengths.detach().cpu().numpy(),
                            "angles": _angles.detach().cpu().numpy(),
                            "sample_idx": samples_so_far
                            + self.global_rank * len(batch["num_atoms"])
                            + idx_in_batch,
                        }
                    )
                    start_idx = start_idx + num_atom
            t_end = time.time()

            # Compute generation metrics
            gen_metrics_dict = generation_evaluators[dataset].get_metrics(
                save=self.hparams.sampling.visualize,
                save_dir=self.hparams.sampling.save_dir + f"/{dataset}_{stage}_{self.global_rank}",
            )
            gen_metrics_dict["sampling_time"] = t_end - t_start
            for k, v in gen_metrics_dict.items():
                metrics[dataset][k](v)
                self.log(
                    f"{stage}_{dataset}/{k}",
                    metrics[dataset][k],
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False if k != "valid_rate" else True,
                    sync_dist=True,
                    add_dataloader_idx=False,
                )

            if self.hparams.sampling.visualize and type(self.logger) == WandbLogger:
                pred_table = generation_evaluators[dataset].get_wandb_table(
                    current_epoch=self.current_epoch,
                    save_dir=self.hparams.sampling.save_dir
                    + f"/{dataset}_{stage}_{self.global_rank}",
                )
                self.logger.experiment.log(
                    {f"{dataset}_{stage}_samples_table_device{self.global_rank}": pred_table}
                )

    #####################################################################################################

    def sample_and_decode(
        self,
        num_nodes_bincount,
        spacegroups_bincount,
        batch_size,
        cfg_scale=4.0,
        dataset_idx=0,
    ):
        # sample random lengths from distribution: (B, 1)
        sample_lengths = torch.multinomial(
            num_nodes_bincount.float(),
            batch_size,
            replacement=True,
        ).to(self.device)

        # create dataset_idx tensor
        # NOTE 0 -> null class within DiT, while 0 -> MP20 elsewhere, so increment by 1
        dataset_idx = torch.full(
            (batch_size,), dataset_idx + 1, dtype=torch.int64, device=self.device
        )

        # create spacegroup tensor
        if not self.hparams.conditioning.spacegroup or spacegroups_bincount is None:
            # null spacegroup
            spacegroup = torch.zeros(batch_size, dtype=torch.int64, device=self.device)
        else:
            # sample random spacegroups from distribution: (B, 1)
            spacegroup = torch.multinomial(
                spacegroups_bincount.float(),
                batch_size,
                replacement=True,
            ).to(self.device)

        # create token mask for visualization
        token_mask = torch.zeros(
            batch_size,
            max(sample_lengths),
            dtype=torch.bool,
            device=self.device,
        )
        for idx, length in enumerate(sample_lengths):
            token_mask[idx, :length] = True

        # create new samples from interpolant
        samples = self.interpolant.sample_with_classifier_free_guidance(
            batch_size=batch_size,
            num_tokens=max(sample_lengths),
            emb_dim=self.denoiser.d_x,
            model=self.denoiser,
            dataset_idx=dataset_idx,
            spacegroup=spacegroup,
            cfg_scale=cfg_scale,
            token_mask=token_mask,
        )
        # get final samples and remove padding (to PyG format)
        x = samples["clean_traj"][-1][token_mask]

        batch = {
            "x": x,
            "num_atoms": sample_lengths,
            "batch": torch.repeat_interleave(
                torch.arange(len(sample_lengths), device=self.device), sample_lengths
            ),
            "token_idx": (torch.cumsum(token_mask, dim=-1, dtype=torch.int64) - 1)[token_mask],
        }
        # decode samples to crystal structures using frozen decoder
        out = self.autoencoder.decode(batch)
        return out, batch, samples

    #####################################################################################################

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        try:
            # Clear cache for Equiformer SO3 embeddings
            self.autoencoder.encoder.mappingReduced.device = self.device
            self.autoencoder.encoder.mappingReduced.mask_indices_cache = None
            self.autoencoder.encoder.mappingReduced.rotate_inv_rescale_cache = None
            for rotation_module in self.autoencoder.encoder.SO3_rotation:
                rotation_module.mapping.device = self.device
                rotation_module.mapping.mask_indices_cache = None
                rotation_module.mapping.rotate_inv_rescale_cache = None
            log.info("Clear Equiformer checkpoint SO3 rotation mapping cache.")
        except AttributeError:
            pass

        if self.hparams.compile and stage == "fit":
            self.autoencoder = torch.compile(self.autoencoder)
            self.denoiser = torch.compile(self.denoiser)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_mp20/valid_rate",
                    "interval": "epoch",
                    "frequency": self.hparams.scheduler_frequency,
                },
            }
        return {"optimizer": optimizer}
