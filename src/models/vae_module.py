"""Copyright (c) Meta Platforms, Inc. and affiliates."""

import copy
from typing import Any, Dict, Literal, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from lightning import LightningModule
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig
from torch.nn import ModuleDict
from torch_geometric.data import Data
from torch_scatter import scatter
from torchmetrics import MeanMetric

from src.eval.crystal_reconstruction import CrystalReconstructionEvaluator
from src.eval.mof_reconstruction import MOFReconstructionEvaluator
from src.eval.molecule_reconstruction import MoleculeReconstructionEvaluator
from src.models.components.kabsch_utils import (
    differentiable_kabsch,
    random_rotation_matrix,
    rototranslate,
)
from src.utils import pylogger

log = pylogger.RankedLogger(__name__)


DATASET_TO_IDX = {
    "mp20": 0,  # periodic
    "qm9": 1,  # non-periodic
    "qmof150": 0,  # periodic
}


class DiagonalGaussianDistribution:
    """Diagonal Gaussian distribution with mean and logvar parameters.

    Adapted from: https://github.com/CompVis/latent-diffusion, with modifications for our tensors,
    which are of shape (N, d) instead of (B, H, W, d) for 2D images.
    """

    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=-1)  # split along channel dim
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(device=self.parameters.device)

    def sample(self):
        x = self.mean + self.std * torch.randn(self.mean.shape).to(device=self.parameters.device)
        return x

    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.0])
        else:
            if other is None:
                return 0.5 * torch.sum(
                    torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar, dim=1
                )
            else:
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var
                    - 1.0
                    - self.logvar
                    + other.logvar,
                    dim=1,
                )

    def mode(self):
        return self.mean

    def __repr__(self):
        return f"DiagonalGaussianDistribution(mean={self.mean}, logvar={self.logvar})"


class VariationalAutoencoderLitModule(LightningModule):
    """LightningModule for autoencoding 3D atomic systems. Implements VAE loss.

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
        datasets: DictConfig,
        encoder: torch.nn.Module,
        decoder: torch.nn.Module,
        latent_dim: int,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        scheduler_frequency: int,
        loss_weights: Dict,
        augmentations: DictConfig,
        visualization: DictConfig,
        compile: bool,
    ) -> None:
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # encoder and decoder models
        self.encoder = encoder
        self.decoder = decoder

        # quantization layers (following naming convention from Latent Diffusion)
        self.quant_conv = torch.nn.Linear(self.encoder.d_model, 2 * latent_dim, bias=False)
        self.post_quant_conv = torch.nn.Linear(latent_dim, self.decoder.d_model, bias=False)
        # NOTE these layers actually output the mean and logvar of the posterior distribution

        # weights for scaling loss functions per dataset type
        self.loss_weights = loss_weights
        self.loss_weights_atom_types = torch.nn.Parameter(
            torch.tensor([*self.loss_weights["loss_atom_types"].values()]),
            requires_grad=False,
        )
        self.loss_weights_lengths = torch.nn.Parameter(
            torch.tensor([*self.loss_weights["loss_lengths"].values()]),
            requires_grad=False,
        )
        self.loss_weights_angles = torch.nn.Parameter(
            torch.tensor([*self.loss_weights["loss_angles"].values()]),
            requires_grad=False,
        )
        self.loss_weights_frac_coords = torch.nn.Parameter(
            torch.tensor([*self.loss_weights["loss_frac_coords"].values()]),
            requires_grad=False,
        )
        self.loss_weights_pos = torch.nn.Parameter(
            torch.tensor([*self.loss_weights["loss_pos"].values()]),
            requires_grad=False,
        )
        self.loss_weights_kl = torch.nn.Parameter(
            torch.tensor([*self.loss_weights["loss_kl"].values()]),
            requires_grad=False,
        )

        # metric objects for calculating and averaging across batches
        self.train_metrics = ModuleDict(
            {
                "loss": MeanMetric(),
                "loss_atom_types": MeanMetric(),
                "loss_lengths": MeanMetric(),
                "loss_angles": MeanMetric(),
                "loss_frac_coords": MeanMetric(),
                "loss_pos": MeanMetric(),
                "loss_kl": MeanMetric(),
                "unscaled/loss_atom_types": MeanMetric(),
                "unscaled/loss_lengths": MeanMetric(),
                "unscaled/loss_angles": MeanMetric(),
                "unscaled/loss_frac_coords": MeanMetric(),
                "unscaled/loss_pos": MeanMetric(),
                "unscaled/loss_kl": MeanMetric(),
                "dataset_idx": MeanMetric(),
            }
        )

        # evaluator objects for computing metrics
        self.val_reconstruction_evaluators = {}
        self.val_metrics = ModuleDict()
        if self.hparams.datasets.mp20.proportion>0:
            self.val_reconstruction_evaluators['mp20'] = CrystalReconstructionEvaluator()
            self.val_metrics['mp20'] = ModuleDict(
                    {
                        "loss": MeanMetric(),
                        "loss_atom_types": MeanMetric(),
                        "loss_lengths": MeanMetric(),
                        "loss_angles": MeanMetric(),
                        "loss_frac_coords": MeanMetric(),
                        "loss_pos": MeanMetric(),
                        "loss_kl": MeanMetric(),
                        "unscaled/loss_atom_types": MeanMetric(),
                        "unscaled/loss_lengths": MeanMetric(),
                        "unscaled/loss_angles": MeanMetric(),
                        "unscaled/loss_frac_coords": MeanMetric(),
                        "unscaled/loss_pos": MeanMetric(),
                        "unscaled/loss_kl": MeanMetric(),
                        "match_rate": MeanMetric(),
                        "rms_dist": MeanMetric(),
                    }
                )

        if self.hparams.datasets.qm9.proportion>0:
            self.val_reconstruction_evaluators['qm9'] = MoleculeReconstructionEvaluator()
            self.val_metrics['qm9'] = ModuleDict(
                    {
                        "loss": MeanMetric(),
                        "loss_atom_types": MeanMetric(),
                        "loss_lengths": MeanMetric(),
                        "loss_angles": MeanMetric(),
                        "loss_frac_coords": MeanMetric(),
                        "loss_pos": MeanMetric(),
                        "loss_kl": MeanMetric(),
                        "unscaled/loss_atom_types": MeanMetric(),
                        "unscaled/loss_lengths": MeanMetric(),
                        "unscaled/loss_angles": MeanMetric(),
                        "unscaled/loss_frac_coords": MeanMetric(),
                        "unscaled/loss_pos": MeanMetric(),
                        "unscaled/loss_kl": MeanMetric(),
                        "match_rate": MeanMetric(),
                        "rms_dist": MeanMetric(),
                    }
                )

        if self.hparams.datasets.qmof150.proportion>0:
            self.val_reconstruction_evaluators['qmof150'] = MOFReconstructionEvaluator()
            self.val_metrics['qmof150'] = ModuleDict(
                    {
                        "loss": MeanMetric(),
                        "loss_atom_types": MeanMetric(),
                        "loss_lengths": MeanMetric(),
                        "loss_angles": MeanMetric(),
                        "loss_frac_coords": MeanMetric(),
                        "loss_pos": MeanMetric(),
                        "loss_kl": MeanMetric(),
                        "unscaled/loss_atom_types": MeanMetric(),
                        "unscaled/loss_lengths": MeanMetric(),
                        "unscaled/loss_angles": MeanMetric(),
                        "unscaled/loss_frac_coords": MeanMetric(),
                        "unscaled/loss_pos": MeanMetric(),
                        "unscaled/loss_kl": MeanMetric(),
                        "match_rate": MeanMetric(),
                        "rms_dist": MeanMetric(),
                    }
                )

        self.test_reconstruction_evaluators = copy.deepcopy(self.val_reconstruction_evaluators)
        self.test_metrics = copy.deepcopy(self.val_metrics)
        self.init_idx_to_dataset()

    def init_idx_to_dataset(self, ):
        datasets = [
            ("mp20", self.hparams.datasets.mp20.proportion),
            ("qm9", self.hparams.datasets.qm9.proportion),
            ("qmof150", self.hparams.datasets.qmof150.proportion)
        ]
        filtered_datasets = [name for name, prop in datasets if prop > 0]
        self.IDX_TO_DATASET = {idx: name for idx, name in enumerate(filtered_datasets)}

    def encode(self, batch):
        encoded_batch = self.encoder(batch)
        encoded_batch["moments"] = self.quant_conv(encoded_batch["x"])
        encoded_batch["posterior"] = DiagonalGaussianDistribution(encoded_batch["moments"])
        return encoded_batch

    def decode(self, encoded_batch):
        encoded_batch["x"] = self.post_quant_conv(encoded_batch["x"])
        out = self.decoder(encoded_batch)
        return out

    def forward(self, batch: Data, sample_posterior: bool = True):
        encoded_batch = self.encode(batch)
        if sample_posterior:
            encoded_batch["x"] = encoded_batch["posterior"].sample()
        else:
            encoded_batch["x"] = encoded_batch["posterior"].mode()
        out = self.decode(encoded_batch)
        return out, encoded_batch

    #####################################################################################################

    def reconstruction_criterion(
        self, batch: Data, out: Dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        # Atom types loss
        loss_atom_types = F.cross_entropy(out["atom_types"], batch.atom_types, reduction="none")

        # Lattice lengths loss, after scaling by num_atoms**(1/3)
        loss_lengths = F.mse_loss(out["lengths"], batch.lengths_scaled, reduction="none").mean(
            dim=1
        )

        # Lattice angles loss, in radians
        loss_angles = F.mse_loss(out["angles"], batch.angles_radians, reduction="none").mean(dim=1)

        # Fractional coordinates loss
        loss_frac_coords = F.mse_loss(
            out["frac_coords"], batch.frac_coords, reduction="none"
        ).mean(dim=1)

        # Coordinates loss after zero-centering, use nm as unit (not A)
        pos_pred = out["pos"]
        pos_true = batch.pos / 10.0  # nm to A
        pos_mean_pred = scatter(pos_pred, batch.batch, dim=0, reduce="mean")[batch.batch]
        pos_mean_true = scatter(pos_true, batch.batch, dim=0, reduce="mean")[batch.batch]
        loss_pos = F.mse_loss(
            pos_pred - pos_mean_pred, pos_true - pos_mean_true, reduction="none"
        ).mean(dim=1)

        return {
            "loss_atom_types": loss_atom_types,
            "loss_lengths": loss_lengths,
            "loss_angles": loss_angles,
            "loss_frac_coords": loss_frac_coords,
            "loss_pos": loss_pos,
        }

    def criterion(
        self, batch: Data, encoded_batch: Dict[str, torch.Tensor], out: Dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        # Reconstruction loss
        loss_reconst = self.reconstruction_criterion(batch, out)

        # KL divergence loss
        loss_kl = encoded_batch["posterior"].kl()

        # Assign loss_weights tensors based on dataset_idx attribute in batch
        weights_atom_types = self.loss_weights_atom_types[batch.dataset_idx[batch.batch]]
        weights_lengths = self.loss_weights_lengths[batch.dataset_idx]
        weights_angles = self.loss_weights_angles[batch.dataset_idx]
        weights_frac_coords = self.loss_weights_frac_coords[batch.dataset_idx[batch.batch]]
        weights_pos = self.loss_weights_pos[batch.dataset_idx[batch.batch]]
        weights_kl = self.loss_weights_kl[batch.dataset_idx[batch.batch]]

        loss = (
            (weights_atom_types * loss_reconst["loss_atom_types"]).mean()
            + (weights_lengths * loss_reconst["loss_lengths"]).mean()
            + (weights_angles * loss_reconst["loss_angles"]).mean()
            + (weights_frac_coords * loss_reconst["loss_frac_coords"]).mean()
            + (weights_pos * loss_reconst["loss_pos"]).mean()
            + (weights_kl * loss_kl).mean()
        )

        return {
            "loss": loss,
            "loss_atom_types": weights_atom_types * loss_reconst["loss_atom_types"],
            "loss_lengths": weights_lengths * loss_reconst["loss_lengths"],
            "loss_angles": weights_angles * loss_reconst["loss_angles"],
            "loss_frac_coords": weights_frac_coords * loss_reconst["loss_frac_coords"],
            "loss_pos": weights_pos * loss_reconst["loss_pos"],
            "loss_kl": weights_kl * loss_kl,
            "unscaled/loss_atom_types": loss_reconst["loss_atom_types"],
            "unscaled/loss_lengths": loss_reconst["loss_lengths"],
            "unscaled/loss_angles": loss_reconst["loss_angles"],
            "unscaled/loss_frac_coords": loss_reconst["loss_frac_coords"],
            "unscaled/loss_pos": loss_reconst["loss_pos"],
            "unscaled/loss_kl": loss_kl,
        }

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
            sample_is_periodic = batch.dataset_idx == DATASET_TO_IDX["mp20"]
            node_is_periodic = sample_is_periodic[batch.batch]

            if self.hparams.augmentations.frac_coords == True:
                if node_is_periodic.any():
                    # sample random translation vector from batch length distribution / 2
                    random_translation = (
                        torch.normal(
                            torch.abs(batch.lengths.mean(dim=0)),
                            torch.abs(batch.lengths.std(dim=0)) + 1e-8,
                        )
                        / 2
                    )
                    # apply same random translation to all Cartesian coordinates
                    pos_aug = batch.pos + random_translation
                    batch.pos = pos_aug
                    # compute new fractional coordinates for samples which are periodic
                    cell_per_node_inv = torch.linalg.inv(batch.cell[batch.batch][node_is_periodic])
                    frac_coords_aug = torch.einsum(
                        "bi,bij->bj", batch.pos[node_is_periodic], cell_per_node_inv
                    )
                    frac_coords_aug = frac_coords_aug % 1.0
                    batch.frac_coords[node_is_periodic] = frac_coords_aug

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

            if self.hparams.augmentations.noise > 0.0:
                total_atoms = batch.num_atoms.sum().item()
                # select X% of atom types to be perturbed
                perturbed_idx = torch.tensor(
                    np.random.choice(
                        total_atoms,
                        int(total_atoms * self.hparams.augmentations.noise),
                        replace=False,
                    ),
                    device=self.device,
                )
                # save original atom types
                atom_types_ = batch.atom_types.clone()
                # set perturbed atom types to 0
                batch.atom_types[perturbed_idx] = 0

                # select X% of positions to be perturbed (new, may overlap)
                perturbed_idx = torch.tensor(
                    np.random.choice(
                        total_atoms,
                        int(total_atoms * self.hparams.augmentations.noise),
                        replace=False,
                    ),
                    device=self.device,
                )
                # save original positions and fractional coordinates
                pos_ = batch.pos.clone()
                frac_coords_ = batch.frac_coords.clone()
                # add random noise to perturbed positions
                corruption_scale = 0.1
                noise = (
                    torch.randn_like(batch.pos[perturbed_idx], device=self.device)
                    * corruption_scale
                )
                batch.pos[perturbed_idx] += noise
                # compute new fractional coordinates for samples which are periodic
                if node_is_periodic.any():
                    cell_per_node_inv = torch.linalg.inv(batch.cell[batch.batch][node_is_periodic])
                    frac_coords_aug = torch.einsum(
                        "bi,bij->bj", batch.pos[node_is_periodic], cell_per_node_inv
                    )
                    frac_coords_aug = frac_coords_aug % 1.0
                    batch.frac_coords[node_is_periodic] = frac_coords_aug

        # forward pass
        out, encoded_batch = self.forward(batch)

        # undo noise augmentation before calculating loss
        if self.hparams.augmentations.noise > 0.0:
            batch.atom_types = atom_types_
            batch.pos = pos_
            batch.frac_coords = frac_coords_

        # calculate loss
        loss_dict = self.criterion(batch, encoded_batch, out)

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
        reconstruction_evaluators = getattr(self, f"{stage}_reconstruction_evaluators")
        for dataset in reconstruction_evaluators.keys():
            reconstruction_evaluators[dataset].clear()  # clear lists for next epoch

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
        metrics = getattr(self, f"{stage}_metrics")[self.IDX_TO_DATASET[dataloader_idx]]
        reconstruction_evaluator = getattr(self, f"{stage}_reconstruction_evaluators")[
            self.IDX_TO_DATASET[dataloader_idx]
        ]
        reconstruction_evaluator.device = metrics["loss"].device

        # forward pass
        out, encoded_batch = self.forward(batch)

        # calculate loss
        loss_dict = self.criterion(batch, encoded_batch, out)

        # update and log per-step val metrics
        for k, v in loss_dict.items():
            metrics[k](v)
            self.log(
                f"{stage}_{self.IDX_TO_DATASET[dataloader_idx]}/{k}",
                metrics[k],
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
                add_dataloader_idx=False,
            )

        # Save predictions for metrics and visualisation (see validation_epoch_end)
        start_idx = 0
        for idx_in_batch, num_atom in enumerate(batch.num_atoms.tolist()):
            _atom_types = (
                out["atom_types"].narrow(0, start_idx, num_atom).argmax(dim=1)
            )  # take argmax
            _atom_types[_atom_types == 0] = 1  # atom type 0 -> 1 (H) to prevent crash
            _pos = out["pos"].narrow(0, start_idx, num_atom) * 10.0  # nm to A
            _frac_coords = out["frac_coords"].narrow(0, start_idx, num_atom)
            _lengths = out["lengths"][idx_in_batch] * float(num_atom) ** (1 / 3)  # unscale lengths
            _angles = torch.rad2deg(out["angles"][idx_in_batch])  # convert to degrees
            reconstruction_evaluator.append_pred_array(
                {
                    "atom_types": _atom_types.detach().cpu().numpy(),
                    "pos": _pos.detach().cpu().numpy(),
                    "frac_coords": _frac_coords.detach().cpu().numpy(),
                    "lengths": _lengths.detach().cpu().numpy(),
                    "angles": _angles.detach().cpu().numpy(),
                    "sample_idx": (batch_idx + self.global_rank) * batch.batch_size + idx_in_batch,
                }
            )
            start_idx = start_idx + num_atom

        # Save groundtruths for metrics and visualisation (see validation_epoch_end)
        for idx_in_batch, _data in enumerate(batch.to_data_list()):
            reconstruction_evaluator.append_gt_array(
                {
                    "atom_types": _data["atom_types"].detach().cpu().numpy(),
                    "pos": _data["pos"].detach().cpu().numpy(),
                    "frac_coords": _data["frac_coords"].detach().cpu().numpy(),
                    "lengths": _data["lengths"].detach().cpu().numpy(),
                    "angles": _data["angles"].detach().cpu().numpy(),
                    "sample_idx": (batch_idx + self.global_rank) * batch.batch_size + idx_in_batch,
                }
            )

    def on_evaluation_epoch_end(self, stage: Literal["val", "test"]) -> None:
        "Lightning hook that is called when a validation/test epoch ends."

        if stage not in ["val", "test"]:
            raise ValueError("stage must be 'val' or 'test'.")
        metrics = getattr(self, f"{stage}_metrics")
        reconstruction_evaluators = getattr(self, f"{stage}_reconstruction_evaluators")
        for dataset in metrics.keys():
            reconstruction_evaluators[dataset].device = metrics[dataset]["loss"].device

        # Compute reconstruction metrics
        for dataset in metrics.keys():
            rec_metrics_dict = reconstruction_evaluators[dataset].get_metrics(
                save=self.hparams.visualization.visualize,
                save_dir=self.hparams.visualization.save_dir
                + f"/{dataset}_{stage}_{self.global_rank}",
            )
            for k, v in rec_metrics_dict.items():
                metrics[dataset][k](v)
                self.log(
                    f"{stage}_{dataset}/{k}",
                    metrics[dataset][k],
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False if k != "match_rate" else True,
                    sync_dist=True,
                    add_dataloader_idx=False,
                )

            if self.hparams.visualization.visualize and type(self.logger) == WandbLogger:
                pred_table = reconstruction_evaluators[dataset].get_wandb_table(
                    current_epoch=self.current_epoch,
                    save_dir=self.hparams.visualization.save_dir
                    + f"/{dataset}_{stage}_{self.global_rank}",
                )
                self.logger.experiment.log(
                    {f"{dataset}_{stage}_pred_table_device{self.global_rank}": pred_table}
                )

    #####################################################################################################

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            # self.net = torch.compile(self.net)
            self.encode = torch.compile(self.encode)
            self.decode = torch.compile(self.decode)
            self.quant_conv = torch.compile(self.quant_conv)
            self.post_quant_conv = torch.compile(self.post_quant_conv)

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
                    "monitor": "val_mp20/match_rate",
                    "interval": "epoch",
                    "frequency": self.hparams.scheduler_frequency,
                },
            }
        return {"optimizer": optimizer}