from functools import partial
from typing import Optional, Sequence

import torch
from lightning import LightningDataModule
from omegaconf import DictConfig
from torch.utils.data import ConcatDataset
from torch_geometric.data import Data
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader

from src.utils import pylogger

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


def custom_transform(data, removeHs=True):
    atoms_to_keep = torch.ones_like(data.z, dtype=torch.bool)
    num_atoms = data.num_nodes
    if removeHs:
        atoms_to_keep = data.z != 1
        num_atoms = atoms_to_keep.sum().item()

    return Data(
        id=f"qm9_{data.name}",
        atom_types=data.z[atoms_to_keep],
        pos=data.pos[atoms_to_keep],
        frac_coords=torch.zeros_like(data.pos[atoms_to_keep]),
        cell=torch.zeros((1, 3, 3)),
        lattices=torch.zeros(1, 6),
        lattices_scaled=torch.zeros(1, 6),
        lengths=torch.zeros(1, 3),
        lengths_scaled=torch.zeros(1, 3),
        angles=torch.zeros(1, 3),
        angles_radians=torch.zeros(1, 3),
        num_atoms=torch.LongTensor([num_atoms]),
        num_nodes=torch.LongTensor([num_atoms]),
        spacegroup=torch.zeros(1, dtype=torch.long),
        token_idx=torch.arange(num_atoms),
        energy=data.y[0, 7],  # Internal energy at 0K
        dataset_idx=torch.tensor([1], dtype=torch.long),
    )


class QM9DataModule(LightningDataModule):
    def __init__(
        self, datasets: DictConfig, num_workers: DictConfig, batch_size: DictConfig
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

    def setup(self, stage: Optional[str] = None) -> None:
        qm9_dataset = QM9(
            root=self.hparams.datasets.qm9.root,
            transform=partial(custom_transform, removeHs=self.hparams.datasets.qm9.removeHs),
        ).shuffle()

        self.qm9_train_dataset = qm9_dataset[:100000]
        self.qm9_val_dataset = qm9_dataset[100000:118000]
        self.qm9_test_dataset = qm9_dataset[118000:]

        self.qm9_train_dataset = self.qm9_train_dataset[
            : int(len(self.qm9_train_dataset) * self.hparams.datasets.qm9.proportion)
        ]
        self.qm9_val_dataset = self.qm9_val_dataset[
            : int(len(self.qm9_val_dataset) * self.hparams.datasets.qm9.proportion)
        ]
        self.qm9_test_dataset = self.qm9_test_dataset[
            : int(len(self.qm9_test_dataset) * self.hparams.datasets.qm9.proportion)
        ]

        if stage is None or stage in ["fit", "validate"]:
            self.train_dataset = ConcatDataset([self.qm9_train_dataset])
            log.info(
                f"Training dataset: {len(self.train_dataset)} samples (QM9: {len(self.qm9_train_dataset)})"
            )
            log.info(f"QM9 validation dataset: {len(self.qm9_val_dataset)} samples")

        if stage is None or stage in ["test", "predict"]:
            log.info(f"QM9 test dataset: {len(self.qm9_test_dataset)} samples")

        # log.info(f"QM9 train dataset: {len(self.qm9_train_dataset)} samples")
        # log.info(f"QM9 val dataset: {len(self.qm9_val_dataset)} samples")
        # log.info(f"QM9 test dataset: {len(self.qm9_test_dataset)} samples")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.qm9_train_dataset,
            batch_size=self.hparams.batch_size.train,
            num_workers=self.hparams.num_workers.train,
            pin_memory=False,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self) -> Sequence[DataLoader]:
        return [
            DataLoader(
                dataset=self.qm9_val_dataset,
                batch_size=self.hparams.batch_size.val,
                num_workers=self.hparams.num_workers.val,
                pin_memory=False,
                shuffle=False,
            )
        ]

    def test_dataloader(self) -> Sequence[DataLoader]:
        return [
            DataLoader(
                dataset=self.qm9_test_dataset,
                batch_size=self.hparams.batch_size.test,
                num_workers=self.hparams.num_workers.test,
                pin_memory=False,
                shuffle=False,
            )
        ]
