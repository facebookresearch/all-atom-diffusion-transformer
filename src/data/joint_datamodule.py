"""Copyright (c) Meta Platforms, Inc. and affiliates."""

import os
from functools import partial
from typing import Any, Dict, Optional, Sequence, Tuple

import torch
from lightning import LightningDataModule
from omegaconf import DictConfig
from torch.utils.data import ConcatDataset
from torch_geometric.data import Data
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader

from src.data.components.mp20_dataset import MP20
from src.data.components.qmof150_dataset import QMOF150
from src.utils import pylogger

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


def custom_transform(data, removeHs=True):
    atoms_to_keep = torch.ones_like(data.z, dtype=torch.bool)
    num_atoms = data.num_nodes
    if removeHs:
        atoms_to_keep = data.z != 1
        num_atoms = atoms_to_keep.sum().item()

    # PyG object attributes consistent with CrystalDataset
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
        num_nodes=torch.LongTensor([num_atoms]),  # special attribute used for PyG batching
        spacegroup=torch.zeros(1, dtype=torch.long),  # null spacegroup
        token_idx=torch.arange(num_atoms),
        dataset_idx=torch.tensor([1], dtype=torch.long),  # 1 --> indicates non-periodic/molecule
    )


class JointDataModule(LightningDataModule):
    """`LightningDataModule` for jointly training on 3D atomic datasets:

    - MP20: crystal structures
    - QM9: small molecules
    - QMOF150: metal-organic frameworks

    A `LightningDataModule` implements 7 key methods:

    ```python
        def prepare_data(self):
        # Things to do on 1 GPU/TPU (not on every GPU/TPU in DDP).
        # Download data, pre-process, split, save to disk, etc...

        def setup(self, stage):
        # Things to do on every process in DDP.
        # Load data, set variables, etc...

        def train_dataloader(self):
        # return train dataloader

        def val_dataloader(self):
        # return validation dataloader

        def test_dataloader(self):
        # return test dataloader

        def predict_dataloader(self):
        # return predict dataloader

        def teardown(self, stage):
        # Called on every process in DDP.
        # Clean up after fit or test.
    ```

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """

    def __init__(
        self,
        datasets: DictConfig,
        num_workers: DictConfig,
        batch_size: DictConfig,
    ) -> None:
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        # QM9 dataset
        qm9_dataset = QM9(
            root=self.hparams.datasets.qm9.root,
            transform=partial(custom_transform, removeHs=self.hparams.datasets.qm9.removeHs),
        ).shuffle()
        # # save num_nodes histogram for sampling from generative models
        # num_nodes = torch.tensor([data["num_nodes"] for data in qm9_dataset])
        # torch.save(
        #     torch.bincount(num_nodes),
        #     os.path.join(self.hparams.datasets.qm9.root, "num_nodes_bincount.pt"),
        # )
        # create train, val, test split
        self.qm9_train_dataset = qm9_dataset[:100000]
        self.qm9_val_dataset = qm9_dataset[100000:118000]
        self.qm9_test_dataset = qm9_dataset[118000:]
        # retain subset of dataset; can be used to train on only one dataset, too
        self.qm9_train_dataset = self.qm9_train_dataset[
            : int(len(self.qm9_train_dataset) * self.hparams.datasets.qm9.proportion)
        ]
        self.qm9_val_dataset = self.qm9_val_dataset[
            : int(len(self.qm9_val_dataset) * self.hparams.datasets.qm9.proportion)
        ]
        self.qm9_test_dataset = self.qm9_test_dataset[
            : int(len(self.qm9_test_dataset) * self.hparams.datasets.qm9.proportion)
        ]

        # MP20 dataset
        mp20_dataset = MP20(root=self.hparams.datasets.mp20.root)  # .shuffle()
        # # save num_nodes histogram for sampling from generative models
        # num_nodes = torch.tensor([data["num_nodes"] for data in mp20_dataset])
        # torch.save(
        #     torch.bincount(num_nodes),
        #     os.path.join(self.hparams.datasets.mp20.root, "num_nodes_bincount.pt"),
        # )
        # create train, val, test split
        self.mp20_train_dataset = mp20_dataset[:27138]
        self.mp20_val_dataset = mp20_dataset[27138 : 27138 + 9046]
        self.mp20_test_dataset = mp20_dataset[27138 + 9046 :]
        # retain subset of dataset; can be used to train on only one dataset, too
        self.mp20_train_dataset = self.mp20_train_dataset[
            : int(len(self.mp20_train_dataset) * self.hparams.datasets.mp20.proportion)
        ]
        self.mp20_val_dataset = self.mp20_val_dataset[
            : int(len(self.mp20_val_dataset) * self.hparams.datasets.mp20.proportion)
        ]
        self.mp20_test_dataset = self.mp20_test_dataset[
            : int(len(self.mp20_test_dataset) * self.hparams.datasets.mp20.proportion)
        ]

        # QMOF150 dataset
        qmof150_dataset = QMOF150(root=self.hparams.datasets.qmof150.root).shuffle()
        # # save num_nodes histogram for sampling from generative models
        # num_nodes = torch.tensor([data["num_nodes"] for data in qmof150_dataset])
        # torch.save(
        #     torch.bincount(num_nodes),
        #     os.path.join(self.hparams.datasets.qmof150.root, "num_nodes_bincount.pt"),
        # )
        # create train, val, test split
        self.qmof150_train_dataset = qmof150_dataset[2048:]
        self.qmof150_val_dataset = qmof150_dataset[:1024]
        self.qmof150_test_dataset = qmof150_dataset[1024:2048]
        # retain subset of dataset; can be used to train on only one dataset, too
        self.qmof150_train_dataset = self.qmof150_train_dataset[
            : int(len(self.qmof150_train_dataset) * self.hparams.datasets.qmof150.proportion)
        ]
        self.qmof150_val_dataset = self.qmof150_val_dataset[
            : int(len(self.qmof150_val_dataset) * self.hparams.datasets.qmof150.proportion)
        ]
        self.qmof150_test_dataset = self.qmof150_test_dataset[
            : int(len(self.qmof150_test_dataset) * self.hparams.datasets.qmof150.proportion)
        ]

        if stage is None or stage in ["fit", "validate"]:
            self.train_dataset = ConcatDataset(
                [self.mp20_train_dataset, self.qm9_train_dataset, self.qmof150_train_dataset]
            )
            log.info(
                f"Training dataset: {len(self.train_dataset)} samples (MP20: {len(self.mp20_train_dataset)}, QM9: {len(self.qm9_train_dataset)}, QMOF150: {len(self.qmof150_train_dataset)})"
            )
            log.info(f"MP20 validation dataset: {len(self.mp20_val_dataset)} samples")
            log.info(f"QM9 validation dataset: {len(self.qm9_val_dataset)} samples")
            log.info(f"QMOF150 validation dataset: {len(self.qmof150_val_dataset)} samples")

        if stage is None or stage in ["test", "predict"]:
            log.info(f"MP20 test dataset: {len(self.mp20_test_dataset)} samples")
            log.info(f"QM9 test dataset: {len(self.qm9_test_dataset)} samples")
            log.info(f"QMOF150 test dataset: {len(self.qmof150_test_dataset)} samples")

    def train_dataloader(self) -> DataLoader:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.hparams.batch_size.train,
            num_workers=self.hparams.num_workers.train,
            pin_memory=False,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self) -> Sequence[DataLoader]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return [
            DataLoader(
                dataset=self.mp20_val_dataset,
                batch_size=self.hparams.batch_size.val,
                num_workers=self.hparams.num_workers.val,
                pin_memory=False,
                shuffle=False,
            ),
            DataLoader(
                dataset=self.qm9_val_dataset,
                batch_size=self.hparams.batch_size.val,
                num_workers=self.hparams.num_workers.val,
                pin_memory=False,
                shuffle=False,
            ),
            DataLoader(
                dataset=self.qmof150_val_dataset,
                batch_size=self.hparams.batch_size.val,
                num_workers=self.hparams.num_workers.val,
                pin_memory=False,
                shuffle=False,
            ),
        ]

    def test_dataloader(self) -> Sequence[DataLoader]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return [
            DataLoader(
                dataset=self.mp20_test_dataset,
                batch_size=self.hparams.batch_size.test,
                num_workers=self.hparams.num_workers.test,
                pin_memory=False,
                shuffle=False,
            ),
            DataLoader(
                dataset=self.qm9_test_dataset,
                batch_size=self.hparams.batch_size.test,
                num_workers=self.hparams.num_workers.test,
                pin_memory=False,
                shuffle=False,
            ),
            DataLoader(
                dataset=self.qmof150_test_dataset,
                batch_size=self.hparams.batch_size.test,
                num_workers=self.hparams.num_workers.test,
                pin_memory=False,
                shuffle=False,
            ),
        ]
