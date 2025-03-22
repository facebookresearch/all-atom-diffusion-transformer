"""Copyright (c) Meta Platforms, Inc. and affiliates."""

from typing import Any, Dict

import hydra
from lightning_utilities.core.rank_zero import rank_zero_only
from omegaconf import OmegaConf

from src.utils import pylogger

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


@rank_zero_only
def log_hyperparameters(object_dict: Dict[str, Any]) -> None:
    """Controls which config parts are saved by Lightning loggers.

    Additionally saves:
        - Number of model parameters

    :param object_dict: A dictionary containing the following objects:
        - `"cfg"`: A DictConfig object containing the main config.
        - `"model"`: The Lightning model.
        - `"trainer"`: The Lightning trainer.
    """
    hparams = {}

    cfg = OmegaConf.to_container(object_dict["cfg"])
    model = object_dict["model"]
    trainer = object_dict["trainer"]

    if not trainer.logger:
        log.warning("Logger not found! Skipping hyperparameter logging...")
        return

    hparams["data"] = cfg["data"]
    hparams["trainer"] = cfg["trainer"]

    hparams["callbacks"] = cfg.get("callbacks")
    hparams["extras"] = cfg.get("extras")

    hparams["task_name"] = cfg.get("task_name")
    hparams["tags"] = cfg.get("tags")
    hparams["ckpt_path"] = cfg.get("ckpt_path")
    hparams["seed"] = cfg.get("seed")

    # save model hyperparameters
    if "autoencoder" in hparams["task_name"]:
        hparams["autoencoder_module"] = cfg["autoencoder_module"]
        hparams["encoder"] = cfg["encoder"]
        hparams["decoder"] = cfg["decoder"]
    elif "diffusion" in hparams["task_name"]:
        hparams["diffusion_module"] = cfg["diffusion_module"]
    else:
        raise ValueError(f"Task name {hparams['task_name']} not recognized!")

    # save number of model parameters
    hparams["model/params/total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params/trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    hparams["model/params/non_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )

    # hydra and wandb output directory
    # hparams["output_dir"] = cfg["paths"].get("output_dir")  # TODO how to resolve this?
    hparams["output_dir"] = hydra.core.hydra_config.HydraConfig.get()["runtime"]["output_dir"]

    # send hparams to all loggers
    for logger in trainer.loggers:
        logger.log_hyperparams(hparams)
