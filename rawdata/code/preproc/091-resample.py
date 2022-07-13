#!/usr/bin/env python
import logging
from dataclasses import dataclass

import hydra
from hydra.core.config_store import ConfigStore
from mne.io import read_raw_fif  # type: ignore
from utils import BaseConfig, prepare_script

logger = logging.getLogger(__file__)


@dataclass
class Input:
    raw: str


@dataclass
class Output:
    raw: str


@dataclass
class Config(BaseConfig):
    input: Input
    output: Output
    sfreq: float


cs = ConfigStore.instance()
cs.store(name="schema", node=Config)


@hydra.main(config_path="../configs/", config_name="091-resample")
def main(cfg: Config):
    prepare_script(logger, __file__)

    logger.info(f"Reading data from {cfg.input.raw}")
    raw = read_raw_fif(cfg.input.raw)

    logger.info("Resampling data")
    raw.resample(sfreq=cfg.sfreq)

    logger.info(f"Saving data to {cfg.output.raw}")
    raw.save(cfg.output.raw)


if __name__ == "__main__":
    main()
