#!/usr/bin/env python
"""Apply ICA solution to raw data excluding bad components"""
import logging
from dataclasses import dataclass

import hydra
import mne  # type: ignore
from hydra.core.config_store import ConfigStore
from utils import BaseConfig, prepare_script, read_ica_bads

logger = logging.getLogger(__file__)


@dataclass
class Input:
    raw: str
    bad_ics: str
    ica: str


@dataclass
class Output:
    raw: str


@dataclass
class Config(BaseConfig):
    input: Input
    output: Output


cs = ConfigStore.instance()
cs.store(name="schema", node=Config)


@hydra.main(config_path="../configs/", config_name="061-apply_ica")
def main(cfg: Config):
    prepare_script(logger, __file__)

    logger.info("Loading raw")
    raw = mne.io.read_raw_fif(cfg.input.raw, preload=True)

    logger.info("Loading ICA solution and setting up bad ICs")
    ica = mne.preprocessing.read_ica(cfg.input.ica)
    ica.exclude = read_ica_bads(cfg.input.bad_ics)

    logger.info(f"Excluding {ica.exclude}")
    ica.apply(raw)
    raw.save(cfg.output.raw, overwrite=True)


if __name__ == "__main__":
    main()
