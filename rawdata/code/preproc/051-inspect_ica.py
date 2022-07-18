#!/usr/bin/env python
"""Manually mark bad ICA components"""
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import hydra
import mne  # type: ignore
from hydra.core.config_store import ConfigStore
from mne.preprocessing import read_ica  # type: ignore
from utils import BaseConfig, prepare_script, read_ica_bads, write_ica_bads

logger = logging.getLogger(__file__)


@dataclass
class Input:
    raw: str
    ica: str
    annots: str


@dataclass
class Output:
    bad_ics: str


@dataclass
class FiltParams:
    l_freq: Optional[int]
    h_freq: Optional[int]


@dataclass
class Config(BaseConfig):
    input: Input
    output: Output
    filt: FiltParams


cs = ConfigStore.instance()
cs.store(name="schema", node=Config)


@hydra.main(config_path="../configs/", config_name="051-inspect_ica")
def main(cfg: Config):
    prepare_script(logger, script_name=__file__)

    logger.info("Loading and filtering raw data")
    raw = mne.io.read_raw_fif(cfg.input.raw, preload=True)
    raw.filter(l_freq=cfg.filt.l_freq, h_freq=cfg.filt.h_freq)

    logger.info("Loading ICA solution and setting bad components")
    ica = read_ica(cfg.input.ica)
    ica.exclude = read_ica_bads(cfg.output.bad_ics) if Path(cfg.output.bad_ics).exists() else []
    logger.info(f"ICs premarked as bad: {ica.exclude}")

    ica.plot_sources(raw, block=True)

    logger.info("Saving bad ics")
    write_ica_bads(cfg.output.bad_ics, ica)


if __name__ == "__main__":
    main()
