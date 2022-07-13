#!/usr/bin/env python
"""Manually mark bad segments after maxfilter for ICA"""
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import hydra
import matplotlib  # type: ignore
from hydra.core.config_store import ConfigStore
from mne.annotations import read_annotations  # type: ignore
from mne.io.fiff.raw import read_raw_fif  # type: ignore
from utils import AnnotMode, BaseConfig, prepare_script

matplotlib.use("TkAgg")
logger = logging.getLogger(__file__)


@dataclass
class Input:
    raw: str
    annots: str


@dataclass
class Output:
    annots: str


@dataclass
class Config(BaseConfig):
    input: Input
    output: Output
    mode: AnnotMode = AnnotMode.EDIT
    lowpass: Optional[float] = 100
    highpass: Optional[float] = None
    n_channels: int = 50


cs = ConfigStore.instance()
cs.store(name="schema", node=Config)


@hydra.main(config_path="../configs/", config_name="031-annotate_postmaxfilt")
def main(cfg: Config):
    prepare_script(logger, script_name=__file__)

    raw = read_raw_fif(cfg.input.raw)
    if not Path(cfg.output.annots).exists() or cfg.mode == AnnotMode.NEW:
        annots = read_annotations(cfg.input.annots) if Path(cfg.input.annots).exists() else None
        if annots is None:
            logger.warning(f"Input annotations file is missing at {cfg.input.annots}")
        raw.set_annotations(annots)
    else:
        raw.set_annotations(read_annotations(cfg.output.annots))

    raw.plot(block=True, lowpass=cfg.lowpass, highpass=cfg.highpass, n_channels=cfg.n_channels)

    logger.info(f"Annotations: {raw.annotations}")
    raw.annotations.save(cfg.output.annots, overwrite=True)


if __name__ == "__main__":
    main()
