#!/usr/bin/env python
"""Manually mark bad channels and possibly segments for maxfilter"""
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import hydra
import matplotlib  # type: ignore
from hydra.core.config_store import ConfigStore
from mne.annotations import read_annotations  # type: ignore
from mne.io.fiff.raw import read_raw_fif  # type: ignore
from utils import BaseConfig, prepare_script, read_bad_channels, write_bad_channels

matplotlib.use("TkAgg")
logger = logging.getLogger(__file__)


@dataclass
class Input:
    raw: str


@dataclass
class Output:
    bad_ch: str
    annots: str


@dataclass
class Config(BaseConfig):
    input: Input
    output: Output
    lowpass: Optional[float] = 100
    highpass: Optional[float] = None
    n_channels: int = 50


cs = ConfigStore.instance()
cs.store(name="schema", node=Config)


@hydra.main(config_path="../configs/", config_name="011-annotate_premaxfilt")
def main(cfg: Config):
    prepare_script(logger, script_name=__file__)
    raw = read_raw_fif(cfg.input.raw)
    if Path(cfg.output.bad_ch).exists():
        raw.info["bads"] = read_bad_channels(cfg.output.bad_ch)
    if Path(cfg.output.annots).exists():
        raw.set_annotations(read_annotations(Path(cfg.output.annots)))

    raw.plot(block=True, lowpass=cfg.lowpass, highpass=cfg.highpass, n_channels=cfg.n_channels)

    write_bad_channels(cfg.output.bad_ch, raw.info["bads"])
    raw.annotations.save(cfg.output.annots, overwrite=True)

    logger.info(f"Channels marked as bad: {raw.info['bads']}")
    logger.info(f"Annotations: {raw.annotations}")


if __name__ == "__main__":
    main()
