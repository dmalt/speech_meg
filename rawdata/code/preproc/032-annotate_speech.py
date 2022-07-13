#!/usr/bin/env python

import logging
from dataclasses import dataclass
from pathlib import Path

import hydra
import matplotlib  # type: ignore
from hydra.core.config_store import ConfigStore
from mne import find_events  # type: ignore
from mne.annotations import read_annotations  # type: ignore
from mne.io.fiff.raw import read_raw_fif  # type: ignore
from utils import BaseConfig, prepare_script

matplotlib.use("TkAgg")
logger = logging.getLogger(__file__)


@dataclass
class Input:
    raw: str


@dataclass
class Output:
    annots: str


@dataclass
class Config(BaseConfig):
    input: Input
    output: Output


cs = ConfigStore.instance()
cs.store(name="schema", node=Config)


@hydra.main(config_path="../configs/", config_name="032-annotate_speech")
def main(cfg: Config):
    prepare_script(logger, script_name=__file__)

    annots = read_annotations(cfg.output.annots) if Path(cfg.output.annots).exists() else None
    logger.info("Preparing raw data")
    raw = read_raw_fif(cfg.input.raw)
    if annots is not None:
        raw.set_annotations(annots)
    ev = find_events(raw, min_duration=1, output="step")
    raw.pick_types(meg=False, misc=True)
    raw.plot(block=True, events=ev, decim=20)  # pyright: ignore

    logger.info(f"Annotations: {raw.annotations}")
    raw.annotations.save(cfg.output.annots, overwrite=True)


if __name__ == "__main__":
    main()
