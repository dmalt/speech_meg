#!/usr/bin/env python

import logging
import sys
from dataclasses import dataclass
from functools import reduce
from pathlib import Path
from typing import List

import hydra
import matplotlib  # type: ignore
import mne  # type: ignore
from hydra.core.config_store import ConfigStore
from mne import read_annotations
from utils import AnnotMode, BaseConfig, prepare_script

matplotlib.use("TkAgg")

logger = logging.getLogger(__file__)


@dataclass
class Input:
    raw: str
    annots_list: List[str]


@dataclass
class Output:
    annots: str


@dataclass
class Config(BaseConfig):
    input: Input
    output: Output
    mode: AnnotMode


cs = ConfigStore.instance()
cs.store(name="schema", node=Config)


@hydra.main(config_path="../configs/", config_name="101-merge_annotations")
def main(cfg: Config):
    prepare_script(logger, script_name=__file__)

    if not Path(cfg.output.annots).exists() or cfg.mode == AnnotMode.NEW:
        annots = reduce(lambda x, y: x + y, (read_annotations(a) for a in cfg.input.annots_list))
    else:
        annots = mne.read_annotations(cfg.output.annots)

    raw = mne.io.read_raw_fif(cfg.input.raw)
    raw.set_annotations(annots)
    raw.plot(block=True)

    logger.info(f"Annotations: {annots}")
    raw.annotations.save(cfg.output.annots, overwrite=True)


if __name__ == "__main__":
    main()
