#!/usr/bin/env python
"""Automatically mark muscle segments and edit them manually"""
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List

import hydra
import matplotlib  # type: ignore
import mne  # type: ignore
import mne.preprocessing as mne_preproc  # type: ignore
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from utils import AnnotMode, BaseConfig, prepare_script

matplotlib.use("TkAgg")

logger = logging.getLogger(__file__)


@dataclass
class Input:
    raw: str


@dataclass
class Output:
    annots: str


@dataclass
class AnnotateMuscleParams:
    threshold: float
    filter_freq: List[int]
    min_length_good: float
    ch_type: str


@dataclass
class Config(BaseConfig):
    input: Input
    output: Output
    mode: AnnotMode
    annotate_muscle_params: AnnotateMuscleParams


cs = ConfigStore.instance()
cs.store(name="schema", node=Config)


@hydra.main(config_path="../configs/", config_name="071-annotate_muscles")
def main(cfg: Config) -> None:
    prepare_script(logger, script_name=__file__)

    raw = mne.io.read_raw_fif(cfg.input.raw, preload=True)
    if not Path(cfg.output.annots).exists() or cfg.mode == AnnotMode.NEW:
        logger.info("Creating new muscle annotations")
        params = OmegaConf.to_container(cfg.annotate_muscle_params)
        muscle_annots, _ = mne_preproc.annotate_muscle_zscore(raw, **params)  # pyright: ignore
        raw.set_annotations(muscle_annots)
        logger.info(f"Auto muscle annotations: {muscle_annots}")
    else:
        logger.info(f"Editing existing annotations at {cfg.output.annots}")
        edited_muscle_annots = mne.read_annotations(cfg.output.annots)
        logger.info(f"Loaded annotations: {edited_muscle_annots}")
        raw.set_annotations(edited_muscle_annots)
    raw.plot(block=True)
    logger.info(f"Final annotations: {raw.annotations}")
    raw.annotations.save(cfg.output.annots, overwrite=True)


if __name__ == "__main__":
    main()
