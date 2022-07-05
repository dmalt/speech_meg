#!/usr/bin/env python
"""Automatically mark muscle segments and edit them manually"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Tuple

import hydra
from mne import read_annotations  # type: ignore
from mne.io import read_raw_fif  # type: ignore
from mne.preprocessing import annotate_muscle_zscore  # type: ignore
from omegaconf import OmegaConf

from utils import prepare_script, update_annotations

logger = logging.getLogger(__file__)

Band = Tuple[float, float]


@hydra.main(config_path="../configs/", config_name="071-annotate_muscles")
def main(cfg):
    prepare_script(logger, script_name=__file__)

    raw = read_raw_fif(cfg.input.raw, preload=True)
    if not Path(cfg.output.annots).exists() or cfg.mode == "new":
        logger.info("Creating new muscle annotations")
        params = OmegaConf.to_container(cfg.annotate_muscle_zscore_params)
        muscle_annots, _ = annotate_muscle_zscore(raw, **params)  # pyright: ignore
        raw.set_annotations(muscle_annots)
        prev_annots = read_annotations(cfg.input.annots)
        update_annotations(raw, prev_annots)
        logger.info(f"Auto muscle annotations + preexisting ones: {prev_annots}")
    elif cfg.mode == "edit":
        logger.info(f"Editing existing annotations at {cfg.output.annots}")
        edited_muscle_annots = read_annotations(cfg.output.annots)
        logger.info(f"Loaded annotations: {edited_muscle_annots}")
        raw.set_annotations(edited_muscle_annots)
    else:
        logger.error(f"Bad mode type {cfg.mode=}")
        sys.exit(1)
    raw.plot(block=True)
    logger.info(f"Final annotations: {annotations}")
    raw.annotations.save(cfg.output.annots, overwrite=True)


if __name__ == "__main__":
    main()
