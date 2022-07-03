#!/usr/bin/env python
"""Manually mark bad channels and segments for maxfilter"""
from __future__ import annotations

import logging
from os import PathLike, getcwd
from pathlib import Path

import hydra
from mne import read_annotations  # type: ignore
from mne.io import Raw, read_raw_fif  # type: ignore
from utils import inspect_raw, read_bads, write_bads

logger = logging.getLogger(__name__)


def prepare_raw_check(raw_path: PathLike, bads_path: PathLike, annots_path: PathLike) -> Raw:
    bads_path, annots_path = Path(bads_path), Path(annots_path)
    raw_check = read_raw_fif(raw_path, preload=True)
    raw_check.info["bads"] = read_bads(bads_path)
    annotations = read_annotations(annots_path) if annots_path.exists() else None
    if annotations is not None:
        raw_check.set_annotations(annotations)
    return raw_check


def annotate_fif(raw_path: PathLike, bads_path: PathLike, annots_path: PathLike) -> None:
    raw = prepare_raw_check(raw_path, bads_path, annots_path)
    bads, annotations = inspect_raw(raw)
    logger.info(f"Channels marked as bad: {bads}")
    write_bads(bads_path, bads)
    logger.info(f"Annotations: {annotations}")
    annotations.save(str(annots_path), overwrite=True)


@hydra.main(config_path="../configs/", config_name="01-mark_bads_maxfilter")
def main(cfg):
    logger.info("Starting new session")
    logger.info(f"Current working directory is {getcwd()}")

    annotate_fif(cfg.raw_path, cfg.bad_channels_path, cfg.annotations_path)


if __name__ == "__main__":
    main()
