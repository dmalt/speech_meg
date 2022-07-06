from __future__ import annotations

import logging
import os
from os import PathLike, fspath
from pathlib import Path
from typing import Optional

from mne import Annotations, read_annotations  # type: ignore
from mne.io import Raw, read_raw_fif  # type: ignore
from mne.preprocessing import ICA  # type: ignore


def read_bads(bads_path: Optional[PathLike]) -> list[str]:
    if bads_path is None or not Path(bads_path).exists():
        return []
    with open(bads_path, "r") as f:
        bads = f.readline().split("\t")
    if bads == [""]:
        bads = []
    return bads


def write_bad_channels(path: PathLike, bads: list[str]) -> None:
    with open(path, "w") as f:
        f.write("\t".join(bads))


def write_annotations(path: PathLike, annotations: Annotations) -> None:
    annotations.save(fspath(path), overwrite=True)


def annotate_raw_manually(raw, lowpass=100, highpass=None, n_channels=50):
    """
    Manually mark bad channels and segments in gui signal viewer
    Filter chpi and line noise from data copy for inspection
    """
    raw.plot(block=True, lowpass=lowpass, highpass=highpass, n_channels=n_channels)
    return raw.info["bads"], raw.annotations


def prepare_annotated_raw(
    raw_path: PathLike, bads_path: Optional[PathLike], annots_path: Optional[PathLike]
) -> Raw:
    """Read raw data, and set bad channels and annotations loaded from files"""
    raw_path = Path(raw_path)
    raw = read_raw_fif(raw_path, preload=True)
    raw.info["bads"] = read_bads(bads_path)

    if annots_path is not None and Path(annots_path).exists():
        annotations = read_annotations(Path(annots_path))
        update_annotations(raw, annotations, overwrite=True)
    return raw


def update_annotations(raw: Raw, annotations: Annotations, overwrite=False) -> None:
    # need to set new annotations first and then add existing. this way there's no conflict
    # between their orig_time; see docs for mne.Annotations for more detail on orig_time
    prev_annot = raw.annotations
    raw.set_annotations(annotations)
    if prev_annot is not None and not overwrite:
        raw.set_annotations(raw.annotations + prev_annot)


def annotation_pipeline(logger, cfg):
    logger.info(f"Starting new session for {__file__}")
    logger.info(f"Current working directory is {os.getcwd()}")

    raw = prepare_annotated_raw(cfg.input.raw, cfg.output.bad_ch, cfg.output.annots)
    bads, annotations = annotate_raw_manually(raw)
    write_bad_channels(cfg.output.bad_ch, bads)
    write_annotations(cfg.output.annots, annotations)

    logger.info(f"Channels marked as bad: {bads}")
    logger.info(f"Annotations: {annotations}")


def read_ica_bads(ica_bads_path: PathLike) -> list[int]:
    if not Path(ica_bads_path).exists():
        return []
    with open(ica_bads_path, "r") as f:
        line = f.readline()
        bads = [int(b) for b in line.split("\t")] if line else []
    return bads


def write_ica_bads(ica_bads_path: PathLike, ica: ICA) -> None:
    with open(ica_bads_path, "w") as f:
        f.write("\t".join([str(ic) for ic in ica.exclude]))


def prepare_script(logger: logging.Logger, script_name: str) -> None:
    logger.info(f"Starting new session for {script_name}")
    logger.info(f"Current working directory is {os.getcwd()}")
