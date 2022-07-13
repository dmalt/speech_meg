from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any

import mne  # type: ignore


class AnnotMode(Enum):
    EDIT = auto()
    NEW = auto()


@dataclass
class BaseConfig:
    paths: Any
    subj_id: str
    task: str
    deriv_paths: Any


def read_bad_channels(bads_path: str) -> list[str]:
    with open(bads_path, "r") as f:
        bads = f.readline().split("\t")
    if bads == [""]:
        bads = []
    return bads


def write_bad_channels(savepath: str, bads: list[str]) -> None:
    with open(savepath, "w") as f:
        f.write("\t".join(bads))


def read_ica_bads(ica_bads_path: str) -> list[int]:
    return [int(c) for c in read_bad_channels(ica_bads_path)]


def write_ica_bads(ica_bads_path: str, ica: mne.preprocessing.ICA) -> None:
    write_bad_channels(ica_bads_path, [str(ic) for ic in ica.exclude])


def update_annotations(raw: mne.io.Raw, annotations: mne.Annotations, overwrite=False) -> None:
    """
    Set new annotations first and then add existing

    This way we avoid conflict between annotations orig_time for `raw` and
    `annotations` when added annotations are not coming from the same raw object,
    i.e. generated programmatically

    See also
    --------
    mne.Annotations: orig_time behaviour in more detail

    """
    prev_annot = raw.annotations if hasattr(raw, "annotations") else None
    raw.set_annotations(annotations)
    if prev_annot is not None and not overwrite:
        raw.set_annotations(raw.annotations + prev_annot)


def prepare_script(logger: logging.Logger, script_name: str) -> None:
    logger.info(f"Starting new session for {script_name}")
    logger.info(f"Current working directory is {os.getcwd()}")
