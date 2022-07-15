from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
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
    # field is used to avoid the problem with inheritance and
    # "non-default argument following the default one"
    bids_root: str = field(default=str(Path(__file__).parent.parent.parent), init=False)


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


def is_repo_clean() -> bool:
    return not os.popen("git status --porcelain").read()


def get_latest_commit_hash() -> str:
    return os.popen("git rev-parse HEAD").read().strip()


def should_proceed(logger: logging.Logger) -> bool:
    logger.warning("Git repository is not clean. Continue? (y/n)")
    while (ans := input("-> ")).lower() not in ("y", "n"):
        print("Please input 'y' or 'n'")
    logger.info(f"Answer: {ans}")
    return ans == "y"


def dump_commit_hash(logger: logging.Logger) -> None:
    if is_repo_clean():
        logger.info("Git repository is clean. Dumping commit hash.")
        logger.info(f"Current commit hash: {get_latest_commit_hash()}")
    elif should_proceed(logger):
        logger.info("Git repository is dirty. Dumping latest commit hash.")
        logger.info(f"Latest commit hash: {get_latest_commit_hash()}")
    else:
        sys.exit(0)


def prepare_script(logger: logging.Logger, script_name: str) -> None:
    hostname = os.uname().nodename
    username = os.getlogin()
    logger.info(f"Starting new session for {script_name}")
    logger.info(f"{hostname=}, {username=}")
    logger.debug(f"Current working directory is {os.getcwd()}")
    env = os.popen("conda env export").read()
    logger.info(f"Current environment dump:\n{env}")

    dump_commit_hash(logger)
