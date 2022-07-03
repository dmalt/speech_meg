"""Manually mark bad channels and segments for maxfilter"""
from __future__ import annotations

from os import PathLike, getcwd
from pathlib import Path
from typing import Optional

import hydra
from mne import read_annotations  # type: ignore
from mne.io import Raw, read_raw_fif  # type: ignore
from omegaconf import OmegaConf

# from speech import config as cfg  # type: ignore


def inspect_fif(raw, lowpass=100, highpass=None, n_channels=50):
    """
    Manually mark bad channels and segments in gui signal viewer
    Filter chpi and line noise from data copy for inspection
    """
    raw.plot(block=True, lowpass=lowpass, highpass=highpass, n_channels=n_channels)
    return raw.info["bads"], raw.annotations


def read_bads(bads_path: PathLike) -> Optional[list[str]]:
    with open(bads_path, "r") as f:
        bads = f.readline().split("\t")
    if bads == [""]:
        bads = []
    return bads


def write_bads(bads_path, bads):
    with open(bads_path, "w") as f:
        f.write("\t".join(bads))


def assemble_raw_check(raw_path: PathLike, bads_path: PathLike, annots_path: PathLike) -> Raw:
    bads_path, annots_path = Path(bads_path), Path(annots_path)
    bads = read_bads(bads_path) if bads_path.exists() else None
    annotations = read_annotations(annots_path) if annots_path.exists() else None
    raw_check = read_raw_fif(raw_path, preload=True)
    if bads is not None:
        raw_check.info["bads"] = bads
    if annotations is not None:
        raw_check.set_annotations(annotations)
    return raw_check


def annotate_fif(raw_path, bads_path, annots_path):
    raw = assemble_raw_check(raw_path, bads_path, annots_path)
    bads, annotations = inspect_fif(raw)
    write_bads(bads_path, bads)
    annotations.save(str(annots_path), overwrite=True)


@hydra.main(config_path="../configs/", config_name="01-mark_bads_maxfilter")
def main(cfg):

    OmegaConf.resolve(cfg)  # type: ignore
    print(OmegaConf.to_yaml(cfg))
    print(f"Current working directory is {getcwd()}")
    raw = cfg.raw_path
    bads = cfg.bad_channels
    annot = cfg.annotations

    print(f"{raw=}, {bads=}, {annot=}")
    annotate_fif(raw, bads, annot)


if __name__ == "__main__":
    main()
