from __future__ import annotations

from os import PathLike, fspath
from pathlib import Path
from typing import Optional

from mne import Annotations, read_annotations  # type: ignore
from mne.io import Raw, read_raw_fif  # type: ignore


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


def prepare_annotated_raw(raw_path: PathLike, bads_path: PathLike, annots_path: PathLike) -> Raw:
    bads_path, annots_path, raw_path = Path(bads_path), Path(annots_path), Path(raw_path)
    raw = read_raw_fif(raw_path, preload=True)
    raw.info["bads"] = read_bads(bads_path)
    annotations = read_annotations(annots_path) if annots_path.exists() else None
    if annotations is not None:
        update_annotations(raw, annotations)
    return raw


def update_annotations(raw: Raw, annotations: Annotations, overwrite=False) -> None:
    # need to set new annotations first and then add existing. this way there's no conflict
    # between their orig_time; see docs for mne.Annotations for more detail on orig_time
    prev_annot = raw.annotations
    raw.set_annotations(annotations)
    if prev_annot is not None and not overwrite:
        raw.set_annotations(raw.annotations + prev_annot)
