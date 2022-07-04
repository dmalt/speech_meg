# from __future__ import annotations

from pathlib import Path
from typing import Tuple

# import matplotlib.pyplot as plt  # type: ignore
from mne import Annotations  # type: ignore
from mne import read_annotations  # type: ignore
from mne.io import read_raw_fif  # type: ignore
from mne.preprocessing import annotate_muscle_zscore  # type: ignore
from speech import config as cfg  # type: ignore

Band = Tuple[float, float]


def annot_muscle(
    fif_path: Path,
    prev_annot: Annotations,
    z_thresh: float = 5,
    filt_freq: Band = (110, 200),
    debug: bool = True,
) -> Annotations:
    """Automatically annotate muscle segments and set them inplace"""
    raw = read_raw_fif(fif_path, preload=True)
    params = dict(ch_type="mag", threshold=z_thresh, min_length_good=1, filter_freq=filt_freq)
    annots, _ = annotate_muscle_zscore(raw, **params)
    # need to set muscle annotations first and then add existing. this way there's no conflict
    # between their orig_time; see docs for mne.Annotations for more detail on orig_time
    raw.set_annotations(annots)
    if prev_annot is not None:
        raw.set_annotations(raw.annotations + prev_annot)
    if debug:
        raw.plot(block=True, highpass=5, lowpass=300, n_channels=50)
    return raw.annotations


def annotate_fif(raw_path: Path, annot_path: Path, prev_annot: Path) -> None:
    prev_annot = read_annotations(prev_annot)
    annotations = annot_muscle(raw_path, prev_annot)
    annotations.save(str(annot_path), overwrite=True)


if __name__ == "__main__":
    raw = cfg.ica_cleaned
    annot = cfg.final_annotations
    prev_annot = cfg.postmaxfilt_annotations_path

    print(f"{raw=}, {annot=}")
    annotate_fif(raw, annot, prev_annot)
