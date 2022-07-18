from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import hydra
import librosa as lb  # type: ignore
import mne  # type: ignore
import numpy as np
import numpy.typing as npt
from ndp.signal import Signal, Signal1D
from ndp.signal.annotations import Annotation, Annotations

BIDS_ROOT = Path(__file__).parent / "rawdata"


@dataclass
class Info:
    """Meg speech dataset info"""
    mne_info: mne.Info


def read_subject(subject: str) -> tuple[Signal[npt._32Bit], Signal1D[npt._32Bit], Info]:
    with hydra.initialize(config_path="rawdata/code/configs"):
        overrides = [f"bids_root={str(BIDS_ROOT)}", f"+subject={subject}"]
        paths = hydra.compose(config_name="paths", overrides=overrides)
    raw_path = paths["091-resample"].raw
    audio_path = paths["081-align_audio"].aligned_audio
    annotations_path = paths["101-merge_annotations"].annots
    return _read_dataset(raw_path, audio_path, annotations_path)


def _read_dataset(
    raw_path: str, audio_path: str, annotations_path: str
) -> tuple[Signal[npt._32Bit], Signal1D[npt._32Bit], Info]:
    X, info = _read_raw(raw_path, annotations_path)
    Y = _read_wav(audio_path)
    Y.annotations = X.annotations
    assert abs(X.duration - Y.duration) < 0.01, "inconsistent durations for audio and MEG"
    return X, Y, info


def _read_wav(path: str, sr: int | None = None) -> Signal1D[npt._32Bit]:
    data, sr_final = lb.load(path, sr=sr)  # pyright: ignore
    return Signal1D(data[:, np.newaxis], sr_final, [])


def _read_raw(raw_path: str, annot_path: str | None) -> tuple[Signal[npt._32Bit], Info]:
    raw = mne.io.read_raw_fif(raw_path, verbose="ERROR", preload=True)
    if annot_path is not None:
        annots = mne.read_annotations(annot_path)
        raw.set_annotations(annots)
    X_data = raw.get_data(picks="meg").astype("float32").T
    return Signal(X_data, raw.info["sfreq"], _annotations_from_raw(raw)), Info(raw.info)


def _annotations_from_raw(raw: mne.io.BaseRaw) -> Annotations:
    if not hasattr(raw, "annotations"):
        return []
    onsets: list[float] = list(raw.annotations.onset)
    durations: list[float] = list(raw.annotations.duration)
    types: list[str] = list(raw.annotations.description)
    onsets = [o - raw.first_samp / raw.info["sfreq"] for o in raw.annotations.onset]
    return [Annotation(o, d, t) for o, d, t, in zip(onsets, durations, types)]
