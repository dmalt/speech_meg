#!/usr/bin/env python
"""
Align low sampling frequency audio channel from MEG with high-resolution wav
Also, downsample wav audio (when loading) to the target frequency

"""

import logging
from dataclasses import dataclass, field
from typing import List

import hydra
import librosa as lb  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import mne  # type: ignore
import numpy as np  # type: ignore
import scipy.signal  # type: ignore
import soundfile as sf  # type: ignore
from hydra.core.config_store import ConfigStore
from librosa import display
from utils import BaseConfig, prepare_script

logger = logging.getLogger(__file__)


@dataclass
class Input:
    raw: str
    audio_hr: str


@dataclass
class Output:
    aligned_audio: str
    report: str


@dataclass
class Config(BaseConfig):
    input: Input
    output: Output
    audio_ch: str
    audio_dsamp_freq: int
    correction_samp: int
    report_segments_sec: List


cs = ConfigStore.instance()
cs.store(name="schema", node=Config)


@dataclass
class AlignmentReport:
    audio_x: np.ndarray
    sr_x: int
    label_x: str
    audio_y: np.ndarray
    sr_y: int
    label_y: str
    report: mne.Report = field(default_factory=mne.Report)

    def __post_init__(self) -> None:
        self.audio_x = scale(self.audio_x)
        self.audio_y = scale(self.audio_y)

    def add_segment(self, start_sec: float, stop_sec: float) -> None:
        slice_lr = slice(int(start_sec * self.sr_x), int(stop_sec * self.sr_x))
        slice_hr = slice(int(start_sec * self.sr_y), int(stop_sec * self.sr_y))

        fig, ax = plt.subplots()
        params = dict(x_axis="ms", ax=ax, alpha=0.9)
        display.waveshow(self.audio_x[slice_lr], sr=self.sr_x, label=self.label_x, **params)
        display.waveshow(self.audio_y[slice_hr], sr=self.sr_y, label=self.label_y, **params)
        fig.set_size_inches(18, 10)
        ax.legend()
        self.report.add_figure(fig, title=f"Alignment at {start_sec}--{stop_sec} sec")

    def save(self, savepath: str) -> None:
        self.report.save(savepath, overwrite=True)


def scale(signal: np.ndarray) -> np.ndarray:
    signal -= signal.mean()
    signal /= signal.std()
    return signal


def compute_shift(audio_meg: np.ndarray, audio_lowres: np.ndarray, correction_samp: int) -> int:
    """
    Compute shift for audio_lowres to be aligned with audio_meg

    Identical sampling rates are assumed

    """
    corr = scipy.signal.correlate(audio_meg, audio_lowres, mode="full")
    return corr.argmax() - len(audio_lowres) + correction_samp


def align_audio(audio: np.ndarray, shift: int, target_duration: int) -> np.ndarray:
    """Shift and crop or pad audio to match the target_duration"""
    if shift >= 0:
        audio = np.pad(audio, pad_width=((shift, 0),))  # pyright: ignore
    else:
        audio = audio[-shift:]
    return audio[:target_duration]


def read_meg_audio(raw_path: str, audio_ch):
    raw = mne.io.read_raw_fif(raw_path, preload=True)
    audio_meg = np.squeeze(raw.get_data(picks=audio_ch, reject_by_annotation=None))
    sr_meg = raw.info["sfreq"]
    return audio_meg, sr_meg


def resample(nsamples: int, sr_from: float, sr_to: float) -> int:
    return int(nsamples / sr_from * sr_to)


@hydra.main(config_path="../configs/", config_name="081-align_audio")
def main(cfg: Config):
    prepare_script(logger, script_name=__file__)

    audio_meg, sr_meg = read_meg_audio(cfg.input.raw, cfg.audio_ch)

    logger.info("Loading and downsampling lowres wav audio")
    audio_lowres, sr_lowres = lb.load(cfg.input.audio_hr, sr=sr_meg)

    logger.info("Computing lowres shift")
    shift_lowres = compute_shift(audio_meg, audio_lowres, cfg.correction_samp)
    logger.info(f"{shift_lowres=}")

    logger.info("Loading and downsampling highres wav audio")
    audio_highres, sr_highres = lb.load(cfg.input.audio_hr, sr=cfg.audio_dsamp_freq)

    logger.info("Aligning highres audio and saving the result")
    shift_highres = resample(shift_lowres, sr_lowres, sr_highres)
    aligned_highres_duration = resample(len(audio_meg), sr_lowres, sr_highres)
    audio_highres_aligned = align_audio(audio_highres, shift_highres, aligned_highres_duration)
    sf.write(cfg.output.aligned_audio, audio_highres_aligned, sr_highres)

    logger.info("Preparing and saving report")
    report = AlignmentReport(audio_meg, sr_meg, "MEG", audio_highres_aligned, sr_highres, "wav HR")
    for seg in cfg.report_segments_sec:
        report.add_segment(*seg)
    report.save(cfg.output.report)
    logger.info("Finished")


if __name__ == "__main__":
    main()
