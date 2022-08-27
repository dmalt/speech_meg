#!/usr/bin/env python
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import hydra
import matplotlib.pyplot as plt  # type: ignore
import mne  # type: ignore
import numpy as np
from hydra.core.config_store import ConfigStore
from matplotlib.figure import Figure  # type: ignore
from mne.preprocessing import ICA, read_ica  # type: ignore
from sklearn.feature_selection import mutual_info_regression  # type: ignore
from tqdm import tqdm, trange  # type: ignore
from utils import BaseConfig, prepare_script

logger = logging.getLogger(__file__)


@dataclass
class Input:
    raw: str
    ica: str
    annots: str


@dataclass
class Output:
    report: str


@dataclass
class IcaMuscleBandFilt:
    l_freq: Optional[float]
    h_freq: Optional[float]


@dataclass
class Config(BaseConfig):
    input: Input
    output: Output
    audio_ch: str
    ica_muscle_band_filt: IcaMuscleBandFilt
    mi_thresh: float
    dsamp_sfreq: float


cs = ConfigStore.instance()
cs.store(name="schema", node=Config)


def amplitude_env(signal: np.ndarray, frame_size: int = 50) -> np.ndarray:
    return np.array([max(signal[i : i + frame_size]) for i in range(len(signal))])


def smooth(signal: np.ndarray, window: int = 50, pad_width: int = 500) -> np.ndarray:
    sig_pad = np.pad(np.squeeze(signal), pad_width, mode="reflect")
    smooth = np.convolve(np.squeeze(sig_pad), np.hanning(window), mode="same")
    return smooth[pad_width:-pad_width]


def compute_crosscorr(
    c1: np.ndarray, c2: np.ndarray, sr: float, shift_nsamp: int = 500
) -> Tuple[np.ndarray, np.ndarray]:
    c1 = c1.copy()
    c2 = c2[shift_nsamp:-shift_nsamp].copy()
    c1 -= c1.mean()
    c2 -= c2.mean()
    cc1 = np.correlate(c1, c1)
    cc2 = np.correlate(c2, c2)
    times = np.arange(-shift_nsamp, shift_nsamp + 1) / sr
    return times, np.correlate(c2, c1) / np.sqrt(cc1 * cc2)


def retreive_audio_envelope(raw: mne.io.Raw, audio_ch: str, dsamp_sfreq: float) -> np.ndarray:
    audio_array = raw.get_data(picks=audio_ch, reject_by_annotation="omit")
    audio_env = smooth(amplitude_env(np.squeeze(audio_array)))[np.newaxis, :]
    audio_info = raw.info.copy().pick_channels([audio_ch])
    audio_env_smooth_raw = mne.io.RawArray(audio_env, audio_info).resample(dsamp_sfreq)
    return np.squeeze(audio_env_smooth_raw.get_data())


def retreive_ics_envelope(
    ica: ICA, raw: mne.io.Raw, dsamp_sfreq: float, ica_muscle_band_filt: IcaMuscleBandFilt
) -> np.ndarray:
    ics = ica.get_sources(raw)
    assert isinstance(ics, mne.io.Raw)
    ics._first_samps = raw._first_samps
    ics._last_samps = raw._last_samps
    del raw

    ics.resample(dsamp_sfreq)
    ics.filter(l_freq=ica_muscle_band_filt.l_freq, h_freq=ica_muscle_band_filt.h_freq)
    ics = ics.get_data(reject_by_annotation="omit")
    ics -= ics.mean(axis=1, keepdims=True)

    ica_env_smooth = np.empty_like(ics)
    for i in tqdm(range(len(ics)), desc="Computing smoothed envelopes"):
        ica_env_smooth[i, :] = smooth(amplitude_env(ics[i, :]))
    return ica_env_smooth


def gen_mi_scores_figure(ica: ICA, mi: np.ndarray, bad_mi_inds: np.ndarray, n_comp: int) -> Figure:
    fig_mi = ica.plot_scores(mi, bad_mi_inds, labels="muscle-audio MI", show=False)
    fig_mi.set_figwidth(20)
    fig_mi.set_figwidth(10)
    fig_mi.axes[0].grid(axis="y")
    fig_mi.axes[0].xaxis.set_ticks(list(range(0, n_comp, 2)))
    fig_mi.show()
    return fig_mi


def gen_crosscorrelation_fig(times: np.ndarray, corr: np.ndarray) -> Figure:
    corr_fig, ax = plt.subplots()
    ax.plot(times, corr)
    ax.set_xlabel("Time, seconds")
    ax.grid()
    ax.set_ylim(-0.3, 0.3)
    return corr_fig


def plot_envelopes(ica_env: np.ndarray, audio_env: np.ndarray, dsamp_sfreq: float) -> None:
    import matplotlib
    from mne import create_info

    matplotlib.use("TkAgg")
    ch_names = [f"IC {i}" for i in range(len(ica_env))] + ["Audio"]
    info = create_info(sfreq=dsamp_sfreq, ch_names=ch_names)
    data = np.concatenate([ica_env, audio_env[np.newaxis, :]], axis=0)
    raw = mne.io.RawArray(data, info)
    raw.plot(block=True)


@hydra.main(config_path="../configs/", config_name="042-gen_ic_audio_mi_report")
def main(cfg: Config) -> None:
    prepare_script(logger, script_name=__file__)

    raw = mne.io.read_raw_fif(cfg.input.raw, preload=True)
    if Path(cfg.input.annots).exists():
        raw.set_annotations(mne.read_annotations(cfg.input.annots))
    else:
        logger.warning(f"Annotation file is missing at {cfg.input.annots}")

    report = mne.Report()
    report.add_raw(raw, title="Raw")

    audio_env = retreive_audio_envelope(raw, cfg.audio_ch, cfg.dsamp_sfreq)
    ica = read_ica(cfg.input.ica)
    ica_env = retreive_ics_envelope(ica, raw, cfg.dsamp_sfreq, cfg.ica_muscle_band_filt)
    logger.info("Computing mutual info.")
    mi = mutual_info_regression(ica_env.T, np.squeeze(audio_env))
    logger.info("Done")

    fig_mi = gen_mi_scores_figure(ica, mi, np.nonzero(mi > cfg.mi_thresh)[0], len(ica_env))
    report.add_figure(fig_mi, title="Muscle - audio MI")

    for i_comp in trange(len(ica_env), desc="Computing cross-correlation"):
        times, corr = compute_crosscorr(ica_env[i_comp, :], audio_env, cfg.dsamp_sfreq)
        corr_fig = gen_crosscorrelation_fig(times, corr)
        topo_fig = ica.plot_components(picks=i_comp, show=False)
        report.add_figure(topo_fig, title=f"ICA {i_comp} topo")
        report.add_figure(corr_fig, title=f"ICA {i_comp} - audio cross-correlation")

    report.save(cfg.output.report, overwrite=True)
    plot_envelopes(ica_env, audio_env, cfg.dsamp_sfreq)


if __name__ == "__main__":
    main()
