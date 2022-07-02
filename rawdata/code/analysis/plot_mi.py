import matplotlib.pyplot as plt  # type: ignore
import mne  # type: ignore
import numpy as np
from mne import Report
from mne.io import read_raw_fif  # type: ignore
from mne.viz import plot_topomap  # type: ignore
from speech import config as cfg  # type: ignore

report = Report()

n_shifts = 11
shifts = np.linspace(-1, 1, n_shifts)

raw = read_raw_fif(cfg.meg_env['alpha'])
idx_grad = mne.pick_types(raw.info, meg="grad")
idx_mag = mne.pick_types(raw.info, meg="mag")
info_grad = raw.copy().pick_channels([raw.ch_names[c] for c in idx_grad]).info
info_mag = raw.copy().pick_channels([raw.ch_names[c] for c in idx_mag]).info

for band in cfg.bands:

    d = np.load(cfg.mfcc_mi_paths[band])
    if d.ndim == 3:
        d = d.mean(axis=2) / d.std(axis=2) * np.sqrt(d.shape[2])
        # d = d.mean(axis=2)

    figs = []
    captions = []
    for time_idx in range(n_shifts):
        fig, ax = plt.subplots(1, 2)
        ax[0].set_title("magnetometers")
        ax[1].set_title("gradiometers")
        d_mag = d[idx_mag, time_idx]
        # im, c = plot_topomap(d_mag, info_mag, show=False, vmax=0.025, axes=ax[0])
        im, c = plot_topomap(d_mag, info_mag, show=False, axes=ax[0])
        plt.colorbar(im, ax=ax[0])
        d_grad = d[idx_grad, time_idx]
        # im, c = plot_topomap(d_grad, info_grad, show=False, vmax=0.025, axes=ax[1])
        im, c = plot_topomap(d_grad, info_grad, show=False, axes=ax[1])
        plt.colorbar(im, ax=ax[1])
        fig.set_figwidth(20)
        figs.append(fig)
        shift_sec = round(shifts[time_idx], 2)
        captions.append(f"audio shift = {shift_sec} sec")

    report.add_figure(
        fig=figs, title=f"{band=} {cfg.bands[band]}", caption=captions
    )
    fig = plt.figure()
    fig.set_figwidth(20)
    plt.plot(shifts, d.max(axis=0))
    plt.grid()
    plt.xlabel("Audio shift, sec")
    plt.ylabel("MI, max over sensors")
    plt.xticks(shifts)
    report.add_figure(fig, title='', caption="Max MEG env-audio mfccs MI")

report.save("meg_env_audio_mfcc_ica_mi.html", overwrite=True)
