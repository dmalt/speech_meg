import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
from mne import Report  # type: ignore
from mne.io import RawArray, read_raw_fif  # type: ignore
from mne.preprocessing import read_ica  # type: ignore
from sklearn.feature_selection import mutual_info_regression  # type: ignore
from speech import config as cfg  # type: ignore
from tqdm import tqdm  # type: ignore

dsamp_sfreq = 250


def amplitude_env(signal, frame_size=50):
    amps = []

    for i in range(len(signal)):
        amps.append(max(signal[i : i + frame_size]))
    return np.array(amps)


def smooth(signal, window=50, pad_width=500):
    sig_pad = np.pad(np.squeeze(signal), pad_width, mode="reflect")
    smooth = np.convolve(np.squeeze(sig_pad), np.hanning(window), mode="same")
    return smooth[pad_width:-pad_width]


def compute_crosscorr(ica_env, audio_env, i_comp, shift_nsamp=500):
    c1 = ica_env[i_comp, :]
    c2 = audio_env[shift_nsamp:-shift_nsamp]
    c1 -= c1.mean()
    c2 -= c2.mean()
    cc1 = np.correlate(c1, c1)
    cc2 = np.correlate(c2, c2)
    times = np.arange(
        -shift_nsamp / dsamp_sfreq,
        (shift_nsamp + 1) / dsamp_sfreq,
        1 / dsamp_sfreq,
    )
    return np.correlate(c1, c2) / np.sqrt(cc1 * cc2), times


# raw_path = cfg.cropped_path
raw_path = cfg.maxfilt_path
ica_sol = cfg.ica_sol_path
ica_bads_path = cfg.ica_bads_path

raw = read_raw_fif(raw_path, preload=True)
rawinfo = raw.info.copy()

report = Report()
report.add_raw(raw, title="Raw")

audio_ch = "MISC008"
audio_array = raw.get_data(picks=audio_ch, reject_by_annotation="omit")


audio_env = amplitude_env(np.squeeze(audio_array))
audio_env_smooth_raw = RawArray(
    smooth(audio_env)[np.newaxis, :], raw.info.copy().pick_channels([audio_ch])
).resample(dsamp_sfreq)
audio_env_smooth = np.squeeze(audio_env_smooth_raw.get_data())

ica = read_ica(ica_sol)
src = ica.get_sources(raw)
src._first_samps = raw._first_samps
src._last_samps = raw._last_samps
del raw


src.resample(dsamp_sfreq)
src.filter(**cfg.ica_comp_filt)
ica_comp = src.get_data(reject_by_annotation="omit")
ica_comp -= ica_comp.mean(axis=1, keepdims=True)
n_comp, n_samp = ica_comp.shape

ica_env_smooth = np.empty_like(ica_comp)
for i in tqdm(range(n_comp)):
    ica_env_smooth[i, :] = smooth(amplitude_env(ica_comp[i, :]))


res = mutual_info_regression(ica_env_smooth.T, np.squeeze(audio_env_smooth))

fig_mi = ica.plot_scores(
    res,
    np.nonzero(res > cfg.mi_thresh)[0],
    labels="muscle-audio MI",
    show=False,
)
fig_mi.set_figwidth(20)
fig_mi.set_figwidth(10)
fig_mi.axes[0].grid(axis="y")
fig_mi.axes[0].xaxis.set_ticks(list(range(0, n_comp, 2)))
fig_mi.show()
report.add_figure(fig_mi, title="Muscle - audio MI")


for i_comp in range(len(ica_comp)):
    topo_fig = ica.plot_components(picks=i_comp, show=False)
    corr, times = compute_crosscorr(ica_env_smooth, audio_env_smooth, i_comp)
    corr_fig, ax = plt.subplots()
    ax.plot(times, corr)
    ax.set_xlabel("Time, seconds")
    ax.grid()
    ax.set_ylim(-0.3, 0.3)
    report.add_figure(topo_fig, title=f"ICA {i_comp} topo")
    report.add_figure(
        corr_fig, title=f"ICA {i_comp} - audio cross-correlation"
    )

report.save(cfg.ica_muscle_report)
