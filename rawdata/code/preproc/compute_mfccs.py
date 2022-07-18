import librosa as lb  # type: ignore
import librosa.feature as lbf  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
import scipy.interpolate as sci  # type: ignore
from mne.io import read_raw_fif  # type: ignore
from scipy.linalg import svdvals  # type: ignore
from sklearn.decomposition import FastICA  # type: ignore
from sklearn.feature_selection import mutual_info_regression  # type: ignore
from speech import config as cfg  # type: ignore
from tqdm import tqdm  # type: ignore


def interpolated_mfcc(audio, raw_sr, raw_nsamp, audio_sr, hop, n_mfcc=7):
    """
    Interpolate audio mfccs to match number of samples in raw

    librosa.feature.mfcc() computes mfcc timeseries with a skip between
    adjacent frames equal to hop_length argument (512 samples by default),
    which leads to getting a downsampled output. Decreasing hop_length is not
    an option --- array won't fit into memory. Therefore, we interpolate
    missing mfcc samples.

    """
    x = np.arange(0, len(audio_wav), hop)
    y = lbf.mfcc(y=audio_wav, sr=sr, n_mfcc=n_mfcc, hop_length=hop)
    itp = sci.interp1d(x, y, bounds_error=False, fill_value="extrapolate")

    raw_times = np.arange(0, raw_nsamp / raw_sr, 1 / raw_sr)
    raw_samp = (raw_times * audio_sr).astype(int)
    return itp(raw_samp)


def compute_mi_matrix(signals):
    n_comp = signals.shape[0]
    res = np.empty((n_comp, n_comp))
    with tqdm(total=int(n_comp * (n_comp + 1) / 2), desc="MI matrix") as pbar:
        for i in range(n_comp):
            for j in range(i, n_comp):
                mi = mutual_info_regression(
                    signals[i : i + 1, :].T, signals[j, :]
                )
                res[i, j] = mi
                res[j, i] = mi
                pbar.update(1)
    return res


def get_ica_n_components(signals, thresh=0.99):
    s_2 = svdvals(signals) ** 2
    pve = np.cumsum(s_2) / np.sum(s_2)
    for i, p in enumerate(pve):
        if p > thresh:
            break
    return i + 1


audio_wav, sr = lb.load(cfg.audio_align_path)

band = "alpha"
raw = read_raw_fif(cfg.meg_env[band], verbose="ERROR")
raw_data = raw.get_data()
raw_sr, (n_sen, raw_nsamp) = raw.info["sfreq"], raw_data.shape


mfccs = interpolated_mfcc(
    audio_wav, raw_sr, raw_nsamp, sr, cfg.hop_length, cfg.n_mfcc
)


mfccs = mfccs[:, cfg.skip_samp :]
mfccs -= mfccs.mean(axis=1, keepdims=True)
mfccs /= mfccs.std(axis=1, keepdims=True)

mi_mat = compute_mi_matrix(mfccs)

raw_data = raw_data[:, cfg.skip_samp :]
n_comp = get_ica_n_components(mfccs)
transformer = FastICA(n_components=n_comp, max_iter=1000)
mfccs_ica = transformer.fit_transform(mfccs.T).T

mi_mat_ica = compute_mi_matrix(mfccs_ica)

n_shifts = 11
shifts = np.linspace(
    cfg.tmin * raw_sr, cfg.tmax * raw_sr, cfg.n_shifts
).astype(int)
plt.show()

a, b = 2000, 18000
a += cfg.skip_samp
b += cfg.skip_samp

pp = (mfccs[:, a:b] - mfccs[:, a:b].mean(axis=1, keepdims=True)) / mfccs[
    :, a:b
].std(axis=1, keepdims=True)
scale = 10
plt.plot(pp.T + np.arange(cfg.n_mfcc) * scale)

pp_ica = (
    mfccs_ica[:, a:b] - mfccs_ica[:, a:b].mean(axis=1, keepdims=True)
) / mfccs_ica[:, a:b].std(axis=1, keepdims=True)
plt.plot(pp_ica.T + np.arange(n_comp) * scale - (cfg.n_mfcc + n_comp) * 5)
plt.show()

np.save(cfg.mfccs_ica_path, mfccs_ica)
