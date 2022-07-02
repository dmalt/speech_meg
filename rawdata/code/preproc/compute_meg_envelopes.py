import numpy as np  # type: ignore
from mne.io import RawArray, read_raw_fif  # type: ignore
from scipy.signal import hilbert  # type: ignore
from speech import config as cfg  # type: ignore
from tqdm import tqdm  # type: ignore

RESAMPLE_INITIAL = 500
RESAMPLE_FINAL = 100


def amplitude_env(signal, frame_size=50):
    amps = []

    for i in range(len(signal)):
        amps.append(max(signal[i : i + frame_size]))
    return np.array(amps)


raw = read_raw_fif(cfg.ica_cleaned)
raw.pick_types(meg=True)
raw.resample(RESAMPLE_INITIAL)

for band_name, band in tqdm(cfg.bands.items(), desc="Computing envelopes"):
    raw_filt = raw.copy().filter(l_freq=band[0], h_freq=band[1])
    raw_data = raw_filt.get_data(reject_by_annotation="omit")
    del raw_filt

    raw_data = np.abs(hilbert(raw_data))
    raw_band = RawArray(raw_data, raw.info)
    raw_band.resample(RESAMPLE_FINAL)
    raw_band.save(cfg.meg_env[band_name])
    del raw_band
