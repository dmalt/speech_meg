import numpy as np  # type: ignore
from joblib import Parallel, delayed  # type: ignore
from mne.io import read_raw_fif  # type: ignore
from sklearn.feature_selection import mutual_info_regression  # type: ignore
from speech import config as cfg  # type: ignore
from tqdm.contrib.itertools import product  # type: ignore

mfccs_ica = np.load(cfg.mfccs_ica_path)
shifts = np.linspace(cfg.tmin, cfg.tmax, cfg.n_shifts)

for band in cfg.bands:
    print(f"Processing {band} band...")
    raw = read_raw_fif(cfg.meg_env[band], verbose="ERROR")
    raw_data = raw.get_data()[:, cfg.skip_samp :]
    raw_data -= raw_data.mean(axis=1, keepdims=True)
    raw_data /= raw_data.std(axis=1, keepdims=True)
    n_sen, n_samp = raw_data.shape
    win_len_samp = int(cfg.win_len_sec * raw.info["sfreq"])
    winds = np.arange(0, n_samp, win_len_samp)
    mi = np.zeros((n_sen, cfg.n_shifts, len(winds)))
    shifts = (shifts * raw.info["sfreq"]).astype(int)

    def process_iter(m, i_shift, i_sen, i_win):
        mfcc_shift = np.roll(m, shifts[i_shift])[
            winds[i_win] : winds[i_win] + win_len_samp
        ]
        sen_data = raw_data[i_sen, winds[i_win] : winds[i_win] + win_len_samp][
            :, np.newaxis
        ]
        mi[i_sen, i_shift, i_win] += mutual_info_regression(
            sen_data, mfcc_shift
        )

    Parallel(n_jobs=8, require="sharedmem", prefer="threads")(
        delayed(process_iter)(m, i, j, w)
        for m, i, j, w in product(
            mfccs_ica, range(cfg.n_shifts), range(n_sen), range(len(winds))
        )
    )

    np.save(cfg.mfcc_mi_paths[band], mi)
