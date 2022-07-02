from os import PathLike

from mne.io import read_raw_fif
from speech import config as cfg


def resample(src_raw_path: PathLike, dst_raw_path: PathLike, sfreq: float) -> None:
    raw = read_raw_fif(src_raw_path)
    raw.resample(sfreq=sfreq)
    raw.save(dst_raw_path)


if __name__ == "__main__":
    src_raw = cfg.ica_cleaned
    dst_raw = cfg.resampled_path
    resamp_freq = cfg.resamp_freq

    print(f"{src_raw=}, {dst_raw=}")
    resample(src_raw, dst_raw, resamp_freq)
