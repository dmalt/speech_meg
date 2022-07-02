from mne.io import read_raw_fif  # type: ignore
from speech import config as cfg  # type: ignore

raw = read_raw_fif(cfg.annotated_path)
raw.crop(tmax=2497)
raw.plot(block=True)
raw.save(cfg.cropped_path, overwrite=True)
