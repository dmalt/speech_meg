import h5py
import librosa as lb
from mne.io import read_raw_fif
from speech import config as cfg  # type: ignore

raw = read_raw_fif(cfg.ica_cleaned)
raw.pick_types(meg=True)
# raw.pick_channels(
#     [c for c in raw.ch_names if c.startswith("MEG") or c == cfg.audio_ch]
# )

data = raw.get_data(reject_by_annotation="omit").T
audio, sr = lb.load(cfg.audio_align_path)


with h5py.File(cfg.h5_path, "w") as hf:
    g = hf.create_group("RawData")
    g.create_dataset("Samples", data=data)
    g.create_dataset("Audio", data=audio)
