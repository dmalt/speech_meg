"""
Align low sampling frequency audio channel from MEG with full sfreq wav, crop
segments annotated as bad from full sfreq audio. Also, downsample audio
(when loading) to the standard 22050Hz

"""

import librosa as lb  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
import soundfile as sf  # type: ignore
from librosa import display
# from mne import create_info  # type: ignore
from mne import Report  # type: ignore
# from mne.io import RawArray, read_raw_fif  # type: ignore
from mne.io import read_raw_fif  # type: ignore
from scipy import signal  # type: ignore
from speech import config as cfg  # type: ignore

audio_ch = cfg.audio_ch
raw_path = cfg.ica_cleaned

raw = read_raw_fif(raw_path, preload=True)
audio_meg = np.squeeze(raw.get_data(picks=audio_ch, reject_by_annotation=None))

audio_lowres, sr_lowres = lb.load(cfg.audio_path, sr=raw.info["sfreq"])
audio_highres, sr_highres = lb.load(cfg.audio_path, sr=None)
corr = signal.correlate(audio_meg, audio_lowres, mode="full")


shift_lowres = corr.argmax() - audio_lowres.shape[0] + 110
# shift_lowres = corr.argmax() - audio_lowres.shape[0] + 1
shift_highres = int(shift_lowres / sr_lowres * sr_highres)
duration = int(len(audio_meg) / sr_lowres * sr_highres)
if shift_lowres >= 0:
    # pad full audio signal
    audio_highres = np.pad(audio_highres, pad_width=((shift_highres, 0),))
    audio_highres = audio_highres[:duration]
else:
    # TODO: negative shifts require audio_lowres cropping
    audio_highres = audio_highres[-shift_highres:]
    audio_highres = audio_highres[:duration]

# info_audio_hr = create_info(["audio"], sfreq=sr_highres)
# audio_hr_raw = RawArray(audio_highres[np.newaxis, :], info_audio_hr)
# audio_hr_raw.set_meas_date(raw.info["meas_date"])

# # hack to set annotations when orig times of annotations differ;
# # see mne.Annotations.orig_time for more details
# audio_hr_raw._first_samps = (raw._first_samps / sr_lowres * sr_highres).astype(int)
# audio_hr_raw._last_samps = (raw._last_samps / sr_lowres * sr_highres).astype(int)
# audio_hr_raw.set_annotations(raw.annotations)

# audio_hr_no_bads = np.squeeze(audio_hr_raw.get_data(picks="audio", reject_by_annotation="omit"))
# sf.write(cfg.audio_align_path, audio_hr_no_bads, sr_highres)
sf.write(cfg.audio_align_path, audio_highres, sr_highres)

report = Report()

plot_window = 90
start_sec = 70
display.waveshow(
    audio_meg[int(start_sec * sr_lowres) : int(plot_window * sr_lowres)], sr=sr_lowres, x_axis="ms"
)
display.waveshow(
    audio_highres[int(start_sec * sr_highres) : int(plot_window * sr_highres)] * 3,
    sr=sr_highres,
    x_axis="ms",
)

report.add_figure(plt.gcf(), title="Meg and wav audio alignment")
report.save(cfg.audio_align_report_path, overwrite=True)

