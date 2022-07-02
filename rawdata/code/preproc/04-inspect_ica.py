"""Manually mark bad ICA components and apply solution"""
import mne  # type: ignore
from mne.io import RawArray, read_raw_fif  # type: ignore
from mne.preprocessing import read_ica  # type: ignore
from speech import config as cfg  # type: ignore


def read_ica_bads(ica_bads_path):
    with open(ica_bads_path, "r") as f:
        line = f.readline()
        bads = [int(b) for b in line.split("\t")] if line else []
    return bads


def write_ica_bads(ica_bads_path, ica):
    with open(ica_bads_path, "w") as f:
        f.write("\t".join([str(ic) for ic in ica.exclude]))


def get_raw_with_scaled_audio(raw):
    # raw.set_channel_types({"MISC008": "ecg"})
    data = raw.get_data()
    info = raw.info
    # eog, aud = mne.pick_channels(raw.ch_names, include=["MISC008", "EOG061"])
    aud = mne.pick_channels(raw.ch_names, include=["MISC008"])
    data[aud, :] /= (data[aud, :].std() * 100)
    # data[aud, :] *= data[0, :].std()
    return RawArray(data, info)


def inspect_ica(filt_path, ica_sol_path, ica_bads_path):
    raw = read_raw_fif(filt_path, preload=True)
    raw.filter(l_freq=1, h_freq=None)
    # raw = get_raw_with_scaled_audio(raw)
    ica = read_ica(ica_sol_path)
    if ica_bads_path.exists():
        ica.exclude = read_ica_bads(ica_bads_path)
    ica.plot_sources(raw, block=True)
    write_ica_bads(ica_bads_path, ica)


if __name__ == "__main__":
    # raw = cfg.cropped_path
    raw = cfg.maxfilt_path
    ica_sol = cfg.ica_sol_path
    # output
    ica_bads_path = cfg.ica_bads_path

    inspect_ica(raw, ica_sol, ica_bads_path)
