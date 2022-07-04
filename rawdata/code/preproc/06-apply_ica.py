"""Apply ICA solution with marked bad components to raw data"""
from mne.io import read_raw_fif  # type: ignore
from mne.preprocessing import read_ica  # type: ignore
from speech import config as cfg  # type: ignore


def read_ica_bads(ica_bads_path):
    with open(ica_bads_path, "r") as f:
        line = f.readline()
        bads = [int(b) for b in line.split("\t")] if line else []
    return bads


def clean_fif(fif_path, ica_sol_path, ica_bads_path, cleaned_fif_path):
    raw = read_raw_fif(fif_path, preload=True)
    ica = read_ica(ica_sol_path)
    ica.exclude = read_ica_bads(ica_bads_path)
    print(f"Excluding {ica.exclude}")
    ica.apply(raw)
    raw.save(cleaned_fif_path, overwrite=True)


if __name__ == "__main__":

    # raw = cfg.cropped_path
    raw = cfg.maxfilt_path
    ica_sol = cfg.ica_sol_path
    ica_bads_path = cfg.ica_bads_path
    print(f"{ica_bads_path=}")
    ica_cleaned = cfg.ica_cleaned
    clean_fif(raw, ica_sol, ica_bads_path, ica_cleaned)
