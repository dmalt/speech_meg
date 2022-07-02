"""Manually mark bad channels and segments for maxfilter"""
from mne import read_annotations  # type: ignore
from mne.io import read_raw_fif  # type: ignore
from speech import config as cfg  # type: ignore


def inspect_fif(fif_path, bads, annotations, is_emptyroom):
    """Manually mark bad channels and segments in gui signal viewer
    Filter chpi and line noise from data copy for inspection
    """
    raw_check = read_raw_fif(fif_path, preload=True)
    if bads is not None:
        raw_check.info["bads"] = bads
    if annotations is not None:
        raw_check.set_annotations(annotations)
    raw_check.plot(block=True, lowpass=100, highpass=0.5, n_channels=100)
    return raw_check.info["bads"], raw_check.annotations


def read_bads(bads_path):
    with open(bads_path, "r") as f:
        bads = f.readline().split("\t")
    if bads == [""]:
        bads = []
    return bads


def write_bads(bads_path, bads):
    with open(bads_path, "w") as f:
        f.write("\t".join(bads))


def annotate_fif(raw_path, bads_path, annot_path, is_emptyroom):
    bads = read_bads(bads_path) if bads_path.exists() else None
    annotations = read_annotations(annot_path) if annot_path.exists() else None

    bads, annotations = inspect_fif(raw_path, bads, annotations, is_emptyroom)

    write_bads(bads_path, bads)
    annotations.save(str(annot_path), overwrite=True)


if __name__ == "__main__":
    raw = cfg.maxfilt_path
    bads = cfg.postmaxfilt_bads_path
    annot = cfg.postmaxfilt_annotations_path

    print(f"{raw=}, {bads=}, {annot=}")
    annotate_fif(raw, bads, annot, is_emptyroom=False)
