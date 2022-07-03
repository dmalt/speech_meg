from __future__ import annotations

from os import PathLike
from pathlib import Path


def read_bads(bads_path: PathLike) -> list[str]:
    if not Path(bads_path).exists():
        return []
    with open(bads_path, "r") as f:
        bads = f.readline().split("\t")
    if bads == [""]:
        bads = []
    return bads


def write_bads(bads_path, bads):
    with open(bads_path, "w") as f:
        f.write("\t".join(bads))


def inspect_raw(raw, lowpass=100, highpass=None, n_channels=50):
    """
    Manually mark bad channels and segments in gui signal viewer
    Filter chpi and line noise from data copy for inspection
    """
    raw.plot(block=True, lowpass=lowpass, highpass=highpass, n_channels=n_channels)
    return raw.info["bads"], raw.annotations
