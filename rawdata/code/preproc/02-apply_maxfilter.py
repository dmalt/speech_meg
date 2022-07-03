#!/usr/bin/env python
"""
Perform Maxwell filtering on annotated data with SSS
Note
----
requires mne >= 0.20 for filtering line noise with filter_chpi
for emptyroom data
"""
import logging
import os

import hydra
from mne.channels import fix_mag_coil_types  # type: ignore
from mne.chpi import filter_chpi  # type: ignore
from mne.io import Raw, read_raw_fif  # type: ignore
from mne.preprocessing import maxwell_filter  # type: ignore
from speech import config as cfg  # type: ignore
from utils import read_bads

logger = logging.getLogger(__name__)


def prepare_raw(raw_path: os.PathLike, bads_path: os.PathLike) -> Raw:
    """Load raw, filter chpi and line noise, set bads"""
    raw = read_raw_fif(raw_path, preload=True)
    filter_chpi(raw, t_window=cfg.maxfilt_config["t_window"])
    fix_mag_coil_types(raw.info)
    raw.info["bads"] = read_bads(bads_path)
    return raw


@hydra.main(config_path="../configs/", config_name="02-apply_maxfilter")
def main(cfg):
    logger.info("Starting new session")
    logger.info(f"Current working directory is {os.getcwd()}")

    raw = prepare_raw(cfg.raw_path, cfg.annotate_for_maxfilt.bad_channels_path)
    raw = maxwell_filter(raw, cross_talk=cfg.ct_path, calibration=cfg.cal_path, coord_frame="meg")
    raw.save(cfg.maxfiltered_path, overwrite=True)


if __name__ == "__main__":
    main()
