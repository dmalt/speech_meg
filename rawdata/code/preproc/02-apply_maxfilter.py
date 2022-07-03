#!/usr/bin/env python
"""
Perform Maxwell filtering on annotated data with SSS

Apply annotations from the previous step on the way

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
from mne.preprocessing import maxwell_filter  # type: ignore
from utils import prepare_annotated_raw

logger = logging.getLogger(__name__)


@hydra.main(config_path="../configs/", config_name="02-apply_maxfilter")
def main(cfg):
    logger.info("Starting new session")
    logger.info(f"Current working directory is {os.getcwd()}")

    raw = prepare_annotated_raw(
        cfg.raw_path,
        cfg.annotate_for_maxfilt.bad_channels_path,
        cfg.annotate_for_maxfilt.annotations_path,
    )
    fix_mag_coil_types(raw.info)
    filter_chpi(raw, t_window=cfg.t_window)
    raw = maxwell_filter(raw, cross_talk=cfg.ct_path, calibration=cfg.cal_path, coord_frame="meg")
    raw.save(cfg.maxfiltered_path, overwrite=True)


if __name__ == "__main__":
    main()
