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

logger = logging.getLogger(__file__)


@hydra.main(config_path="../configs/", config_name="02-apply_maxfilter")
def main(cfg):
    logger.info(f"Starting new session for {__name__}")
    logger.info(f"Current working directory is {os.getcwd()}")

    raw = prepare_annotated_raw(cfg.input.raw, cfg.input.bad_ch, cfg.input.annots)
    fix_mag_coil_types(raw.info)
    filter_chpi(raw, t_window=cfg.t_window)
    raw = maxwell_filter(
        raw, cross_talk=cfg.input.ct, calibration=cfg.input.cal, coord_frame="meg"
    )
    raw.save(cfg.output.maxfilt_raw, overwrite=True)


if __name__ == "__main__":
    main()
