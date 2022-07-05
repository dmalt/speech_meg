#!/usr/bin/env python
import logging

import hydra
from mne.io import read_raw_fif  # type: ignore

from utils import prepare_script

logger = logging.getLogger(__file__)


@hydra.main(config_path="../configs/", config_name="091-resample")
def main(cfg):
    prepare_script(logger, __file__)

    logger.info(f"Reading data from {cfg.input.raw}")
    raw = read_raw_fif(cfg.input.raw)

    logger.info("Resampling data")
    raw.resample(sfreq=cfg.sfreq)

    logger.info(f"Saving data to {cfg.output.raw}")
    raw.save(cfg.output.raw)


if __name__ == "__main__":
    main()
