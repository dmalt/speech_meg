#!/usr/bin/env python
"""Apply ICA solution to raw data excluding bad components"""
import logging

import hydra
from mne.io import read_raw_fif  # type: ignore
from mne.preprocessing import read_ica  # type: ignore

from utils import prepare_script, read_ica_bads

logger = logging.getLogger(__file__)


@hydra.main(config_path="../configs/", config_name="061-apply_ica")
def main(cfg):
    prepare_script(logger, __file__)

    raw = read_raw_fif(cfg.input.raw, preload=True)
    ica = read_ica(cfg.input.ica)
    ica.exclude = read_ica_bads(cfg.input.bad_ics)
    logger.info(f"Excluding {ica.exclude}")
    ica.apply(raw)
    raw.save(cfg.output.raw, overwrite=True)


if __name__ == "__main__":
    main()
