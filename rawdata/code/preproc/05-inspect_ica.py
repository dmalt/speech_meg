#!/usr/bin/env python
"""Manually mark bad ICA components"""
import logging

import hydra
from mne.io import read_raw_fif  # type: ignore
from mne.preprocessing import read_ica  # type: ignore
from utils import prepare_script, read_ica_bads, write_ica_bads

logger = logging.getLogger(__file__)


@hydra.main(config_path="../configs/", config_name="05-inspect_ica")
def main(cfg):
    prepare_script(logger, script_name=__file__)

    raw = read_raw_fif(cfg.input.raw, preload=True)
    raw.filter(l_freq=cfg.filt.l_freq, h_freq=cfg.filt.h_freq)

    ica = read_ica(cfg.input.ica)
    ica.exclude = read_ica_bads(cfg.output.bad_ics)
    logger.info(f"ICs marked as bad: {ica.exclude}")
    ica.plot_sources(raw, block=True)
    write_ica_bads(cfg.output.bad_ics, ica)


if __name__ == "__main__":
    main()
