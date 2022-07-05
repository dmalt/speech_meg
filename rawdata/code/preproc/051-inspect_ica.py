#!/usr/bin/env python
"""Manually mark bad ICA components"""
import logging

import hydra
from mne.preprocessing import read_ica  # type: ignore
from utils import prepare_annotated_raw, prepare_script, read_ica_bads, write_ica_bads

logger = logging.getLogger(__file__)


@hydra.main(config_path="../configs/", config_name="051-inspect_ica")
def main(cfg):
    prepare_script(logger, script_name=__file__)

    logger.info("Loading and filtering raw data")
    raw = prepare_annotated_raw(cfg.input.raw, bads_path=None, annots_path=cfg.input.annots)
    raw.filter(l_freq=cfg.filt.l_freq, h_freq=cfg.filt.h_freq)

    logger.info("Loading ICA solution and setting bad components")
    ica = read_ica(cfg.input.ica)
    ica.exclude = read_ica_bads(cfg.output.bad_ics)
    logger.info(f"ICs marked as bad: {ica.exclude}")

    ica.plot_sources(raw, block=True)
    write_ica_bads(cfg.output.bad_ics, ica)


if __name__ == "__main__":
    main()
