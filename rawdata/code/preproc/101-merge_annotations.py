#!/usr/bin/env python

import logging
import sys
from functools import reduce

import hydra
import matplotlib  # type: ignore
from mne import read_annotations  # type: ignore
from mne.io.fiff.raw import read_raw_fif  # type: ignore
from utils import prepare_script, write_annotations

matplotlib.use("TkAgg")

logger = logging.getLogger(__file__)


@hydra.main(config_path="../configs/", config_name="101-merge_annotations")
def main(cfg):
    prepare_script(logger, script_name=__file__)

    if cfg.mode == "new":
        annots = reduce(lambda x, y: x + y, (read_annotations(a) for a in cfg.input.annots_list))
    elif cfg.mode == "edit":
        annots = read_annotations(cfg.output.annots)
    else:
        logger.error(f"Bad mode type {cfg.mode=}. Should be `new` or `edit`")
        sys.exit(1)

    raw = read_raw_fif(cfg.input.raw)
    raw.set_annotations(annots)
    raw.plot(block=True)

    logger.info(f"Annotations: {annots}")
    write_annotations(cfg.output.annots, annots)


if __name__ == "__main__":
    main()
