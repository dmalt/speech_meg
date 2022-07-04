#!/usr/bin/env python
"""Manually mark bad channels and possibly segments for maxfilter"""
import logging
from os import getcwd

import hydra
from utils import (
    annotate_raw_manually,
    prepare_annotated_raw,
    write_annotations,
    write_bad_channels,
)

logger = logging.getLogger(__name__)


@hydra.main(config_path="../configs/", config_name="01-annotate_premaxfilt")
def main(cfg):
    logger.info(f"Starting new session for {__file__}")
    logger.info(f"Current working directory is {getcwd()}")

    paths = cfg["01-annotate_premaxfilt"]
    raw = prepare_annotated_raw(paths.input.raw, paths.output.bad_ch, paths.output.annots)
    bads, annotations = annotate_raw_manually(raw)
    write_bad_channels(paths.output.bad_ch, bads)
    write_annotations(paths.output.annots, annotations)

    logger.info(f"Channels marked as bad: {bads}")
    logger.info(f"Annotations: {annotations}")


if __name__ == "__main__":
    main()
