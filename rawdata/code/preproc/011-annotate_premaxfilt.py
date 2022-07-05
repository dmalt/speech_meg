#!/usr/bin/env python
"""Manually mark bad channels and possibly segments for maxfilter"""
import logging

import hydra

from utils import annotation_pipeline

logger = logging.getLogger(__file__)


@hydra.main(config_path="../configs/", config_name="011-annotate_premaxfilt")
def main(cfg):
    annotation_pipeline(logger, cfg)


if __name__ == "__main__":
    main()
