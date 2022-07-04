#!/usr/bin/env python
"""Manually mark bad segments after maxfilter for ICA"""
import logging

import hydra
from utils import annotation_pipeline

logger = logging.getLogger(__file__)


@hydra.main(config_path="../configs/", config_name="03-annotate_postmaxfilt")
def main(cfg):
    annotation_pipeline(logger, cfg)


if __name__ == "__main__":
    main()
