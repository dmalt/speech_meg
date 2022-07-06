#!/usr/bin/env python

import logging
from pathlib import Path

import hydra
from mne import find_events  # type: ignore
from utils import prepare_annotated_raw, prepare_script, write_annotations

logger = logging.getLogger(__file__)


@hydra.main(config_path="../configs/", config_name="032-annotate_speech")
def main(cfg):
    prepare_script(logger, script_name=__file__)

    annots_path = cfg.output.annots if Path(cfg.output.annots).exists() else cfg.input.annots
    logger.info("Preparing raw data")
    raw = prepare_annotated_raw(cfg.input.raw, None, annots_path)
    ev = find_events(raw, min_duration=1, output="step")
    # raw.filter(h_freq=498 / 3)
    raw.pick_types(meg=False, misc=True)
    # raw.load_data()
    raw.plot(block=True, events=ev, decim=20)  # pyright: ignore

    logger.info(f"Annotations: {raw.annotations}")
    write_annotations(cfg.output.annots, raw.annotations)


if __name__ == "__main__":
    main()
