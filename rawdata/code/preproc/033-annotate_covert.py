#!/usr/bin/env python

import logging
from pathlib import Path

import hydra
import numpy as np
from mne import annotations_from_events, find_events  # type: ignore
from utils import prepare_annotated_raw, prepare_script, update_annotations, write_annotations

logger = logging.getLogger(__file__)


@hydra.main(config_path="../configs/", config_name="033-annotate_covert")
def main(cfg):
    prepare_script(logger, script_name=__file__)

    # annots_path = cfg.output.annots if Path(cfg.output.annots).exists() else cfg.input.annots
    annots_path = cfg.input.annots
    logger.info("Preparing raw data")
    logger.info(f"{annots_path=}, {Path(annots_path).exists()=}")
    raw = prepare_annotated_raw(cfg.input.raw, None, annots_path)
    ev = find_events(raw, min_duration=1, output="step")
    ev_annot = []
    for e in filter(lambda x: x[2] > 10, np.copy(ev)):
        e[0] += int(0.6 * raw.info["sfreq"]) - raw.first_samp
        e[2] = 0
        ev_annot.append(e)
    annots = annotations_from_events(
        np.array(ev_annot), sfreq=raw.info["sfreq"], event_desc={0: "covert"}
    )
    annots.set_durations(1)
    update_annotations(raw, annots)
    raw.pick_types(meg=False, misc=True)
    raw.plot(block=True, events=ev, decim=20)  # pyright: ignore

    logger.info(f"Annotations: {raw.annotations}")
    write_annotations(cfg.output.annots, raw.annotations)


if __name__ == "__main__":
    main()
