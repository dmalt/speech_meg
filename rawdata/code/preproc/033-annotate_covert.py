#!/usr/bin/env python
"""Covert segments automatic annotation based on events"""
import logging
from dataclasses import dataclass
from typing import Optional

import hydra
import matplotlib  # type: ignore
import mne  # type: ignore
import numpy as np
from hydra.core.config_store import ConfigStore
from utils import BaseConfig, prepare_script

matplotlib.use("TkAgg")
logger = logging.getLogger(__file__)


@dataclass
class Input:
    raw: str


@dataclass
class Output:
    annots: str


@dataclass
class Config(BaseConfig):
    input: Input
    output: Output
    check: bool
    decim: Optional[int] = None


cs = ConfigStore.instance()
cs.store(name="schema", node=Config)


def get_covert_event_annots(raw: mne.io.Raw, ev: np.ndarray) -> mne.Annotations:
    ev_annot = []
    for e in filter(lambda x: x[2] > 10, np.copy(ev)):
        e[0] += int(0.6 * raw.info["sfreq"]) - raw.first_samp
        e[2] = 0
        ev_annot.append(e)
    annots = mne.annotations_from_events(
        np.array(ev_annot), sfreq=raw.info["sfreq"], event_desc={0: "covert"}
    )
    annots.set_durations(1)
    return annots


@hydra.main(config_path="../configs/", config_name="033-annotate_covert")
def main(cfg: Config):
    prepare_script(logger, script_name=__file__)
    assert cfg.task in ("overtcovert", "test"), f"No covert speech for {cfg.task}"

    logger.info("Preparing raw data")
    raw = mne.io.read_raw_fif(cfg.input.raw)
    ev = mne.find_events(raw, min_duration=1, output="step")
    annots = get_covert_event_annots(raw, ev)
    raw.set_annotations(annots)

    if cfg.check:
        raw.pick_types(meg=False, misc=True)
        decim = cfg.decim if cfg.decim is not None else "auto"
        raw.plot(block=True, events=ev, decim=decim)  # pyright: ignore

    logger.info(f"Annotations: {raw.annotations}")
    raw.annotations.save(cfg.output.annots, overwrite=True)


if __name__ == "__main__":
    main()
