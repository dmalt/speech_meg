#!/usr/bin/env python
"""
Perform Maxwell filtering on annotated data with SSS

Apply annotations from the previous step on the way

Note
----
requires mne >= 0.20 for filtering line noise with filter_chpi
for emptyroom data

"""
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import hydra
from hydra.core.config_store import ConfigStore
from mne.channels import fix_mag_coil_types  # type: ignore
from mne.chpi import filter_chpi  # type: ignore
from mne.io.fiff.raw import read_raw_fif  # type: ignore
from mne.preprocessing import maxwell_filter  # type: ignore

from utils import BaseConfig, prepare_script, read_bad_channels

logger = logging.getLogger(__file__)


@dataclass
class Input:
    raw: str
    ct: str
    cal: str
    bad_ch: str
    annots: str


@dataclass
class Output:
    maxfilt_raw: str


@dataclass
class Config(BaseConfig):
    input: Input
    output: Output
    t_window: Any = "auto"  # float or str; Union types are not supported yet


cs = ConfigStore.instance()
cs.store(name="schema", node=Config)


@hydra.main(config_path="../configs/", config_name="021-apply_maxfilter")
def main(cfg: Config):
    prepare_script(logger, script_name=__file__)

    raw = read_raw_fif(cfg.input.raw, preload=True)
    if Path(cfg.input.bad_ch).exists():
        raw.info["bads"] = read_bad_channels(cfg.input.bad_ch)
    else:
        logger.warning(f"Missing bad channels file at {cfg.input.bad_ch}")
    fix_mag_coil_types(raw.info)
    filter_chpi(raw, t_window=cfg.t_window)
    raw = maxwell_filter(
        raw, cross_talk=cfg.input.ct, calibration=cfg.input.cal, coord_frame="meg"
    )
    raw.save(cfg.output.maxfilt_raw, overwrite=True)


if __name__ == "__main__":
    main()  # pyright: ignore
