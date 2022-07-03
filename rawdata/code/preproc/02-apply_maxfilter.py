"""
Perform Maxwell filtering on annotated data with SSS
Note
----
requires mne >= 0.20 for filtering line noise with filter_chpi
for emptyroom data
"""
import logging
import os

import hydra
from mne.channels import fix_mag_coil_types  # type: ignore
from mne.chpi import filter_chpi  # type: ignore
from mne.io import read_raw_fif  # type: ignore
from mne.preprocessing import maxwell_filter  # type: ignore
from omegaconf import OmegaConf
from speech import config as cfg  # type: ignore

logger = logging.getLogger(__name__)


def prepare_raw(raw_path, bads_path, annot_path, is_er):
    """Load raw, filter chpi and line noise, set bads and annotations"""
    raw = read_raw_fif(raw_path, preload=True)
    filter_chpi(raw, allow_line_only=is_er, t_window=cfg.maxfilt_config["t_window"])
    fix_mag_coil_types(raw.info)

    with open(bads_path, "r") as f:
        bads = f.readline().split("\t")
        if bads == [""]:
            bads = []
        raw.info["bads"] = bads

    return raw


def apply_maxfilter(raw_path, bads_path, annot_path, maxfilt_path, is_er):
    raw = prepare_raw(raw_path, bads_path, annot_path, is_er)

    coord_frame = "head" if is_er else "meg"

    raw_sss = maxwell_filter(
        raw,
        cross_talk=cfg.crosstalk_path,
        calibration=cfg.cal_path,
        skip_by_annotation=[],
        coord_frame=coord_frame,
    )

    raw_sss.save(maxfilt_path, overwrite=True)


@hydra.main(config_path="../configs/", config_name="02-apply_maxfilter")
def main(cfg):

    print(OmegaConf.to_yaml(cfg))
    OmegaConf.resolve(cfg)  # type: ignore
    print(OmegaConf.to_yaml(cfg))
    print(f"Current working directory is {os.getcwd()}")
    # input
    raw = cfg.raw_path
    # output
    bads = cfg.annotate_for_maxfilt.bad_channels
    annot = cfg.annotate_for_maxfilt.annotations
    # output
    maxfilt = cfg.maxfiltered

    logger.info(f"{bads=}, {annot=}, {maxfilt=}")

    apply_maxfilter(raw, bads, annot, maxfilt, False)


if __name__ == "__main__":
    main()
