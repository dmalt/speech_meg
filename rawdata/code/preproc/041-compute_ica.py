#!/usr/bin/env python
"""Compute ICA solution for raw data"""
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import hydra
import mne  # type: ignore
from hydra.core.config_store import ConfigStore
from mne.preprocessing import ICA  # type: ignore
from utils import BaseConfig

logger = logging.getLogger(__name__)


@dataclass
class Input:
    raw: str
    annots: str


@dataclass
class Output:
    report: str
    solution: str


@dataclass
class IcaInitParams:
    n_components: float
    random_state: int


@dataclass
class IcaFitParams:
    decim: int
    annot_rej: bool


@dataclass
class FiltParams:
    l_freq: Optional[int]
    h_freq: Optional[int]


@dataclass
class Config(BaseConfig):
    input: Input
    output: Output
    ica_init: IcaInitParams
    ica_fit: IcaFitParams
    filt: FiltParams


cs = ConfigStore.instance()
cs.store(name="schema", node=Config)


def generate_report(ica: ICA) -> mne.Report:
    report = mne.Report(verbose=False)
    fig_topo = ica.plot_components(picks=range(ica.n_components_), show=False)
    report.add_figs_to_section(fig_topo, section="ICA", captions="Timeseries")
    return report


def compute_ica(raw: mne.io.Raw, init_cfg: IcaInitParams, fit_cfg: IcaFitParams) -> ICA:
    ica = ICA(init_cfg.n_components, random_state=init_cfg.random_state)
    ica.fit(raw, picks="data", decim=fit_cfg.decim, reject_by_annotation=fit_cfg.annot_rej)
    logger.info(f"Fitted {ica.n_components_} ICA components")
    return ica


@hydra.main(config_path="../configs/", config_name="041-compute_ica")
def main(cfg: Config):
    logger.info(f"Starting new session for {__file__}")
    logger.info(f"Current working directory is {os.getcwd()}")

    raw = mne.io.read_raw_fif(cfg.input.raw, preload=True)
    if Path(cfg.input.annots).exists():
        raw.set_annotations(mne.read_annotations(cfg.input.annots))
    else:
        logger.warning(f"Annotation file is missing at {cfg.input.annots}")

    raw.filter(l_freq=cfg.filt.l_freq, h_freq=cfg.filt.h_freq)

    logger.info("Computing ICA...")
    ica = compute_ica(raw, cfg.ica_init, cfg.ica_fit)
    ica.save(cfg.output.solution, overwrite=True)
    logger.info(f"Fitted ICA solution: {ica}")

    logger.info("Generating report...")
    report = generate_report(ica)
    report.save(cfg.output.report, overwrite=True, open_browser=False)


if __name__ == "__main__":
    main()
