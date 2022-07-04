#!/usr/bin/env python
"""Compute ICA solution for raw data"""
import logging
import os

import hydra
from mne import Report  # type: ignore
from mne.io import Raw  # type: ignore
from mne.preprocessing import ICA  # type: ignore
from utils import prepare_annotated_raw

logger = logging.getLogger(__name__)


def generate_report(ica: ICA) -> Report:
    logger.info("Generating report...")
    report = Report(verbose=False)
    fig_topo = ica.plot_components(picks=range(ica.n_components_), show=False)
    report.add_figs_to_section(fig_topo, section="ICA", captions="Timeseries")
    return report


def compute_ica(raw: Raw, init_cfg, fit_cfg) -> ICA:
    logger.info("Computing ICA...")
    ica = ICA(init_cfg.n_components, random_state=init_cfg.random_state)
    ica.fit(raw, picks="data", decim=fit_cfg.decim, reject_by_annotation=fit_cfg.annot_rej)
    logger.info(f"Fitted {ica.n_components_} ICA components")
    return ica


@hydra.main(config_path="../configs/", config_name="04-compute_ica")
def main(cfg):
    logger.info(f"Starting new session for {__file__}")
    logger.info(f"Current working directory is {os.getcwd()}")

    raw = prepare_annotated_raw(cfg.input.raw, None, cfg.input.annots)
    raw.filter(l_freq=cfg.filt.l_freq, h_freq=cfg.filt.h_freq)

    ica = compute_ica(raw, cfg.ica_init, cfg.ica_fit)
    ica.save(cfg.output.solution, overwrite=True)

    report = generate_report(ica)
    report.save(cfg.output.report, overwrite=True, open_browser=False)


if __name__ == "__main__":
    main()
