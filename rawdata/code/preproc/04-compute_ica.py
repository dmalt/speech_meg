from mne import Report  # type: ignore
from mne import read_annotations  # type: ignore
from mne.io import read_raw_fif  # type: ignore
from mne.preprocessing import ICA  # type: ignore
from speech import config as cfg  # type: ignore


def generate_report(raw, ica, report_savepath):
    report = Report(verbose=False)

    fig_topo = ica.plot_components(picks=range(ica.n_components_), show=False)
    report.add_figs_to_section(fig_topo, section="ICA", captions="Timeseries")
    report.save(report_savepath, overwrite=True, open_browser=False)


def compute_ica(fif_path, ica_sol_path, annot_path):
    raw = read_raw_fif(fif_path, preload=True)
    annotations = read_annotations(annot_path) if annot_path.exists() else None
    if annotations is not None:
        raw.set_annotations(annotations)
    raw.filter(l_freq=1, h_freq=None)
    ica = ICA(cfg.ica_config["n_components"], random_state=cfg.ica_config["random_state"])
    ica.fit(
        raw,
        picks="data",
        decim=cfg.ica_config["decim"],
        reject_by_annotation=cfg.ica_config["annot_rej"],
    )
    ica.save(ica_sol_path, overwrite=True)

    report_path = ica_sol_path.with_suffix(".html")
    generate_report(raw, ica, report_path)


if __name__ == "__main__":
    # raw = cfg.cropped_path
    raw = cfg.maxfilt_path
    # output
    ica_sol = cfg.ica_sol_path
    annot = cfg.postmaxfilt_annotations_path

    compute_ica(raw, ica_sol, annot)
