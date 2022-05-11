from pathlib import Path

from .mi_config import bands

BIDS_ROOT = Path("/home/altukhov/Data/speech/rawdata")
DERIVATIVES = BIDS_ROOT / "derivatives"

BADS = DERIVATIVES / "01-maxfilt_bads"
MAXFILT = DERIVATIVES / "02-apply_maxfilt"
ICA = DERIVATIVES / "03-compute_ica"
ICA_BADS = DERIVATIVES / "04-inspect_ica"
ICA_CLEAN = DERIVATIVES / "05-apply_ica"
SPEECH_ALIGN = DERIVATIVES / "06-align_speech"
MEG_ENV = DERIVATIVES / "07-meg_envelopes"
MFCCS = DERIVATIVES / "08-mfccs"
MFCCS_MI = DERIVATIVES / "08-mfcc_mi"


s = subject = "01"
t = task = "speech"
ses = "20220222"
bs = bids_subj = f"sub-{subject}"
bt = bids_task = f"task-{task}"
crop_proc = "proc-maxfiltannottwicecrop"

subj_path = BIDS_ROOT / "sub-01" / "meg"
raw_path = subj_path / f"{bs}_{bt}_meg.fif"
cal_path = subj_path / f"{bs}_acq-calibration_meg.dat"
crosstalk_path = subj_path / f"{bs}_acq-crosstalk_meg.fif"

er_dir = BIDS_ROOT / "sub-emptyroom" / f"ses-{ses}" / "meg"
er_path = er_dir / f"sub-emptyroom_ses-{ses}_task-noise_meg.fif"

maxfilt_bads_path = BADS / bs / f"{bs}_{bt}_bads.tsv"
maxfilt_annotations_path = BADS / bs / f"{bs}_{bt}_annot.fif"
maxfilt_path = MAXFILT / bs / f"{bs}_{bt}_proc-maxfiltnoannot_meg.fif"

maxfilt_path = MAXFILT / bs / f"{bs}_{bt}_proc-maxfiltannottwice_meg.fif"
annotated_path = MAXFILT / bs / f"{bs}_{bt}_proc-maxfiltannottwice_meg.fif"
cropped_path = MAXFILT / bs / f"{bs}_{bt}_{crop_proc}_meg.fif"

ica_sol_path = ICA / bs / f"{bs}_{bt}_proc-maxfiltannottwice_ica.fif"
ica_bads_path = ICA_BADS / bs / f"{bs}_{bt}_proc-maxfiltannottwice_bads.tsv"
ica_cleaned = ICA_CLEAN / bs / f"{bs}_{bt}_proc-icaclean_meg.fif"
ica_muscle_report = ICA_BADS / bs / "find_muscle_ics.html"

audio_path = BIDS_ROOT / f"/{bs}/beh/{bs}_beh.wav"
audio_align_path = SPEECH_ALIGN / "sub-01_proc-align_beh.wav"
audio_align_report_path = SPEECH_ALIGN / "sub-01_proc-align_report.html"

mfccs_ica_path = MFCCS / bs / "mfccs_ica.npy"

meg_env = {b: MEG_ENV / (b + ".fif") for b in bands}
mfcc_mi_paths = {b: MFCCS_MI / f"mfcc_shifts_{b}.npy" for b in bands}

h5_path = BIDS_ROOT / "meg.h5"


DERIVATIVES.mkdir(exist_ok=True)
BADS.mkdir(exist_ok=True)
MAXFILT.mkdir(exist_ok=True)
ICA.mkdir(exist_ok=True)
ICA_BADS.mkdir(exist_ok=True)
ICA_CLEAN.mkdir(exist_ok=True)
SPEECH_ALIGN.mkdir(exist_ok=True)
MEG_ENV.mkdir(exist_ok=True)
MFCCS.mkdir(exist_ok=True)
MFCCS_MI.mkdir(exist_ok=True)

maxfilt_bads_path.parent.mkdir(exist_ok=True)
maxfilt_path.parent.mkdir(exist_ok=True)
ica_sol_path.parent.mkdir(exist_ok=True)
ica_bads_path.parent.mkdir(exist_ok=True)
ica_cleaned.parent.mkdir(exist_ok=True)
mfccs_ica_path.parent.mkdir(exist_ok=True)

if __name__ == "__main__":
    assert BIDS_ROOT.exists()
    assert subj_path.exists()
    assert raw_path.exists()
    assert cal_path.exists()
    assert crosstalk_path.exists()
    assert DERIVATIVES.exists()
    assert er_path.exists()
