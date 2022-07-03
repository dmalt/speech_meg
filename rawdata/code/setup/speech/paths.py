from pathlib import Path

from .mi_config import bands

BIDS_ROOT = Path("/home/altukhov/Data/speech_meg/rawdata")
DERIVATIVES = BIDS_ROOT / "derivatives"

BADS = DERIVATIVES / "01-maxfilt_bads"
MAXFILT = DERIVATIVES / "02-apply_maxfilt"
MAXFILT_ANNOT = DERIVATIVES / "02a-maxfilt_bad_segments"
ICA = DERIVATIVES / "03-compute_ica"
ICA_BADS = DERIVATIVES / "04-inspect_ica"
ICA_CLEAN = DERIVATIVES / "05-apply_ica"
FINAL = DERIVATIVES / "06-annotate_muscles"
SPEECH_ALIGN = DERIVATIVES / "07-align_speech"
MEG_ENV = DERIVATIVES / "meg_envelopes"
MFCCS = DERIVATIVES / "mfccs"
MFCCS_MI = DERIVATIVES / "mfcc_mi"
RESAMPLED = DERIVATIVES / "08-resample"


# s = subject = "01"
s = subject = "test"
t = task = "speech"
# s = subject = "02"
# t = task = "overtcovert"
ses = "20220222"
bs = bids_subj = f"sub-{subject}"
bt = bids_task = f"task-{task}"
crop_proc = "proc-maxfiltannotcrop"

subj_path = BIDS_ROOT / f"sub-{s}" / "meg"
raw_path = subj_path / f"{bs}_{bt}_meg.fif"
cal_path = subj_path / f"{bs}_acq-calibration_meg.dat"
crosstalk_path = subj_path / f"{bs}_acq-crosstalk_meg.fif"

er_dir = BIDS_ROOT / "sub-emptyroom" / f"ses-{ses}" / "meg"
er_path = er_dir / f"sub-emptyroom_ses-{ses}_task-noise_meg.fif"

maxfilt_bads_path = BADS / bs / f"{bs}_{bt}_bads.tsv"
maxfilt_annotations_path = BADS / bs / f"{bs}_{bt}_annot.fif"
maxfilt_path = MAXFILT / bs / f"{bs}_{bt}_proc-maxfiltnoannot_meg.fif"

postmaxfilt_bads_path = MAXFILT_ANNOT / bs / f"{bs}_{bt}_bads.tsv"
postmaxfilt_annotations_path = MAXFILT_ANNOT / bs / f"{bs}_{bt}_annot.fif"

# maxfilt_path = MAXFILT / bs / f"{bs}_{bt}_proc-maxfiltannottwice_meg.fif"
# annotated_path = MAXFILT / bs / f"{bs}_{bt}_proc-maxfiltannottwice_meg.fif"
cropped_path = MAXFILT / bs / f"{bs}_{bt}_{crop_proc}_meg.fif"

ica_sol_path = ICA / bs / f"{bs}_{bt}_proc-maxfiltannottwice_ica.fif"
ica_bads_path = ICA_BADS / bs / f"{bs}_{bt}_proc-maxfiltannottwice_bads.tsv"
ica_cleaned = ICA_CLEAN / bs / f"{bs}_{bt}_proc-icaclean_meg.fif"
ica_muscle_report = ICA_BADS / bs / "find_muscle_ics.html"

final_annotations = FINAL / bs / f"{bs}_{bt}_proc-badseg_annot.fif"

audio_path = BIDS_ROOT / f"{bs}/beh/{bs}_beh.wav"
audio_align_path = SPEECH_ALIGN / f"sub-{s}_proc-align_beh.wav"
audio_align_report_path = SPEECH_ALIGN / f"sub-{s}_proc-align_report.html"

mfccs_ica_path = MFCCS / bs / "mfccs_ica.npy"

meg_env = {b: MEG_ENV / (b + ".fif") for b in bands}
mfcc_mi_paths = {b: MFCCS_MI / f"mfcc_shifts_{b}.npy" for b in bands}

h5_path = BIDS_ROOT / "meg.h5"

resampled_path = RESAMPLED / f"sub-{s}" / f"{bs}_{bt}_proc-resample_meg.fif"


DERIVATIVES.mkdir(exist_ok=True)
BADS.mkdir(exist_ok=True)
MAXFILT.mkdir(exist_ok=True)
ICA.mkdir(exist_ok=True)
ICA_BADS.mkdir(exist_ok=True)
ICA_CLEAN.mkdir(exist_ok=True)
FINAL.mkdir(exist_ok=True)
SPEECH_ALIGN.mkdir(exist_ok=True)
MEG_ENV.mkdir(exist_ok=True)
MFCCS.mkdir(exist_ok=True)
MFCCS_MI.mkdir(exist_ok=True)
MAXFILT_ANNOT.mkdir(exist_ok=True)
RESAMPLED.mkdir(exist_ok=True)

postmaxfilt_bads_path.parent.mkdir(exist_ok=True)
maxfilt_bads_path.parent.mkdir(exist_ok=True)
maxfilt_path.parent.mkdir(exist_ok=True)
ica_sol_path.parent.mkdir(exist_ok=True)
ica_bads_path.parent.mkdir(exist_ok=True)
ica_cleaned.parent.mkdir(exist_ok=True)
mfccs_ica_path.parent.mkdir(exist_ok=True)
final_annotations.parent.mkdir(exist_ok=True)
resampled_path.parent.mkdir(exist_ok=True)

if __name__ == "__main__":
    assert BIDS_ROOT.exists()
    assert subj_path.exists()
    assert raw_path.exists()
    assert cal_path.exists()
    assert crosstalk_path.exists()
    assert DERIVATIVES.exists()
    assert er_path.exists()
