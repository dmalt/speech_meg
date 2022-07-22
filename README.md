DVC repository for MEG overt/covert speech dataset.

It contains conda environment setup and scripts for data preprocessing with MNE-Python,
DVC configuration to download the data from GDrive and Python API to load the data
directly into a Python project.


Quickstart
----------
### To only load the data
1) Clone this repo:
```bash
git clone https://github.com/dmalt/speech_meg.git
```

2) Install DVC and DVC-gdrive:
- with pip:
```bash
pip install dvc dvc[gdrive]
```
- with conda:
```bash
conda install -c conda-forge dvc dvc-gdrive
```
3) From the project root run
```
dvc pull
```

4) Complete the authentification step.

At this point DVC will ask for an authentification with your Google account.
Follow the link in the terminal. In the opened browser window select
the Google account with which the data were shared and click on both checkboxes.
If the data were shared with you, the download should start after the authentification.

5) Come back next morning :)

In case of success, the following data will be loaded (18 GB in total):
- raw MEG and audio data @ `rawdata`,
- data annotations @ `rawdata/derivatives/011-annotate_premaxfilt`,
  `rawdata/derivatives/031-annotate_postmaxfilt`,
  `rawdata/derivatives/032-annotate_speech`
  `rawdata/derivatives/033-annotate_covert`,
  `rawdata/derivatives/071-annotate_muscles`,
  `rawdata/derivatives/101-merge_annotations`
- manually marked bad ICA components @ `rawdata/derivatives/051-inspect_ica`
- aligned audio data @ `rawdata/derivatives/081-align_audio`
- downsampled and ICA-cleaned MEG data @ `rawdata/derivatives/091-resample`

All the intermediate files will not be downloaded since they can be recomputed via
running the corresponding scripts from `rawdata/code/preproc`
