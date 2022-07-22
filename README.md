DVC repository for MEG overt/covert speech dataset.

It contains conda environment setup and scripts for data preprocessing with MNE-Python,
DVC configuration to download the data from GDrive and Python API to load the data
directly into a Python project.


Quickstart
----------
### To only load the data
1) Install DVC and DVC-gdrive:
- with pip:
```bash
pip install dvc dvc[gdrive]
```
- with conda:
```bash
conda install -c conda-forge dvc dvc-gdrive
```
2) From the project root run
```
dvc pull
```

At this point DVC will ask for an authentification with a Google account.
If the data were shared with you, the download should start after the authentification.
