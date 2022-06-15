This package manages dependencies for speech decoding from MEG data.
It contains conda environment setup and tools for data preprocessing with MNE-Python, mostly
different paths to temporary files.

Speech decoding project uses pytorch for deep learning models definition, and it can be run
both on CPU and GPU. But the required environment for these two cases is slightly different.

Installation
------------

### CPU
- Install [conda](https://docs.conda.io/en/latest/)
- Create new conda environment from yaml:
```bash
conda env create -f environment_freeze.yml
```
- Activate the environment with
```
conda activate speech3.9
```

- Optionally, install tool from this package:
```bash
pip install -e .
```
