CPU Installation
----------------
- Install conda
- Create new conda environment from yaml:
```
conda env create -f environment.yml
```
- Install poetry
- Activate the environment with
```
conda activate speech3.8
```
- Install packages with
```
poetry install
```
- Install custom utils with
```
pip install -e .
```
