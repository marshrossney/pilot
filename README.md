# Pilot

## Installation

Dependencies:
- numpy
- scipy
- matplotlib
- pandas
- tqdm
- pandoc
- pypandoc

Install pilot:
```bash
python -m pip install -e .
```

## Usage

Run the sampler:
```bash
pilot-sample
```

## To do

Code:
- Yaml runcard instead of params.py
- minimal command line args for script
- unit tests
- add docstrings

Content:
- Concurrent samples as well as bootstrap
- (Over-relaxed) Heat bath
- Additional fields (scalar, CPN)
