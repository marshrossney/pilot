# Pilot

## Installation

Dependencies:
- numpy
- scipy
- matplotlib
- pandas
- tqdm
- pypandoc
- configargparser
- pytest

Install pilot:
```bash
python -m pip install -e .
```

## Usage

Run the sampler:
```bash
pilot-sample -c parameters_file.yml -o path_to_output_dir
```

## To do

Code:
- more unit tests
- add more docstrings
- bootstrap sample size in runcard

Content:
- O2 heat bath
- Concurrent samples as well as bootstrap
- (Over-relaxed) Heat bath
- Additional fields (scalar, CPN)
