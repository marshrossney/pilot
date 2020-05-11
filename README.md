# Pilot

## Installation

Create a conda environment and install dependencies:
```
conda create --name pilot python=3.7
conda install -c anaconda numpy scipy matplotlib pandas
conda install -c anaconda tqdm configargparse pypandoc tabulate
```

Now pilot can be installed:
```
python -m pip install -e .
```

To run the tests, one needs to first install pytest
```
conda install -c anaconda pytest
pytest --pyargs pilot
```

## Usage

Run the sampler using configuration file `input.yml`
```
pilot-sample -c input.yml
```

Specify a different output directory:
```
pilot-sample -c input.yml -o test_directory
```

Just calculate autocorrelations (skip the observables)
```
pilot-sample -c input.yml -m autocorr
```

Example runcards are provided in the `examples/` directory.


## To do

Code:
- More unit tests
- More docstrings
- Bootstrap sample size in runcard

Content:
- O2 heat bath
- Concurrent samples as well as bootstrap
- (Over-relaxed) Heat bath
- Additional fields (scalar, CPN)
