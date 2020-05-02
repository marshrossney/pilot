from setuptools import setup, find_packages

with open("README.md") as f:
    LONG_DESC = f.read()

setup(
    name="pilot",
    version="0.1",
    description="Markov chain Monte Carlo sampling for lattice field theories",
    author="Joe Marsh Rossney",
    url="https://github.com/marshrossney/pilot",
    long_description=LONG_DESC,
    packages=find_packages(),
    entry_points={"console_scripts": ["pilot-sample = pilot.scripts.pilot_sample:main",]},
)
