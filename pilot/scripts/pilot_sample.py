from pilot.fields import ClassicalSpinField
from pilot.lattice import Lattice2D
from pilot.sample import ClassicalSpinSampler, XYSampler, HeisenbergSampler
from pilot.observables import Observables
from pilot.report import make_report

from argparse import ArgumentParser

import pilot.params as p  # TODO: runcards..

F = ClassicalSpinField
if p.N == 2:
    S = XYSampler
elif p.N == 3:
    S = HeisenbergSampler
else:
    S = ClassicalSpinSampler

parser = ArgumentParser()
parser.add_argument(
    "-o",
    "--output",
    metavar="",
    help="output directory, default: 'output/'",
    default="output/",
)
parser.add_argument(
    "-m",
    "--mode",
    metavar="",
    help="reduced output for preliminary measurements. Options: 'full', 'therm', 'autocorr'. Default: 'full'.",
    choices=["full", "therm", "autocorr"],
    default="full",
)


def main():

    args = parser.parse_args()

    # Construct lattice object
    lattice = Lattice2D(p.lattice_length)

    # Construct field object with random initial configuration
    field = F.from_random(lattice, N=p.N, beta=p.beta)

    # Construct MCMC algorithm object
    sampler = S(field, algorithm=p.algorithm, delta=p.delta)

    # Generate sample from MCMC
    sample = sampler(p.sample_size, p.sample_interval, p.thermalisation)

    # Construct ensemble object
    ensemble = F.new_like(sample, template=field)

    # Construct observables object
    observables = Observables(ensemble)

    # Compute observables and create report
    make_report(
        lattice, field, sampler, observables, output=args.output, mode=args.mode,
    )


if __name__ == "__main__":
    main()
