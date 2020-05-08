from pilot.lattice import Lattice2D
from pilot.fields import ClassicalSpinField
from pilot.sample import XYSampler, HeisenbergSampler, ClassicalSpinSampler
from pilot.observables import Observables
from pilot.report import make_report

from pilot.config import args


def main():

    # Construct lattice object
    lattice = Lattice2D(args.lattice_length)

    # Construct field object with random initial configuration
    field = ClassicalSpinField.from_random(
        lattice, N=args.euclidean_dimension, beta=args.beta
    )

    # Construct MCMC algorithm object
    if args.euclidean_dimension == 2:
        sampler = XYSampler(field, delta=args.max_step)
    elif args.euclidean_dimension == 3:
        sampler = HeisenbergSampler(field, delta=args.max_step)
    else:
        sampler = ClassicalSpinSampler(field)

    # Generate sample from MCMC
    sample = sampler(args.sample_size, args.sample_interval, args.thermalisation)

    # Construct ensemble object
    ensemble = ClassicalSpinField.new_like(sample, template=field)

    # Construct observables object
    observables = Observables(ensemble)

    # Compute observables and create report
    make_report(
        lattice, field, sampler, observables, output=args.output, mode=args.mode,
    )


if __name__ == "__main__":
    main()
