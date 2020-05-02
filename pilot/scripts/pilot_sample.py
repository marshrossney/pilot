from pilot.fields import ClassicalSpinField
from pilot.lattice import Lattice2D
from pilot.sample import Metropolis
from pilot.observables import Observables
from pilot.report import make_report

import pilot.params as p  # TODO: runcards..

F = ClassicalSpinField
A = Metropolis


def main():

    # Construct lattice object
    lattice = Lattice2D(p.lattice_length)

    # Construct field object with random initial configuration
    field = F.from_random(lattice, N=p.N, beta=p.beta)

    # Construct MCMC algorithm object
    algorithm = A(
        field,
        sample_size=p.sample_size,
        thermalisation=p.thermalisation,
        sample_interval=p.sample_interval,
    )

    # Generate sample from MCMC
    sample = algorithm.run()

    # Construct ensemble object
    ensemble = F.new_like(sample, template=field)

    # Construct observables object
    observables = Observables(ensemble)

    # Compute observables and create report
    make_report(lattice, field, algorithm, observables)


if __name__ == "__main__":
    main()
