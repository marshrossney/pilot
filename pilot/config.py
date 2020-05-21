import configargparse

parser = configargparse.ArgParser()

# ------------------------ #
#  Command line arguments  #
# ------------------------ #
parser.add(
    "-c", "--config", required=True, is_config_file=True, help="path to config file"
)
parser.add_argument(
    "-o",
    "--output",
    help="path to output directory, default: 'output/'",
    default="output/",
)
parser.add_argument(
    "-m",
    "--mode",
    help="reduced output for preliminary measurements. Options: 'full', 'therm', 'autocorr'. Default: 'full'.",
    choices=["full", "therm", "autocorr"],
    default="full",
)

# ------------------------ #
#  Config file parameters  #
# ------------------------ #
parser.add(
    "--lattice_length",
    metavar="LENGTH",
    type=int,
    required=True,
    help="size of single lattice dimension",
)
parser.add(
    "--euclidean_dimension",
    metavar="N",
    type=int,
    required=True,
    help="euclidean dimension of the fields",
)
parser.add("--beta", type=float, required=True, help="coupling strength parameter")
parser.add(
    "--algorithm",
    help="MCMC algorithm used for sampling",
    choices=["metropolis", "heatbath"],
    default="metropolis",
)
parser.add(
    "--max_step",
    metavar="STEP",
    type=float,
    help="step size parameter for MCMC algorithm",
    default=1,
)

parser.add(
    "--sample_size",
    metavar="SIZE",
    type=int,
    help="number of configurations in output sample",
    default=10000,
)
parser.add(
    "--sample_interval",
    metavar="INTERVAL",
    type=int,
    help="number of updates (measured in sweeps) to discard between configurations which make up the output sample",
    default=1,
)
parser.add(
    "--thermalisation",
    metavar="THERM",
    type=int,
    help="number of updates (measured in sweeps) to discard before configurations start to be added to the output sample",
    default=1,
)
# NOTE: currently does nothing
parser.add(
    "--bootstrap_sample_size",
    metavar="SIZE",
    type=int,
    help="number of ensembles in the bootstrap sample",
    default=1000,
)

args = parser.parse_args()

# ----------------- #
#  Run some checks  #
# ----------------- #
if args.euclidean_dimension != 3 and args.algorithm == "heatbath":
    parser.error("Heat bath not implemented for N != 3")

# ------------- #
#  Adjustments  #
# ------------- #
if args.algorithm == "heatbath":
    args.delta = "N/A"
if args.algorithm == "metropolis" and args.euclidean_dimension > 3:
    args.delta = "N/A"
