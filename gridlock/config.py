"""Global simulation constants.

Every tunable lives here so parameters can be found and changed in one place
rather than hunted through the rest of the code.
"""

import random

import numpy as np

# --- Spatial discretisation -------------------------------------------------

#: Physical size of one road cell in metres (roughly one car length).
CELL_SIZE_METERS = 5

# --- Nagel-Schreckenberg parameters ----------------------------------------

#: Maximum speed, in cells per time step.
DEFAULT_MAX_SPEED = 5

#: Probability that a moving car randomly drops one unit of speed each step.
#: This is the "dawdle" rule, and it is what makes phantom jams possible.
DEFAULT_PROB_SLOW = 0.5

# --- Simulation parameters --------------------------------------------------

#: Number of cars placed on the network in a single trial.
DEFAULT_NUM_CARS = 100

#: Total time steps per trial, burn-in included.
DEFAULT_NUM_STEPS = 50

#: Leading steps discarded before statistics are recorded, so that measurements
#: describe a quasi-steady state rather than the artificial initial condition.
DEFAULT_BURN_IN_STEPS = 20

#: Independent trials used for the empirical congestion analysis.
NUM_TRIALS = 100

# --- Networks ---------------------------------------------------------------

#: Radius in metres around the centre address to download.
NETWORK_RADIUS_METERS = 1000

#: Dense, irregular inner-city grid in Kreuzberg.
BERLIN_ADDRESS = "Adalbertstrasse 58, Berlin, Germany"

#: Radial, arterial-dominated network around the Obelisco. "s/n" is "sin numero"
#: (Spanish: without number), used for addresses at landmarks and intersections
#: that carry no building number.
BUENOS_AIRES_ADDRESS = "Avenida 9 de Julio s/n, Buenos Aires, Argentina"

# --- Reproducibility --------------------------------------------------------

#: Base seed. Individual trials use RANDOM_SEED + trial_index.
RANDOM_SEED = 42


def set_global_seed(seed: int = RANDOM_SEED) -> None:
    """Seed both random number generators used by the simulation."""
    random.seed(seed)
    np.random.seed(seed)
