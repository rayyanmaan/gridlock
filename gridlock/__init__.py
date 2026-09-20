"""
Gridlock
========

Agent-based traffic simulation on real city street networks, plus tools for
asking which roads jam and whether network topology alone can predict it.

Quick start
-----------
>>> from gridlock import RoadNetwork, TrafficSimulation, TrafficVisualizer
>>> net = RoadNetwork("Adalbertstrasse 58, Berlin, Germany", radius_meters=1000)
>>> sim = TrafficSimulation(net, num_cars=100, num_steps=50)
>>> sim.run()
>>> TrafficVisualizer.plot_congestion_map(net, sim.get_mean_congestion_per_edge())
"""

from gridlock.config import (
    CELL_SIZE_METERS,
    DEFAULT_MAX_SPEED,
    DEFAULT_PROB_SLOW,
    DEFAULT_NUM_CARS,
    DEFAULT_NUM_STEPS,
    DEFAULT_BURN_IN_STEPS,
    NUM_TRIALS,
    NETWORK_RADIUS_METERS,
    RANDOM_SEED,
    BERLIN_ADDRESS,
    BUENOS_AIRES_ADDRESS,
    set_global_seed,
)
from gridlock.network import RoadNetwork
from gridlock.simulation import Car, TrafficSimulation
from gridlock.visualize import TrafficVisualizer
from gridlock.analysis import (
    compute_ns_flow_vs_density,
    find_congestion_threshold,
    is_edge_congested,
    run_multiple_trials,
    compute_network_metrics,
    attach_congestion_to_metrics,
    compute_metric_correlations,
    compute_network_fundamental_diagram,
    METRIC_COLUMNS,
)

__version__ = "1.0.0"

__all__ = [
    "RoadNetwork",
    "Car",
    "TrafficSimulation",
    "TrafficVisualizer",
    "compute_ns_flow_vs_density",
    "find_congestion_threshold",
    "is_edge_congested",
    "run_multiple_trials",
    "compute_network_metrics",
    "attach_congestion_to_metrics",
    "compute_metric_correlations",
    "compute_network_fundamental_diagram",
    "METRIC_COLUMNS",
    "set_global_seed",
    "CELL_SIZE_METERS",
    "DEFAULT_MAX_SPEED",
    "DEFAULT_PROB_SLOW",
    "DEFAULT_NUM_CARS",
    "DEFAULT_NUM_STEPS",
    "DEFAULT_BURN_IN_STEPS",
    "NUM_TRIALS",
    "NETWORK_RADIUS_METERS",
    "RANDOM_SEED",
    "BERLIN_ADDRESS",
    "BUENOS_AIRES_ADDRESS",
    "__version__",
]
