"""Fundamental diagrams, multi-trial congestion statistics, and network metrics."""

import random

import networkx as nx
import numpy as np
import pandas as pd
import scipy.stats as stats

from gridlock.config import (
    DEFAULT_BURN_IN_STEPS,
    DEFAULT_MAX_SPEED,
    DEFAULT_NUM_CARS,
    DEFAULT_NUM_STEPS,
    DEFAULT_PROB_SLOW,
    NUM_TRIALS,
    RANDOM_SEED,
)
from gridlock.network import RoadNetwork
from gridlock.simulation import TrafficSimulation

#: The five structural metrics tested as congestion predictors.
METRIC_COLUMNS = [
    "edge_betweenness_centrality",
    "avg_endpoint_degree",
    "avg_endpoint_betweenness",
    "avg_endpoint_closeness",
    "edge_length_meters",
]


# --------------------------------------------------------------------------
# Single-road fundamental diagram
# --------------------------------------------------------------------------

def compute_ns_flow_vs_density(
    densities: np.ndarray,
    road_length: int = 200,
    max_speed: int = DEFAULT_MAX_SPEED,
    prob_slow: float = DEFAULT_PROB_SLOW,
    num_trials: int = 50,
    burn_in_steps: int = 50,
) -> tuple:
    """
    Flow versus density for the NS model on an isolated circular road.

    This is the calibration step. Running the same NS rules and parameters on a
    closed ring, where density is a controlled input rather than an emergent
    outcome, locates the density at which flow peaks. That peak is the
    congestion threshold used on the street network.

    Parameters
    ----------
    densities : np.ndarray
        Density values in ``[0, 1]`` to evaluate.
    road_length : int
        Cells on the circular test road.
    max_speed, prob_slow : int, float
        NS parameters. Match these to the street simulation for the resulting
        threshold to transfer.
    num_trials : int
        Independent replicates per density value.
    burn_in_steps : int
        Steps discarded before flow is measured.

    Returns
    -------
    tuple of (np.ndarray, np.ndarray)
        Mean flow and its standard error, one value per density.
    """
    mean_flows, se_flows = [], []

    for density in densities:
        trial_flows = []
        for _ in range(num_trials):
            num_cars = max(1, int(round(density * road_length)))

            road_state = np.full(road_length, -1, dtype=int)
            positions = np.random.choice(road_length, size=num_cars, replace=False)
            road_state[positions] = np.random.randint(
                0, max_speed + 1, size=num_cars
            )

            speeds = np.array([])
            for _ in range(burn_in_steps + 1):
                occupied = np.where(road_state >= 0)[0]
                if len(occupied) == 0:
                    break
                speeds = road_state[occupied].copy()

                # Rule 1: accelerate.
                speeds = np.minimum(speeds + 1, max_speed)

                # Rule 2: brake, with the gap wrapping around the ring.
                next_positions = np.roll(occupied, -1)
                gaps = (next_positions - occupied) % road_length - 1
                speeds = np.where(speeds > gaps, gaps, speeds)
                speeds = np.maximum(speeds, 0)

                # Rule 3: dawdle.
                moving = speeds > 0
                dawdle = np.random.uniform(size=len(speeds)) < prob_slow
                speeds = np.where(moving & dawdle, speeds - 1, speeds)

                # Rule 4: move.
                new_positions = (occupied + speeds) % road_length
                road_state.fill(-1)
                road_state[new_positions] = speeds

            trial_flows.append(np.sum(speeds) / road_length)

        mean_flows.append(np.mean(trial_flows))
        se_flows.append(stats.sem(trial_flows))

    return np.array(mean_flows), np.array(se_flows)


def find_congestion_threshold(
    density_resolution: int = 1001, **kwargs
) -> tuple:
    """
    Locate the peak-flow density: the congestion threshold.

    Above this density, adding cars reduces flow rather than increasing it.
    The value is a consequence of ``max_speed`` and ``prob_slow``, not a free
    parameter, so changing either moves the threshold with it.

    Returns
    -------
    tuple of (float, np.ndarray, np.ndarray, np.ndarray)
        The threshold density, the density grid, mean flows, and standard
        errors.
    """
    densities = np.linspace(0, 1, density_resolution)
    mean_flows, se_flows = compute_ns_flow_vs_density(densities, **kwargs)
    threshold = float(densities[np.argmax(mean_flows)])
    return threshold, densities, mean_flows, se_flows


def is_edge_congested(
    edge_key_tuple: tuple,
    edge_car_counts: dict,
    road_network: RoadNetwork,
    threshold: float,
) -> bool:
    """
    True when a segment's local density exceeds the congestion threshold.

    Parameters
    ----------
    edge_key_tuple : tuple
        The ``(source, target, key)`` identifier.
    edge_car_counts : dict
        Maps segment identifiers to car counts.
    road_network : RoadNetwork
        Used to look up the segment's cell count.
    threshold : float
        Density above which the segment counts as congested, typically from
        :func:`find_congestion_threshold`.
    """
    source, target, key = edge_key_tuple
    graph = road_network.graph
    if not graph.has_edge(source, target, key):
        return False
    num_cells = graph[source][target][key].get("num_cells", 1)
    return edge_car_counts.get(edge_key_tuple, 0) / num_cells > threshold


# --------------------------------------------------------------------------
# Multi-trial congestion statistics
# --------------------------------------------------------------------------

def run_multiple_trials(
    road_network: RoadNetwork,
    num_trials: int = NUM_TRIALS,
    num_cars: int = DEFAULT_NUM_CARS,
    num_steps: int = DEFAULT_NUM_STEPS,
    burn_in_steps: int = DEFAULT_BURN_IN_STEPS,
    progress_every: int = 5,
) -> tuple:
    """
    Run many independent trials and aggregate per-segment congestion.

    A single trial conflates roads that are structurally prone to congestion
    with roads that happened to get unlucky origin-destination draws. Averaging
    over many independently seeded trials separates the two.

    Returns
    -------
    tuple of (dict, dict, dict)
        Mean car count per segment across all trials and steps; the standard
        deviation of the per-trial means, showing run-to-run variability; and
        the maximum count ever observed on each segment.
    """
    per_trial_means, per_trial_peaks = [], []

    for trial_index in range(num_trials):
        # A distinct seed per trial keeps origin-destination sampling and NS
        # randomisation independent across runs.
        seed = RANDOM_SEED + trial_index
        random.seed(seed)
        np.random.seed(seed)

        simulation = TrafficSimulation(
            road_network=road_network,
            num_cars=num_cars,
            num_steps=num_steps,
            burn_in_steps=burn_in_steps,
        ).run()

        per_trial_means.append(simulation.get_mean_congestion_per_edge())
        per_trial_peaks.append(simulation.get_peak_congestion_per_edge())

        if progress_every and (trial_index + 1) % progress_every == 0:
            print(f"  Completed trial {trial_index + 1} / {num_trials}")

    all_edges = list(per_trial_means[0].keys())

    mean_congestion = {
        edge: float(np.mean([trial[edge] for trial in per_trial_means]))
        for edge in all_edges
    }
    std_congestion = {
        edge: float(np.std([trial[edge] for trial in per_trial_means]))
        for edge in all_edges
    }
    peak_congestion = {
        edge: max(trial[edge] for trial in per_trial_peaks) for edge in all_edges
    }

    return mean_congestion, std_congestion, peak_congestion


# --------------------------------------------------------------------------
# Structural metrics
# --------------------------------------------------------------------------

def compute_network_metrics(
    road_network: RoadNetwork, verbose: bool = True
) -> pd.DataFrame:
    """
    Compute five structural metrics for every segment.

    Node-level quantities (degree, betweenness, closeness) are assigned to a
    segment as the mean of its two endpoints, since a segment has no degree or
    closeness of its own. Note that ``graph.degree()`` on a directed
    MultiDiGraph returns in-degree plus out-degree, so high-degree junctions
    are those with many connections in either direction.

    Betweenness is computed with ``weight="travel_time_seconds"`` so the
    shortest paths it counts are the same ones the cars actually drive, and
    with ``normalized=True`` so values are comparable across networks of
    different sizes.

    Returns
    -------
    pd.DataFrame
        One row per segment: identifiers plus the five metrics.
    """
    graph = road_network.graph

    if verbose:
        print("Computing edge betweenness centrality (this may take a moment)...")
    edge_betweenness = nx.edge_betweenness_centrality(
        graph, weight="travel_time_seconds", normalized=True
    )

    if verbose:
        print("Computing node-level metrics...")
    node_degree = dict(graph.degree())
    node_betweenness = nx.betweenness_centrality(
        graph, weight="travel_time_seconds"
    )
    node_closeness = nx.closeness_centrality(graph)

    rows = []
    for source, target, key, data in graph.edges(data=True, keys=True):
        rows.append(
            {
                "source_node": source,
                "target_node": target,
                "edge_key": key,
                "edge_betweenness_centrality": edge_betweenness.get(
                    (source, target, key), 0.0
                ),
                "avg_endpoint_degree": (
                    node_degree.get(source, 0) + node_degree.get(target, 0)
                ) / 2,
                "avg_endpoint_betweenness": (
                    node_betweenness.get(source, 0) + node_betweenness.get(target, 0)
                ) / 2,
                "avg_endpoint_closeness": (
                    node_closeness.get(source, 0) + node_closeness.get(target, 0)
                ) / 2,
                "edge_length_meters": data.get("length", 0),
            }
        )

    return pd.DataFrame(rows)


def attach_congestion_to_metrics(
    metrics_df: pd.DataFrame, mean_congestion: dict, peak_congestion: dict
) -> pd.DataFrame:
    """Add ``mean_congestion`` and ``peak_congestion`` columns to the metrics."""
    df = metrics_df.copy()
    keys = list(zip(df["source_node"], df["target_node"], df["edge_key"]))
    df["mean_congestion"] = [mean_congestion.get(k, 0.0) for k in keys]
    df["peak_congestion"] = [peak_congestion.get(k, 0) for k in keys]
    return df


def compute_metric_correlations(
    analysis_df: pd.DataFrame,
    metric_columns: list = None,
    congestion_columns: tuple = ("mean_congestion", "peak_congestion"),
) -> pd.DataFrame:
    """
    Spearman rank correlation of each metric against each congestion measure.

    Spearman rather than Pearson: congestion is heavily right-skewed, with most
    segments carrying almost nothing and a handful carrying a lot. Pearson
    would be dominated by that tail; rank correlation is robust to it.
    """
    metric_columns = metric_columns or METRIC_COLUMNS

    rows = []
    for metric in metric_columns:
        row = {"metric": metric}
        for congestion in congestion_columns:
            result = stats.spearmanr(analysis_df[metric], analysis_df[congestion])
            row[f"rho_{congestion}"] = round(result.statistic, 3)
            row[f"p_value_{congestion}"] = round(result.pvalue, 4)
        rows.append(row)

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Network-level fundamental diagram
# --------------------------------------------------------------------------

def compute_network_fundamental_diagram(
    road_network: RoadNetwork,
    car_counts_to_test: list,
    num_trials_per_count: int = 10,
    num_steps: int = DEFAULT_NUM_STEPS,
    burn_in_steps: int = DEFAULT_BURN_IN_STEPS,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Sweep the total car count and measure mean network speed at each level.

    This asks whether the regime transition visible on a single road survives
    at city scale. Mean network speed is the network-level analogue of flow:
    high means free flow, falling means congestion is setting in.

    Returns
    -------
    pd.DataFrame
        One row per condition, with ``num_cars``, mean network density, mean
        network speed, its standard error across trials, and the mean fraction
        of cars that completed their trip.
    """
    total_cells = road_network.total_cells
    rows = []

    for num_cars in car_counts_to_test:
        densities, speeds, arrivals = [], [], []

        for trial_index in range(num_trials_per_count):
            seed = RANDOM_SEED + trial_index + num_cars
            random.seed(seed)
            np.random.seed(seed)

            simulation = TrafficSimulation(
                road_network=road_network,
                num_cars=num_cars,
                num_steps=num_steps,
                burn_in_steps=burn_in_steps,
            ).run()

            mean_counts = simulation.get_mean_congestion_per_edge()
            densities.append(sum(mean_counts.values()) / total_cells)
            speeds.append(simulation.get_mean_network_speed())
            arrivals.append(simulation.get_arrival_fraction())

        rows.append(
            {
                "num_cars": num_cars,
                "network_density": float(np.mean(densities)),
                "mean_network_speed": float(np.mean(speeds)),
                "se_network_speed": float(stats.sem(speeds))
                if len(speeds) > 1
                else 0.0,
                "fraction_cars_arrived": float(np.mean(arrivals)),
            }
        )

        if verbose:
            print(
                f"  num_cars = {num_cars}: mean speed = {rows[-1]['mean_network_speed']:.3f} "
                f"cells/step, arrived = {rows[-1]['fraction_cars_arrived']:.3f}"
            )

    return pd.DataFrame(rows)
