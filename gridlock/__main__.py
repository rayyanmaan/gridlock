"""
Command-line entry point: ``python -m gridlock``.

Runs the full study for one city and prints the headline numbers.

Examples
--------
Reproduce the Berlin analysis::

    python -m gridlock --city berlin

A fast smoke test that finishes in a couple of minutes::

    python -m gridlock --city berlin --trials 3 --radius 400 --skip-sweep

Any address works::

    python -m gridlock --address "Shibuya Crossing, Tokyo, Japan" --trials 20
"""

import argparse
import sys

import numpy as np

from gridlock.analysis import (
    METRIC_COLUMNS,
    attach_congestion_to_metrics,
    compute_metric_correlations,
    compute_network_fundamental_diagram,
    compute_network_metrics,
    find_congestion_threshold,
    run_multiple_trials,
)
from gridlock.config import (
    BERLIN_ADDRESS,
    BUENOS_AIRES_ADDRESS,
    DEFAULT_NUM_CARS,
    DEFAULT_NUM_STEPS,
    NETWORK_RADIUS_METERS,
    NUM_TRIALS,
    set_global_seed,
)
from gridlock.network import RoadNetwork

CITIES = {
    "berlin": BERLIN_ADDRESS,
    "buenos-aires": BUENOS_AIRES_ADDRESS,
}

CAR_COUNT_SWEEP = [20, 40, 60, 80, 100, 130, 160, 200, 250, 300]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m gridlock",
        description="Simulate traffic on a real street network and test "
        "whether topology predicts congestion.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--city",
        choices=sorted(CITIES),
        default="berlin",
        help="One of the two networks from the report (default: berlin).",
    )
    parser.add_argument(
        "--address",
        default=None,
        help="Any address, overriding --city.",
    )
    parser.add_argument(
        "--radius",
        type=int,
        default=NETWORK_RADIUS_METERS,
        help=f"Network radius in metres (default: {NETWORK_RADIUS_METERS}).",
    )
    parser.add_argument(
        "--cars",
        type=int,
        default=DEFAULT_NUM_CARS,
        help=f"Cars per trial (default: {DEFAULT_NUM_CARS}).",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=DEFAULT_NUM_STEPS,
        help=f"Time steps per trial (default: {DEFAULT_NUM_STEPS}).",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=NUM_TRIALS,
        help=f"Independent trials (default: {NUM_TRIALS}).",
    )
    parser.add_argument(
        "--skip-threshold",
        action="store_true",
        help="Skip deriving the congestion threshold from the fundamental diagram.",
    )
    parser.add_argument(
        "--skip-sweep",
        action="store_true",
        help="Skip the network-level car-count sweep.",
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    address = args.address or CITIES[args.city]

    set_global_seed()

    print("=" * 68)
    print("GRIDLOCK")
    print("=" * 68)

    network = RoadNetwork(address=address, radius_meters=args.radius)

    if not args.skip_threshold:
        print("\nDeriving the congestion threshold from the NS fundamental diagram...")
        threshold, _, _, _ = find_congestion_threshold(density_resolution=201)
        print(f"  Congestion threshold: {threshold:.3f} cars per cell")

    print(f"\nRunning {args.trials} independent trials...")
    mean_congestion, std_congestion, peak_congestion = run_multiple_trials(
        road_network=network,
        num_trials=args.trials,
        num_cars=args.cars,
        num_steps=args.steps,
        progress_every=max(1, args.trials // 5),
    )

    print("\nComputing structural metrics...")
    analysis = attach_congestion_to_metrics(
        compute_network_metrics(network), mean_congestion, peak_congestion
    )
    correlations = compute_metric_correlations(analysis, METRIC_COLUMNS)

    print("\n" + "=" * 68)
    print(f"RESULTS  |  {address}")
    print("=" * 68)
    print(
        f"  Network size: {network.graph.number_of_nodes()} intersections, "
        f"{network.graph.number_of_edges()} road segments"
    )
    print(
        f"  Segments ever occupied: "
        f"{sum(1 for v in mean_congestion.values() if v > 0)}"
    )
    print(f"  Busiest segment, mean: {max(mean_congestion.values()):.3f} cars")
    print(f"  Busiest segment, peak: {max(peak_congestion.values())} cars")

    print("\n  Spearman rho vs. mean congestion, best predictor first:")
    ranked = correlations.reindex(
        correlations["rho_mean_congestion"].abs().sort_values(ascending=False).index
    )
    for _, row in ranked.iterrows():
        print(
            f"    {row['metric']:<32} rho = {row['rho_mean_congestion']:+.3f} "
            f"(p = {row['p_value_mean_congestion']:.4f})"
        )

    if not args.skip_sweep:
        print("\n  Network-level fundamental diagram:")
        sweep = compute_network_fundamental_diagram(
            road_network=network,
            car_counts_to_test=CAR_COUNT_SWEEP,
            num_trials_per_count=10,
            num_steps=args.steps,
            verbose=False,
        )
        best = sweep.loc[sweep["mean_network_speed"].idxmax()]
        print(
            f"    Optimal load: {int(best['num_cars'])} cars "
            f"({best['mean_network_speed']:.3f} cells per step)"
        )

    print("=" * 68)
    return 0


if __name__ == "__main__":
    sys.exit(main())
