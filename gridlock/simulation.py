"""Cars and the Nagel-Schreckenberg simulation engine."""

import random

import networkx as nx
import numpy as np

from gridlock.config import (
    DEFAULT_BURN_IN_STEPS,
    DEFAULT_MAX_SPEED,
    DEFAULT_NUM_CARS,
    DEFAULT_NUM_STEPS,
    DEFAULT_PROB_SLOW,
)
from gridlock.network import RoadNetwork


class Car:
    """
    A single vehicle.

    A car has a fixed origin and destination and follows a pre-computed
    shortest-path route. Within the segment it currently occupies it has a cell
    position and an integer speed in cells per time step.

    Attributes
    ----------
    origin_node, destination_node : int
        Start and end intersections.
    planned_route_nodes : list of int
        Ordered nodes from origin to destination. Empty when no path exists.
    current_node_index : int
        Index into ``planned_route_nodes`` of the node the car departed from.
    current_cell_position : int
        Position within the current segment, in cells from its start.
    current_speed_cells : int
        Current speed in cells per time step.
    has_arrived : bool
        True once the destination has been reached.
    """

    __slots__ = (
        "origin_node",
        "destination_node",
        "planned_route_nodes",
        "current_node_index",
        "current_cell_position",
        "current_speed_cells",
        "has_arrived",
    )

    def __init__(self, origin_node: int, destination_node: int):
        self.origin_node = origin_node
        self.destination_node = destination_node
        self.planned_route_nodes = []
        self.current_node_index = 0
        self.current_cell_position = 0
        self.current_speed_cells = 0
        self.has_arrived = False

    def is_on_valid_route(self) -> bool:
        """True when this car has a route of at least two nodes."""
        return len(self.planned_route_nodes) >= 2

    def current_edge(self):
        """
        The ``(source, target)`` pair for the segment being traversed.

        Returns ``None`` when the car has no valid route or has run off the end
        of the one it has.
        """
        if not self.is_on_valid_route():
            return None
        if self.current_node_index >= len(self.planned_route_nodes) - 1:
            return None
        return (
            self.planned_route_nodes[self.current_node_index],
            self.planned_route_nodes[self.current_node_index + 1],
        )

    def __repr__(self) -> str:
        return (
            f"Car(origin={self.origin_node}, destination={self.destination_node}, "
            f"arrived={self.has_arrived})"
        )


class TrafficSimulation:
    """
    Nagel-Schreckenberg traffic on a real street network.

    Cars get random origin-destination pairs and route themselves by shortest
    path (weighted by travel time). Within each segment they are modelled as
    particles in a one-dimensional cell array following the four NS rules:
    accelerate, brake for the gap, dawdle, move.

    After every post-burn-in step the number of cars on each segment is
    recorded; those counts drive all downstream congestion statistics.

    Attributes
    ----------
    road_network : RoadNetwork
        The network being simulated.
    num_cars, num_steps, burn_in_steps : int
        Trial size, length, and discarded lead-in.
    prob_slow : float
        NS dawdle probability.
    max_speed : int
        NS maximum speed in cells per step.
    cars : list of Car
        Every car in the trial.
    edge_cells : dict
        Maps ``(source, target, key)`` to an int array of length ``num_cells``
        where ``-1`` means empty and any value ``>= 0`` is an occupying car's
        speed.
    edge_car_count_history : dict
        Maps ``(source, target, key)`` to the list of per-step car counts
        recorded after burn-in.
    """

    def __init__(
        self,
        road_network: RoadNetwork,
        num_cars: int = DEFAULT_NUM_CARS,
        num_steps: int = DEFAULT_NUM_STEPS,
        burn_in_steps: int = DEFAULT_BURN_IN_STEPS,
        prob_slow: float = DEFAULT_PROB_SLOW,
        max_speed: int = DEFAULT_MAX_SPEED,
        verbose: bool = False,
    ):
        self.road_network = road_network
        self.num_cars = num_cars
        self.num_steps = num_steps
        self.burn_in_steps = burn_in_steps
        self.prob_slow = prob_slow
        self.max_speed = max_speed
        self.verbose = verbose

        self._initialise_edge_cells()
        self._initialise_cars()
        self._reset_congestion_history()

    # -- Initialisation -----------------------------------------------------

    def _initialise_edge_cells(self) -> None:
        """Create an empty cell array for every segment in the network."""
        self.edge_cells = {}
        for source, target, key, data in self.road_network.graph.edges(
            data=True, keys=True
        ):
            num_cells = data.get("num_cells", 1)
            self.edge_cells[(source, target, key)] = np.full(num_cells, -1, dtype=int)

    def _initialise_cars(self) -> None:
        """
        Create cars with random origin-destination pairs and route them.

        In a directed street graph many pairs are unreachable because of
        one-way roads, and ``nx.shortest_path`` raises ``NetworkXNoPath`` for
        those. Such pairs are caught and resampled, capped by ``max_attempts``
        so a sparse graph cannot cause an infinite loop.
        """
        graph = self.road_network.graph
        all_nodes = list(graph.nodes())
        self.cars = []

        attempts = 0
        max_attempts = self.num_cars * 10

        while len(self.cars) < self.num_cars and attempts < max_attempts:
            attempts += 1
            origin = random.choice(all_nodes)
            destination = random.choice(all_nodes)

            if origin == destination:
                continue

            car = Car(origin_node=origin, destination_node=destination)
            try:
                car.planned_route_nodes = nx.shortest_path(
                    graph,
                    source=origin,
                    target=destination,
                    weight="travel_time_seconds",
                )
            except nx.NetworkXNoPath:
                continue

            self._place_car_on_first_edge(car)
            self.cars.append(car)

        if self.verbose:
            print(f"Placed {len(self.cars)} cars on the network.")

    def _place_car_on_first_edge(self, car: Car) -> None:
        """
        Put a newly routed car in cell 0 of its first segment, at speed 0.

        If that cell is already taken the car is not placed on the network for
        this trial and contributes no statistics.
        """
        if not car.is_on_valid_route():
            return

        source = car.planned_route_nodes[0]
        target = car.planned_route_nodes[1]
        edge_key = 0  # Parallel edges always resolve to key 0.

        if self.road_network.graph.has_edge(source, target, edge_key):
            cells = self.edge_cells[(source, target, edge_key)]
            if cells[0] == -1:
                cells[0] = 0
                car.current_cell_position = 0
                car.current_speed_cells = 0

    def _reset_congestion_history(self) -> None:
        """Start per-segment count history empty."""
        self.edge_car_count_history = {edge: [] for edge in self.edge_cells}

    # -- Dynamics -----------------------------------------------------------

    def _apply_ns_rules_to_edge(self, cell_array: np.ndarray) -> np.ndarray:
        """
        Apply one Nagel-Schreckenberg update to a single segment.

        The standard NS rules, adapted for a finite segment rather than the
        usual circular road:

        1. **Accelerate** -- speed rises by 1, capped at ``max_speed``.
        2. **Brake** -- speed is cut to the gap ahead when the gap is smaller.
        3. **Dawdle** -- moving cars lose 1 with probability ``prob_slow``.
        4. **Move** -- each car advances by its updated speed.

        Cars that reach the final cell are held there until
        :meth:`_advance_cars_along_routes` transfers them onward. If several
        cars are clipped to the final cell in the same step only the last is
        retained: a known boundary artefact of the finite-segment
        representation that occurs rarely in practice.
        """
        num_cells = len(cell_array)
        car_positions = np.where(cell_array >= 0)[0]

        if len(car_positions) == 0:
            return cell_array

        car_speeds = cell_array[car_positions].copy()

        # Rule 1: accelerate.
        car_speeds = np.minimum(car_speeds + 1, self.max_speed)

        # Rule 2: brake for the car ahead, or for the end of the segment.
        for index, position in enumerate(car_positions):
            if index < len(car_positions) - 1:
                gap = car_positions[index + 1] - position - 1
            else:
                gap = num_cells - position - 1
            if car_speeds[index] > gap:
                car_speeds[index] = gap

        # Rule 3: dawdle. Only moving cars can slow down.
        moving = np.where(car_speeds > 0)[0]
        slow_mask = np.random.uniform(size=len(moving)) < self.prob_slow
        car_speeds[moving] -= slow_mask.astype(int)
        car_speeds = np.maximum(car_speeds, 0)

        # Rule 4: move.
        new_positions = np.minimum(car_positions + car_speeds, num_cells - 1)

        updated = np.full(num_cells, -1, dtype=int)
        for new_pos, speed in zip(new_positions, car_speeds):
            updated[new_pos] = speed
        return updated

    def _advance_cars_along_routes(self) -> None:
        """
        Hand cars that reached a segment's end over to the next segment.

        A car in the final cell transfers to cell 0 of the next segment on its
        route when that cell is free, and waits otherwise. Cars that exhaust
        their route are marked arrived.

        The route pointer is only incremented once a transfer physically
        succeeds. Incrementing eagerly would let a blocked car's pointer drift
        out of sync with where it actually is.
        """
        graph = self.road_network.graph

        for car in self.cars:
            if car.has_arrived or not car.is_on_valid_route():
                continue

            edge_pair = car.current_edge()
            if edge_pair is None:
                car.has_arrived = True
                continue

            source, target = edge_pair
            if not graph.has_edge(source, target, 0):
                continue

            cells = self.edge_cells[(source, target, 0)]
            last_cell = len(cells) - 1

            # Only cars sitting in the final cell are transfer candidates.
            if cells[last_cell] < 0:
                continue

            # Peek ahead without committing. current_node_index is the node the
            # car departed from, so peek_index is the node it is arriving at.
            peek_index = car.current_node_index + 1

            # Arriving at the second-to-last node means the final segment is
            # done and the trip is complete.
            if peek_index >= len(car.planned_route_nodes) - 1:
                cells[last_cell] = -1
                car.has_arrived = True
                car.current_node_index += 1
                continue

            next_source = car.planned_route_nodes[peek_index]
            next_target = car.planned_route_nodes[peek_index + 1]

            if not graph.has_edge(next_source, next_target, 0):
                # No usable segment: skip the node and move the pointer on.
                car.current_node_index += 1
                continue

            next_cells = self.edge_cells[(next_source, next_target, 0)]
            if next_cells[0] == -1:
                cells[last_cell] = -1
                next_cells[0] = 0
                car.current_cell_position = 0
                car.current_speed_cells = 0
                car.current_node_index += 1
            # Otherwise the next segment is blocked and the car genuinely waits.

    def _record_edge_car_counts(self) -> None:
        """Append this step's per-segment car count to the history."""
        for edge, cells in self.edge_cells.items():
            self.edge_car_count_history[edge].append(int(np.sum(cells >= 0)))

    def step(self, record_statistics: bool = False) -> None:
        """
        Advance one time step.

        NS rules are applied to every segment in parallel, then cars are
        transferred across segment boundaries, then counts are optionally
        recorded.
        """
        for edge, cells in self.edge_cells.items():
            self.edge_cells[edge] = self._apply_ns_rules_to_edge(cells)

        self._advance_cars_along_routes()

        if record_statistics:
            self._record_edge_car_counts()

    def run(self) -> "TrafficSimulation":
        """
        Run the full trial.

        The first ``burn_in_steps`` steps run unrecorded so the simulation can
        settle into a quasi-steady state; everything after that is recorded.
        Returns ``self`` so calls can be chained.
        """
        for step_index in range(self.num_steps):
            self.step(record_statistics=step_index >= self.burn_in_steps)
        return self

    # -- Statistics ---------------------------------------------------------

    def get_mean_congestion_per_edge(self) -> dict:
        """Mean car count per segment across all recorded steps."""
        return {
            edge: float(np.mean(counts)) if counts else 0.0
            for edge, counts in self.edge_car_count_history.items()
        }

    def get_peak_congestion_per_edge(self) -> dict:
        """Maximum car count observed per segment across all recorded steps."""
        return {
            edge: max(counts) if counts else 0
            for edge, counts in self.edge_car_count_history.items()
        }

    def get_mean_network_speed(self) -> float:
        """
        Mean speed across every occupied cell in the network.

        This is the network-level analogue of the flow measure in the
        single-road fundamental diagram: high means free flow, falling means
        congestion is setting in.
        """
        speeds = []
        for cells in self.edge_cells.values():
            speeds.extend(cells[cells >= 0].tolist())
        return float(np.mean(speeds)) if speeds else 0.0

    def get_arrival_fraction(self) -> float:
        """Fraction of cars that reached their destination before the end."""
        if not self.cars:
            return 0.0
        return sum(car.has_arrived for car in self.cars) / len(self.cars)
