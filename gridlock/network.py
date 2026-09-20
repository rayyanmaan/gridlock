"""Loading and preparing road networks from OpenStreetMap."""

import matplotlib.pyplot as plt
import osmnx as ox

from gridlock.config import CELL_SIZE_METERS, NETWORK_RADIUS_METERS


class RoadNetwork:
    """
    A directed drivable street network, discretised into car-length cells.

    Nodes are intersections and edges are road segments. Every edge is
    annotated with two derived attributes:

    ``travel_time_seconds``
        Segment length divided by the posted speed limit. Used as the
        shortest-path weight, so cars minimise time rather than distance.
    ``num_cells``
        How many car-length cells fit on the segment. This is what turns each
        road into a one-dimensional cellular automaton lane.

    Attributes
    ----------
    address : str
        The address the network was centred on.
    radius_meters : int
        Radius around that address that was downloaded.
    graph : networkx.MultiDiGraph
        The annotated street graph.
    """

    #: Fallback speed in km/h for edges where OSM has no usable speed data.
    DEFAULT_SPEED_KMH = 30

    def __init__(
        self,
        address: str,
        radius_meters: int = NETWORK_RADIUS_METERS,
        verbose: bool = True,
    ):
        """
        Download a drivable street network and compute its edge attributes.

        Parameters
        ----------
        address : str
            Human-readable address to centre the download on.
        radius_meters : int
            Radius in metres around that address to include.
        verbose : bool
            Print progress and the resulting network size.
        """
        self.address = address
        self.radius_meters = radius_meters

        if verbose:
            print(f"Loading road network for: {address}")

        self.graph = ox.graph_from_address(
            address, dist=radius_meters, network_type="drive"
        )

        self._compute_edge_travel_times()
        self._compute_edge_cell_counts()

        if verbose:
            print(
                f"Network loaded: {self.graph.number_of_nodes()} intersections, "
                f"{self.graph.number_of_edges()} road segments."
            )

    # -- Edge attributes ----------------------------------------------------

    def _parse_speed_kmh(self, raw_speed_value) -> float:
        """
        Coerce an OSM ``maxspeed`` value into a float in km/h.

        OSM stores speed limits inconsistently: as a number, as the string
        ``"50"``, as ``"30 mph"``, or as a list when several limits apply to
        one segment. All four cases are handled, with a fallback to
        ``DEFAULT_SPEED_KMH`` when nothing usable can be parsed.
        """
        # When several limits apply to one segment, take the first.
        if isinstance(raw_speed_value, list):
            raw_speed_value = raw_speed_value[0]

        try:
            # Strip any unit suffix before converting.
            speed_kmh = float(str(raw_speed_value).split()[0])
        except (ValueError, AttributeError, IndexError):
            speed_kmh = self.DEFAULT_SPEED_KMH

        # Guard against zero or negative values present in the raw data.
        if speed_kmh <= 0:
            speed_kmh = self.DEFAULT_SPEED_KMH

        return speed_kmh

    def _compute_edge_travel_times(self) -> None:
        """Annotate every edge with ``travel_time_seconds``."""
        for source, target, key, data in self.graph.edges(data=True, keys=True):
            length_meters = data.get("length", 0)
            speed_kmh = self._parse_speed_kmh(
                data.get("maxspeed", self.DEFAULT_SPEED_KMH)
            )
            speed_mps = speed_kmh * 1000 / 3600
            self.graph[source][target][key]["travel_time_seconds"] = (
                length_meters / speed_mps if speed_mps > 0 else float("inf")
            )

    def _compute_edge_cell_counts(self) -> None:
        """Annotate every edge with ``num_cells`` (minimum 1)."""
        for source, target, key, data in self.graph.edges(data=True, keys=True):
            length_meters = data.get("length", CELL_SIZE_METERS)
            self.graph[source][target][key]["num_cells"] = max(
                1, int(length_meters / CELL_SIZE_METERS)
            )

    # -- Convenience --------------------------------------------------------

    @property
    def total_cells(self) -> int:
        """Total number of cells across every segment in the network."""
        return sum(
            data.get("num_cells", 1) for _, _, data in self.graph.edges(data=True)
        )

    def plot_base_network(self, title: str = "Road Network"):
        """Draw the bare street network with no traffic data on it."""
        fig, ax = ox.plot_graph(
            self.graph,
            figsize=(8, 8),
            node_size=5,
            edge_linewidth=0.8,
            bgcolor="#FFFFFF",
            node_color="black",
            edge_color="#555555",
            show=False,
            close=False,
        )
        ax.set_title(title, fontsize=13)
        plt.tight_layout()
        plt.show()
        return fig, ax

    def __repr__(self) -> str:
        return (
            f"RoadNetwork(address={self.address!r}, "
            f"radius_meters={self.radius_meters}, "
            f"nodes={self.graph.number_of_nodes()}, "
            f"edges={self.graph.number_of_edges()})"
        )
