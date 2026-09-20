"""Congestion maps."""

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import osmnx as ox

from gridlock.network import RoadNetwork


class TrafficVisualizer:
    """
    Draws the street network with each segment coloured by congestion.

    The key design decision is *what* gets coloured. Raw car counts are
    misleading: a 20 m side street with two cars on it is nearly saturated
    while a 400 m avenue with two cars on it is empty, yet both would get the
    same colour. Segments are therefore coloured by local density
    (``cars / num_cells``), which puts short and long roads on a comparable
    footing.
    """

    #: Name registered for the custom gradient.
    DENSITY_COLORMAP = "gridlock_heat"

    #: Grey (empty) -> gold -> orange -> dark red (jammed), so the map reads
    #: like a traffic light at a glance.
    COLORMAP_STOPS = ["#B0B0B0", "#FFD700", "#FF4500", "#8B0000"]

    MIN_EDGE_WIDTH = 0.5
    MAX_EXTRA_WIDTH = 3.5

    @classmethod
    def colormap(cls):
        """The grey-to-red density gradient."""
        return mcolors.LinearSegmentedColormap.from_list(
            cls.DENSITY_COLORMAP, cls.COLORMAP_STOPS
        )

    @staticmethod
    def _compute_edge_densities_from_counts(
        edge_car_counts: dict, road_network: RoadNetwork
    ) -> dict:
        """
        Convert raw per-segment car counts into fractional occupancy.

        Density is ``cars on segment / cells on segment``. Values above 1 are
        possible in principle when cars stack in the final cell under the
        finite-segment boundary condition, but are rare.
        """
        graph = road_network.graph
        densities = {}
        for (source, target, key), count in edge_car_counts.items():
            if graph.has_edge(source, target, key):
                num_cells = graph[source][target][key].get("num_cells", 1)
                densities[(source, target, key)] = count / num_cells
            else:
                densities[(source, target, key)] = 0.0
        return densities

    @classmethod
    def plot_congestion_map(
        cls,
        road_network: RoadNetwork,
        edge_car_counts: dict,
        title: str = "Traffic Congestion",
        subtitle: str = "",
        figsize: tuple = (9, 9),
    ):
        """
        Draw the network with segments coloured and widened by car density.

        Parameters
        ----------
        road_network : RoadNetwork
            The network to draw.
        edge_car_counts : dict
            Maps ``(source, target, key)`` to a car count. This can be a single
            snapshot or an average across steps and trials.
        title, subtitle : str
            Headings above the figure.
        figsize : tuple
            Figure size in inches.
        """
        graph = road_network.graph
        densities = cls._compute_edge_densities_from_counts(
            edge_car_counts, road_network
        )

        # Guard against an all-empty network, which would divide by zero when
        # normalising colours.
        max_density = max(densities.values()) if densities else 1.0
        if max_density == 0:
            max_density = 1.0

        colormap = cls.colormap()

        # Build colour and width lists in graph.edges() order.
        edge_colors, edge_widths = [], []
        for source, target, key in graph.edges(keys=True):
            normalised = densities.get((source, target, key), 0.0) / max_density
            edge_colors.append(colormap(normalised))
            edge_widths.append(cls.MIN_EDGE_WIDTH + cls.MAX_EXTRA_WIDTH * normalised)

        fig, ax = ox.plot_graph(
            graph,
            figsize=figsize,
            node_size=0,
            edge_color=edge_colors,
            edge_linewidth=edge_widths,
            bgcolor="#FFFFFF",
            show=False,
            close=False,
        )

        # A colorbar in real units, so the scale is interpretable rather than
        # decorative.
        mappable = plt.cm.ScalarMappable(
            cmap=colormap, norm=mcolors.Normalize(vmin=0, vmax=max_density)
        )
        mappable.set_array([])
        colorbar = fig.colorbar(mappable, ax=ax, fraction=0.03, pad=0.02)
        colorbar.set_label("Car density (cars / cell)", fontsize=10, color="black")
        colorbar.ax.yaxis.set_tick_params(color="black")
        plt.setp(colorbar.ax.yaxis.get_ticklabels(), color="black")

        if subtitle:
            ax.set_title(f"{title}\n{subtitle}", fontsize=12, color="black", pad=10)
        else:
            ax.set_title(title, fontsize=13, color="black", pad=10)

        plt.tight_layout()
        plt.show()
        return fig, ax
