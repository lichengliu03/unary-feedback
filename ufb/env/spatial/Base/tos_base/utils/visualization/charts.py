import base64
from io import BytesIO
from typing import Dict, List, Optional
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def _fig_to_data_uri(fig) -> str:
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    data = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    plt.close(fig)
    return f"data:image/png;base64,{data}"


def create_infogain_plot(infogain_per_turn: List[float], title: str) -> str:
    fig, ax = plt.subplots(figsize=(8, 4))
    turns = list(range(1, len(infogain_per_turn) + 1))
    ax.plot(turns, infogain_per_turn, marker='o', linewidth=2, markersize=4)
    ax.set_xlabel('Turn')
    ax.set_ylabel('Average Information Gain')
    ax.set_title(f'Average Information Gain per Turn - {title}')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.5, len(infogain_per_turn) + 0.5)
    return _fig_to_data_uri(fig)


def create_cogmap_metrics_plot(
    series: Dict[str, List[Optional[float]]],
    title: str,
    include_dir: bool = True,
    include_facing: bool = True,
    include_pos: bool = True,
    include_overall: bool = True,
) -> Optional[str]:
    keys = [
        ('dir', include_dir, 'Direction'),
        ('facing', include_facing, 'Facing'),
        ('pos', include_pos, 'Position'),
        ('overall', include_overall, 'Overall'),
    ]

    any_data = any(
        include and isinstance(series.get(k), list) and any(isinstance(v, (int, float)) for v in series.get(k, []))
        for k, include, _ in keys
    )
    if not any_data:
        return None

    # Check if there's only one turn - if so, skip drawing
    max_turns = max(
        len(series.get(k, [])) for k, include, _ in keys
        if include and isinstance(series.get(k), list)
    ) if any_data else 0

    if max_turns <= 1:
        return None

    fig, ax = plt.subplots(figsize=(8, 4))
    turns = None
    for k, include, label in keys:
        if not include:
            continue
        vals = series.get(k, [])
        if not isinstance(vals, list) or len(vals) == 0:
            continue
        y = [np.nan if not isinstance(v, (int, float)) else float(v) for v in vals]
        if turns is None:
            turns = list(range(1, len(y) + 1))
        ax.plot(turns, y, marker='o', linewidth=2, markersize=3, label=label)

    if turns is None:
        plt.close(fig)
        return None

    ax.set_xlabel('Turn')
    ax.set_ylabel('Similarity')
    ax.set_title(f'Cognitive Map Similarity per Turn - {title}')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.5, len(turns) + 0.5)
    ax.set_ylim(0.0, 1.0)
    ax.legend()
    return _fig_to_data_uri(fig)


def create_cognitive_map_sample_plots(
    cogmap_update_data: Dict[str, List[Optional[float]]],
    cogmap_full_data: Dict[str, List[Optional[float]]],
    sample_name: str
) -> Dict[str, Optional[str]]:
    """
    Create 2 plots for a single sample: Global metrics for both update and full modes.

    Args:
        cogmap_update_data: Dict with 'dir', 'facing', 'pos', 'overall' metrics per turn for update mode
        cogmap_full_data: Similar structure for full cognitive map
        sample_name: Name of the sample for plot titles

    Returns:
        Dict with 2 keys: 'global_update', 'global_full'
        Each value is either a data URI string or None if no data available
    """
    results = {}

    # Only process global level now
    modes = [('update', cogmap_update_data), ('full', cogmap_full_data)]

    for mode_name, data in modes:
        key = f"global_{mode_name}"

        if not isinstance(data, dict):
            results[key] = None
            continue

        # Data should already be in the format expected by create_cogmap_metrics_plot
        # (direct dict with 'dir', 'facing', 'pos', 'overall' keys)
        title = f"{sample_name} - Global ({mode_name.title()})"
        plot_uri = create_cogmap_metrics_plot(data, title)
        results[key] = plot_uri

    return results



def create_correlation_scatter_plot(x_values: List[float], y_values: List[float],
                                   x_label: str, y_label: str, title: str,
                                   correlation_info: Dict = None) -> Optional[str]:
    """
    Create a scatter plot to display correlation between two variables.

    Args:
        x_values: X-axis data
        y_values: Y-axis data
        x_label: X-axis label
        y_label: Y-axis label
        title: Chart title
        correlation_info: Correlation information dictionary containing pearson_r, p_value, etc.

    Returns:
        Chart data URI string, or None if no data available
    """
    if not x_values or not y_values or len(x_values) != len(y_values):
        return None

    # Filter valid data points
    valid_pairs = []
    for x, y in zip(x_values, y_values):
        if (isinstance(x, (int, float)) and isinstance(y, (int, float)) and
            not np.isnan(x) and not np.isnan(y)):
            valid_pairs.append((float(x), float(y)))

    if len(valid_pairs) < 2:
        return None

    valid_x = [pair[0] for pair in valid_pairs]
    valid_y = [pair[1] for pair in valid_pairs]

    fig, ax = plt.subplots(figsize=(8, 6))

    # Draw scatter plot
    ax.scatter(valid_x, valid_y, alpha=0.6, s=50, edgecolors='black', linewidths=0.5)

    # Add trend line
    if len(valid_pairs) >= 2:
        try:
            z = np.polyfit(valid_x, valid_y, 1)
            p = np.poly1d(z)
            x_trend = np.linspace(min(valid_x), max(valid_x), 100)
            ax.plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2)
        except Exception:
            pass

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    # Add correlation information to the chart
    if correlation_info:
        pearson_r = correlation_info.get('pearson_r')
        p_value = correlation_info.get('p_value')
        significant = correlation_info.get('significant', False)

        if pearson_r is not None and p_value is not None:
            sig_text = "***" if significant else "n.s."
            text = f"r = {pearson_r:.3f}, p = {p_value:.3f} {sig_text}"
            ax.text(0.05, 0.95, text, transform=ax.transAxes,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                   verticalalignment='top', fontsize=10)

    return _fig_to_data_uri(fig)


def create_correlation_plot(x_values: List[float], y_values: List[float],
                          x_label: str, y_label: str, title: str,
                          correlation_info: Dict = None) -> Optional[str]:
    """
    Create a correlation scatter plot between two lists.

    Args:
        x_values: X-axis data list
        y_values: Y-axis data list
        x_label: X-axis label
        y_label: Y-axis label
        title: Chart title
        correlation_info: Optional correlation information dictionary, auto-calculated if not provided

    Returns:
        Chart data URI string, or None if no data available
    """
    if not x_values or not y_values:
        return None

    return create_correlation_scatter_plot(
        x_values, y_values, x_label, y_label, title, correlation_info
    )

