"""Compare homogenized and fine-model static displacement fields.

The script reads CSV files produced by ``extract_odb_displacement.py``.  Sample
locations are selected from the homogenized mesh by deterministic spatial
binning.  Fine-model displacements at those locations are reconstructed with a
weighted local affine fit (moving least squares, MLS), which both interpolates
through void regions and coarse-grains cell-scale fluctuations.

Outputs include traceable sampled fields, reviewer-facing error metrics,
interpolation-neighbour sensitivity tables, and publication-ready SVG/PDF/TIFF
figures.  Abaqus is not imported or launched by this script.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


CSV_FILES = {
    "shell": ("shell_homo.csv", "shell_fine.csv"),
    "solid": ("solid_homo.csv", "solid_fine.csv"),
}
COORD_COLUMNS = ["x", "y", "z"]
DISPLACEMENT_COLUMNS = ["u1", "u2", "u3"]
REQUIRED_COLUMNS = {"node_label", *COORD_COLUMNS, *DISPLACEMENT_COLUMNS}

COLORS = {
    "fine": "#0072B2",
    "homo": "#D55E00",
    "identity": "#333333",
}

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size": 9.0,
        "axes.labelsize": 9.5,
        "axes.titlesize": 10.0,
        "axes.linewidth": 0.8,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.5,
        "legend.frameon": False,
        "lines.linewidth": 1.1,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "savefig.facecolor": "white",
    }
)


def load_field(path: Path) -> pd.DataFrame:
    """Load and validate one displacement-field CSV."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}. First run extract_odb_displacement.py with Abaqus."
        )
    table = pd.read_csv(path)
    missing = REQUIRED_COLUMNS.difference(table.columns)
    if missing:
        raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")
    numeric_columns = ["node_label", *COORD_COLUMNS, *DISPLACEMENT_COLUMNS]
    table[numeric_columns] = table[numeric_columns].apply(
        pd.to_numeric, errors="raise"
    )
    values = table[[*COORD_COLUMNS, *DISPLACEMENT_COLUMNS]].to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError(f"{path.name} contains NaN or infinite coordinates/U values")
    if table["node_label"].duplicated().any():
        raise ValueError(f"{path.name} contains duplicate node labels")
    return table


def coordinate_summary(coordinates: np.ndarray) -> Dict[str, np.ndarray]:
    lower = coordinates.min(axis=0)
    upper = coordinates.max(axis=0)
    span = upper - lower
    tolerance = max(float(span.max()), 1.0) * 1.0e-10
    return {
        "min": lower,
        "max": upper,
        "mid": 0.5 * (lower + upper),
        "span": span,
        "active": span > tolerance,
    }


def sample_homogenized_nodes(
    coordinates: np.ndarray, max_points: int
) -> np.ndarray:
    """Select nearly uniform, deterministic points from the homo mesh.

    Nodes are grouped into an equal-bin grid over the geometrically active
    dimensions.  The node closest to each occupied bin centre is retained.
    """
    if max_points < 8:
        raise ValueError("max_points must be at least 8")
    count = coordinates.shape[0]
    if count <= max_points:
        return np.arange(count, dtype=int)

    summary = coordinate_summary(coordinates)
    active = np.flatnonzero(summary["active"])
    if active.size == 0:
        return np.array([0], dtype=int)
    bins_per_axis = max(1, int(math.floor(max_points ** (1.0 / active.size))))
    normalized = (
        coordinates[:, active] - summary["min"][active]
    ) / summary["span"][active]
    bin_index = np.minimum(
        (normalized * bins_per_axis).astype(int), bins_per_axis - 1
    )

    selected: Dict[Tuple[int, ...], Tuple[float, int]] = {}
    for index, (point, key_array) in enumerate(zip(normalized, bin_index)):
        key = tuple(int(value) for value in key_array)
        centre = (key_array.astype(float) + 0.5) / bins_per_axis
        distance2 = float(np.dot(point - centre, point - centre))
        previous = selected.get(key)
        if previous is None or distance2 < previous[0]:
            selected[key] = (distance2, index)
    return np.asarray(sorted(value[1] for value in selected.values()), dtype=int)


def map_homo_to_fine_envelope(
    homo_query: np.ndarray, homo_all: np.ndarray, fine_all: np.ndarray
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Align translations (and report any scale mismatch) by bounding boxes."""
    homo = coordinate_summary(homo_all)
    fine = coordinate_summary(fine_all)
    query = np.empty_like(homo_query, dtype=float)
    scale = np.ones(3, dtype=float)
    for direction in range(3):
        if homo["active"][direction] and fine["active"][direction]:
            scale[direction] = fine["span"][direction] / homo["span"][direction]
            query[:, direction] = fine["min"][direction] + (
                homo_query[:, direction] - homo["min"][direction]
            ) * scale[direction]
        elif not homo["active"][direction]:
            # A shell reference surface maps to the middle of the fine envelope.
            query[:, direction] = fine["mid"][direction]
        else:
            raise ValueError(
                "Homo model varies along coordinate %d but fine model does not"
                % (direction + 1)
            )

    active = homo["active"] & fine["active"]
    mismatch = np.zeros(3, dtype=float)
    mismatch[active] = np.abs(scale[active] - 1.0)
    metadata = {
        "homo_min": homo["min"].tolist(),
        "homo_max": homo["max"].tolist(),
        "fine_min": fine["min"].tolist(),
        "fine_max": fine["max"].tolist(),
        "fine_per_homo_scale": scale.tolist(),
        "maximum_span_mismatch_fraction": float(mismatch.max()),
    }
    return query, metadata


def validate_alignment(
    alignment: Dict[str, object], allow_envelope_scaling: bool
) -> None:
    """Reject hidden geometry scaling unless the user explicitly allows it."""
    mismatch = float(alignment["maximum_span_mismatch_fraction"])
    if mismatch <= 1.0e-5:
        return
    message = (
        "Homo/fine envelope spans differ by up to %.3f%%. This is a geometry "
        "mismatch, not a coordinate-origin shift." % (100.0 * mismatch)
    )
    if not allow_envelope_scaling:
        raise ValueError(
            message
            + " Fix the models or rerun with --allow-envelope-scaling only when "
            "the scale transformation is intentional."
        )
    print("WARNING: " + message + " Explicit envelope scaling is enabled.")


def _scaled_coordinates(
    fine_coordinates: np.ndarray, query_coordinates: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    summary = coordinate_summary(fine_coordinates)
    scale = summary["span"].copy()
    scale[~summary["active"]] = 1.0
    fine_scaled = (fine_coordinates - summary["min"]) / scale
    query_scaled = (query_coordinates - summary["min"]) / scale
    active = np.flatnonzero(summary["active"])
    return fine_scaled, query_scaled, active


def moving_least_squares(
    fine_coordinates: np.ndarray,
    fine_displacements: np.ndarray,
    query_coordinates: np.ndarray,
    neighbours: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reconstruct U with a Gaussian-weighted local affine regression."""
    if neighbours < 8:
        raise ValueError("At least 8 MLS neighbours are required")
    if fine_coordinates.shape[0] < 2:
        raise ValueError("Fine field must contain at least two nodes")

    fine_scaled, query_scaled, active = _scaled_coordinates(
        fine_coordinates, query_coordinates
    )
    tree = cKDTree(fine_scaled)
    k = min(int(neighbours), fine_scaled.shape[0])
    distances, indices = tree.query(query_scaled, k=k, workers=-1)
    if k == 1:
        distances = distances[:, None]
        indices = indices[:, None]

    predictions = np.empty((query_coordinates.shape[0], 3), dtype=float)
    condition_numbers = np.empty(query_coordinates.shape[0], dtype=float)
    support_radii = distances[:, -1].copy()

    for row in range(query_coordinates.shape[0]):
        local_coordinates = fine_scaled[indices[row]][:, active]
        centre = query_scaled[row, active]
        offsets = local_coordinates - centre
        design = np.column_stack((np.ones(k), offsets))

        radius = max(float(distances[row, -1]), 1.0e-14)
        weights = np.exp(-4.0 * (distances[row] / radius) ** 2)
        weights = np.maximum(weights, 1.0e-10)
        root_weight = np.sqrt(weights)[:, None]
        weighted_design = design * root_weight
        weighted_values = fine_displacements[indices[row]] * root_weight

        try:
            coefficients, _, rank, singular_values = np.linalg.lstsq(
                weighted_design, weighted_values, rcond=None
            )
            if rank < design.shape[1]:
                raise np.linalg.LinAlgError("rank-deficient MLS design")
            predictions[row] = coefficients[0]
            condition_numbers[row] = (
                float(singular_values[0] / singular_values[-1])
                if singular_values[-1] > 0.0
                else np.inf
            )
        except np.linalg.LinAlgError:
            inverse_distance = 1.0 / np.maximum(distances[row], 1.0e-12) ** 2
            inverse_distance /= inverse_distance.sum()
            predictions[row] = np.sum(
                fine_displacements[indices[row]] * inverse_distance[:, None],
                axis=0,
            )
            condition_numbers[row] = np.inf
    return predictions, condition_numbers, support_radii


def _safe_relative_l2(predicted: np.ndarray, reference: np.ndarray) -> float:
    denominator = float(np.linalg.norm(reference))
    if denominator <= np.finfo(float).eps:
        return float("nan")
    return float(np.linalg.norm(predicted - reference) / denominator)


def _safe_r2(predicted: np.ndarray, reference: np.ndarray) -> float:
    centred = reference - reference.mean()
    denominator = float(np.dot(centred, centred))
    if denominator <= np.finfo(float).eps:
        return float("nan")
    return float(1.0 - np.dot(predicted - reference, predicted - reference) / denominator)


def field_metrics(homo_u: np.ndarray, fine_u: np.ndarray) -> Dict[str, float]:
    """Compute reviewer-facing displacement-field agreement measures."""
    difference = homo_u - fine_u
    homo_magnitude = np.linalg.norm(homo_u, axis=1)
    fine_magnitude = np.linalg.norm(fine_u, axis=1)
    difference_magnitude = np.linalg.norm(difference, axis=1)
    fine_peak = max(float(fine_magnitude.max()), np.finfo(float).eps)

    flattened_homo = homo_u.ravel()
    flattened_fine = fine_u.ravel()
    norm_product = float(
        np.linalg.norm(flattened_homo) * np.linalg.norm(flattened_fine)
    )
    cosine = (
        float(np.dot(flattened_homo, flattened_fine) / norm_product)
        if norm_product > np.finfo(float).eps
        else float("nan")
    )

    metrics: Dict[str, float] = {
        "sample_count": float(homo_u.shape[0]),
        "vector_relative_L2": _safe_relative_l2(homo_u, fine_u),
        "magnitude_relative_L2": _safe_relative_l2(
            homo_magnitude, fine_magnitude
        ),
        "field_cosine_similarity": cosine,
        "displacement_field_MAC": cosine * cosine,
        "fine_peak_U": float(fine_magnitude.max()),
        "homo_peak_U": float(homo_magnitude.max()),
        "peak_U_signed_relative_error": float(
            homo_magnitude.max() / fine_peak - 1.0
        ),
        "pointwise_error_over_fine_peak_mean": float(
            np.mean(difference_magnitude / fine_peak)
        ),
        "pointwise_error_over_fine_peak_p95": float(
            np.percentile(difference_magnitude / fine_peak, 95.0)
        ),
        "pointwise_error_over_fine_peak_max": float(
            np.max(difference_magnitude / fine_peak)
        ),
    }
    for component in range(3):
        label = f"U{component + 1}"
        metrics[f"{label}_relative_L2"] = _safe_relative_l2(
            homo_u[:, component], fine_u[:, component]
        )
        metrics[f"{label}_R2"] = _safe_r2(
            homo_u[:, component], fine_u[:, component]
        )
    return metrics


def choose_plot_slice(coordinates: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int], int | None]:
    """Return a readable 2D section of planar or volumetric sampled points."""
    summary = coordinate_summary(coordinates)
    active = np.flatnonzero(summary["active"])
    if active.size < 2:
        raise ValueError("At least two active coordinate directions are required")
    if active.size == 2:
        return np.ones(coordinates.shape[0], dtype=bool), (int(active[0]), int(active[1])), None

    # For a volume, display a central section normal to its shortest direction.
    slice_direction = int(active[np.argmin(summary["span"][active])])
    plot_directions = tuple(int(value) for value in active if value != slice_direction)
    distance = np.abs(coordinates[:, slice_direction] - summary["mid"][slice_direction])
    threshold = np.percentile(distance, 25.0)
    mask = distance <= threshold + max(summary["span"].max(), 1.0) * 1.0e-12
    if mask.sum() < min(50, coordinates.shape[0]):
        order = np.argsort(distance)
        mask = np.zeros(coordinates.shape[0], dtype=bool)
        mask[order[: min(max(50, coordinates.shape[0] // 4), coordinates.shape[0])]] = True
    return mask, (plot_directions[0], plot_directions[1]), slice_direction


def _style_field_axis(ax: plt.Axes, directions: Tuple[int, int]) -> None:
    labels = ["x", "y", "z"]
    ax.set_xlabel(labels[directions[0]])
    ax.set_ylabel(labels[directions[1]])
    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(pad=1.5)


def _save_individual_figure(fig: plt.Figure, output_stem: Path) -> None:
    """Save one editable/vector and one high-resolution raster panel."""
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    # Keep the declared physical canvas size identical across separately saved
    # panels; bbox_inches="tight" would make later journal-layout alignment harder.
    fig.savefig(output_stem.with_suffix(".svg"))
    fig.savefig(output_stem.with_suffix(".tiff"), dpi=600)
    plt.close(fig)


def make_validation_figures(
    case_name: str,
    sampled: pd.DataFrame,
    metrics: Dict[str, float],
    output_stem: Path,
) -> None:
    """Create four independent half-column panels for later 2x2 assembly."""
    coordinates = sampled[COORD_COLUMNS].to_numpy(float)
    mask, directions, slice_direction = choose_plot_slice(coordinates)
    x = coordinates[mask, directions[0]]
    y = coordinates[mask, directions[1]]
    fine_magnitude = sampled.loc[mask, "fine_umag"].to_numpy(float)
    homo_magnitude = sampled.loc[mask, "homo_umag"].to_numpy(float)
    normalized_error = sampled.loc[
        mask, "pointwise_error_over_fine_peak"
    ].to_numpy(float)

    magnitude_max = max(
        float(sampled["fine_umag"].max()), float(sampled["homo_umag"].max())
    )
    magnitude_norm = Normalize(vmin=0.0, vmax=max(magnitude_max, 1.0e-16))
    error_limit = max(float(np.percentile(normalized_error, 99.0)), 1.0e-12)
    error_norm = Normalize(vmin=0.0, vmax=error_limit)

    panel_width_in = 86.0 / 25.4
    field_height_in = 67.0 / 25.4
    parity_height_in = 76.0 / 25.4
    marker_size = max(7.0, min(22.0, 3500.0 / max(mask.sum(), 1)))

    field_panels = (
        (
            fine_magnitude,
            "Fine model (MLS)",
            "viridis",
            magnitude_norm,
            r"Displacement magnitude, $|\mathbf{U}|$",
            "fine_field",
        ),
        (
            homo_magnitude,
            "Homogenized model",
            "viridis",
            magnitude_norm,
            r"Displacement magnitude, $|\mathbf{U}|$",
            "homo_field",
        ),
        (
            normalized_error,
            r"Pointwise displacement error",
            "magma",
            error_norm,
            r"$|\Delta\mathbf{U}|/U_{\max}^{\mathrm{fine}}$",
            "error_field",
        ),
    )
    for values, title, cmap, norm, colorbar_label, suffix in field_panels:
        fig, ax = plt.subplots(
            1, 1, figsize=(panel_width_in, field_height_in), constrained_layout=True
        )
        handle = ax.scatter(
            x,
            y,
            c=values,
            cmap=cmap,
            norm=norm,
            s=marker_size,
            marker="s",
            linewidths=0.0,
            rasterized=True,
        )
        ax.set_title(title, pad=3.0)
        _style_field_axis(ax, directions)
        colorbar = fig.colorbar(handle, ax=ax, fraction=0.052, pad=0.035)
        colorbar.set_label(colorbar_label, fontsize=9.0)
        colorbar.ax.tick_params(labelsize=8.0)
        if slice_direction is not None:
            labels = ["x", "y", "z"]
            ax.text(
                0.02,
                0.02,
                "Central %s-section" % labels[slice_direction],
                transform=ax.transAxes,
                ha="left",
                va="bottom",
                fontsize=8.0,
                color="#333333",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78},
            )
        _save_individual_figure(fig, Path(str(output_stem) + "_" + suffix))

    fig, parity = plt.subplots(
        1, 1, figsize=(panel_width_in, parity_height_in), constrained_layout=True
    )
    parity.scatter(
        sampled["fine_umag"],
        sampled["homo_umag"],
        s=9.0,
        color=COLORS["homo"],
        alpha=0.62,
        edgecolors="none",
        rasterized=True,
    )
    upper = max(
        float(sampled["fine_umag"].max()), float(sampled["homo_umag"].max())
    )
    parity.plot([0.0, upper], [0.0, upper], color=COLORS["identity"], lw=0.9, ls="--")
    parity.set_xlim(0.0, upper * 1.04)
    parity.set_ylim(0.0, upper * 1.04)
    parity.set_aspect("equal", adjustable="box")
    parity.set_xlabel(r"Fine model $|\mathbf{U}|$")
    parity.set_ylabel(r"Homogenized model $|\mathbf{U}|$")
    parity.set_title("Pointwise field agreement", pad=3.0)
    parity.text(
        0.04,
        0.96,
        "Rel. $L_2$ = %.2f%%\nMAC = %.4f\nPeak error = %+.2f%%"
        % (
            100.0 * metrics["vector_relative_L2"],
            metrics["displacement_field_MAC"],
            100.0 * metrics["peak_U_signed_relative_error"],
        ),
        transform=parity.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
    )
    _save_individual_figure(fig, Path(str(output_stem) + "_parity"))


def metrics_to_table(metrics: Dict[str, float]) -> pd.DataFrame:
    rows = []
    for metric, value in metrics.items():
        if metric == "sample_count":
            display = f"{int(round(value))}"
        elif "similarity" in metric or "MAC" in metric or metric.endswith("R2"):
            display = f"{value:.6g}"
        elif "peak_U" in metric and "error" not in metric:
            display = f"{value:.12e}"
        else:
            display = f"{100.0 * value:.4f}%" if np.isfinite(value) else "NaN"
        rows.append({"metric": metric, "value": value, "display": display})
    return pd.DataFrame(rows)


def analyse_case(
    case_name: str,
    homo: pd.DataFrame,
    fine: pd.DataFrame,
    output_dir: Path,
    max_points: int,
    neighbours: int,
    sensitivity_neighbours: Sequence[int],
    fine_scale: float,
    allow_envelope_scaling: bool,
) -> Tuple[Dict[str, float], Dict[str, object]]:
    homo_coordinates = homo[COORD_COLUMNS].to_numpy(float)
    fine_coordinates = fine[COORD_COLUMNS].to_numpy(float)
    homo_u_all = homo[DISPLACEMENT_COLUMNS].to_numpy(float)
    fine_u_all = fine[DISPLACEMENT_COLUMNS].to_numpy(float)

    sample_indices = sample_homogenized_nodes(homo_coordinates, max_points)
    sample_coordinates = homo_coordinates[sample_indices]
    query_coordinates, alignment = map_homo_to_fine_envelope(
        sample_coordinates, homo_coordinates, fine_coordinates
    )
    validate_alignment(alignment, allow_envelope_scaling)

    fine_predictions_raw, conditions, support_radii = moving_least_squares(
        fine_coordinates, fine_u_all, query_coordinates, neighbours
    )
    fine_predictions = fine_predictions_raw * fine_scale
    homo_u = homo_u_all[sample_indices]
    metrics = field_metrics(homo_u, fine_predictions)

    sampled = pd.DataFrame(
        {
            "homo_node_label": homo.iloc[sample_indices]["node_label"].to_numpy(int),
            "x": sample_coordinates[:, 0],
            "y": sample_coordinates[:, 1],
            "z": sample_coordinates[:, 2],
            "fine_query_x": query_coordinates[:, 0],
            "fine_query_y": query_coordinates[:, 1],
            "fine_query_z": query_coordinates[:, 2],
            "homo_u1": homo_u[:, 0],
            "homo_u2": homo_u[:, 1],
            "homo_u3": homo_u[:, 2],
            "fine_u1": fine_predictions[:, 0],
            "fine_u2": fine_predictions[:, 1],
            "fine_u3": fine_predictions[:, 2],
            "homo_umag": np.linalg.norm(homo_u, axis=1),
            "fine_umag": np.linalg.norm(fine_predictions, axis=1),
            "absolute_vector_error": np.linalg.norm(homo_u - fine_predictions, axis=1),
            "mls_condition_number": conditions,
            "mls_support_radius_normalized": support_radii,
        }
    )
    fine_peak = max(float(sampled["fine_umag"].max()), np.finfo(float).eps)
    sampled["pointwise_error_over_fine_peak"] = (
        sampled["absolute_vector_error"] / fine_peak
    )
    sampled.to_csv(output_dir / f"{case_name}_sampled_displacement_field.csv", index=False)
    metrics_to_table(metrics).to_csv(
        output_dir / f"{case_name}_field_metrics.csv", index=False
    )

    sensitivity_rows: List[Dict[str, float]] = []
    for candidate in sorted(set([neighbours, *sensitivity_neighbours])):
        candidate_prediction, _, _ = moving_least_squares(
            fine_coordinates, fine_u_all, query_coordinates, candidate
        )
        candidate_metrics = field_metrics(homo_u, candidate_prediction * fine_scale)
        sensitivity_rows.append(
            {
                "neighbours": candidate,
                "vector_relative_L2": candidate_metrics["vector_relative_L2"],
                "magnitude_relative_L2": candidate_metrics["magnitude_relative_L2"],
                "displacement_field_MAC": candidate_metrics["displacement_field_MAC"],
                "peak_U_signed_relative_error": candidate_metrics[
                    "peak_U_signed_relative_error"
                ],
            }
        )
    pd.DataFrame(sensitivity_rows).to_csv(
        output_dir / f"{case_name}_interpolation_sensitivity.csv", index=False
    )

    metadata: Dict[str, object] = {
        "case": case_name,
        "homo_node_count": int(homo.shape[0]),
        "fine_node_count": int(fine.shape[0]),
        "sample_count": int(sample_indices.size),
        "max_points_requested": int(max_points),
        "mls_neighbours": int(neighbours),
        "fine_displacement_scale": float(fine_scale),
        "allow_envelope_scaling": bool(allow_envelope_scaling),
        "alignment": alignment,
        "mls_condition_number_median": float(np.median(conditions)),
        "mls_condition_number_max": float(np.max(conditions)),
        "mls_support_radius_normalized_median": float(np.median(support_radii)),
    }
    with (output_dir / f"{case_name}_analysis_metadata.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(metadata, stream, indent=2, sort_keys=True)

    make_validation_figures(
        case_name,
        sampled,
        metrics,
        output_dir / f"{case_name}_displacement_field_validation",
    )
    return metrics, metadata


def _parse_integer_list(text: str) -> List[int]:
    values = [int(token.strip()) for token in text.split(",") if token.strip()]
    if not values or any(value < 8 for value in values):
        raise argparse.ArgumentTypeError(
            "Provide comma-separated neighbour counts, all at least 8"
        )
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare fine and homogenized static displacement fields."
    )
    parser.add_argument(
        "--data-dir", type=Path, required=True,
        help="Directory containing the four CSV files exported from the ODBs",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        help="Output directory (default: <data-dir>/field_validation_outputs)",
    )
    parser.add_argument(
        "--case", choices=("shell", "solid", "all"), default="all"
    )
    parser.add_argument(
        "--max-points", type=int, default=700,
        help="Maximum spatial sampling target for homo nodes (default: 700)",
    )
    parser.add_argument(
        "--neighbours", type=int, default=64,
        help="MLS fine-node neighbourhood used in the main figures (default: 64)",
    )
    parser.add_argument(
        "--sensitivity-neighbours", type=_parse_integer_list,
        default=[32, 64, 128], help="Comma-separated MLS sensitivity values",
    )
    parser.add_argument(
        "--solid-fine-scale", type=float, default=1.0,
        help=(
            "Optional multiplier for solid fine-model U. For the supplied ODBs, "
            "use 0.2564102564102564 to correct the -0.39 versus -0.1 load mismatch."
        ),
    )
    parser.add_argument(
        "--allow-envelope-scaling", action="store_true",
        help=(
            "Allow coordinate scaling when homo/fine envelope sizes differ. "
            "By default a size mismatch is treated as a model error."
        ),
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    data_dir = args.data_dir.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else data_dir / "field_validation_outputs"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_cases = ["shell", "solid"] if args.case == "all" else [args.case]
    summary_rows: List[Dict[str, object]] = []
    for case in selected_cases:
        homo_name, fine_name = CSV_FILES[case]
        homo = load_field(data_dir / homo_name)
        fine = load_field(data_dir / fine_name)
        fine_scale = args.solid_fine_scale if case == "solid" else 1.0
        metrics, metadata = analyse_case(
            case,
            homo,
            fine,
            output_dir,
            max_points=args.max_points,
            neighbours=args.neighbours,
            sensitivity_neighbours=args.sensitivity_neighbours,
            fine_scale=fine_scale,
            allow_envelope_scaling=args.allow_envelope_scaling,
        )
        summary_rows.append(
            {
                "case": case,
                "sample_count": metadata["sample_count"],
                "fine_displacement_scale": fine_scale,
                "vector_relative_L2": metrics["vector_relative_L2"],
                "magnitude_relative_L2": metrics["magnitude_relative_L2"],
                "displacement_field_MAC": metrics["displacement_field_MAC"],
                "peak_U_signed_relative_error": metrics[
                    "peak_U_signed_relative_error"
                ],
                "pointwise_error_over_fine_peak_p95": metrics[
                    "pointwise_error_over_fine_peak_p95"
                ],
            }
        )
        print(
            "%s: relative L2 = %.3f%%, MAC = %.6f, peak error = %+.3f%%"
            % (
                case,
                100.0 * metrics["vector_relative_L2"],
                metrics["displacement_field_MAC"],
                100.0 * metrics["peak_U_signed_relative_error"],
            )
        )

    pd.DataFrame(summary_rows).to_csv(
        output_dir / "static_field_validation_summary.csv", index=False
    )
    print(f"Wrote field-validation package to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
