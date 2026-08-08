"""Publication-ready mesh-convergence analysis for the 10 x 10 x 10 cube.

The script reads ``CH-Cube_E*-3D.txt`` files in the same directory.  The
integer in each filename is the total number of equal-sized hexahedral
elements; its exact integer cube root is used as the per-edge mesh count.

The analytical reference is the isotropic three-dimensional constitutive
matrix for E = 1 and nu = 0.3.  Outputs include traceable CSV data and
editable SVG/PDF plus 600-dpi TIFF and 300-dpi PNG figures.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter, ScalarFormatter
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Inputs and analytical reference (consistent units; E is normalized to 1)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
TXT_GLOB = "CH-Cube_E*-3D.txt"
OUTPUT_DIR = ROOT / "cube_3d_convergence_outputs"

YOUNGS_MODULUS = 1.0
POISSON_RATIO = 0.3
CUBE_EDGE_LENGTH = 10.0

COMPONENTS = {
    "Normal": ("C11", "C22", "C33"),
    "Coupling": ("C12", "C13", "C23"),
    "Shear": ("C44", "C55", "C66"),
}

COLORS = {
    "C11": "#0072B2",
    "C22": "#D55E00",
    "C33": "#009E73",
    "C12": "#0072B2",
    "C13": "#D55E00",
    "C23": "#009E73",
    "C44": "#0072B2",
    "C55": "#D55E00",
    "C66": "#009E73",
}

MARKERS = {
    "C11": "o",
    "C22": "s",
    "C33": "^",
    "C12": "o",
    "C13": "s",
    "C23": "^",
    "C44": "o",
    "C55": "s",
    "C66": "^",
}


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size": 7.0,
        "axes.labelsize": 7.5,
        "axes.titlesize": 8.0,
        "axes.linewidth": 0.8,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 6.5,
        "legend.frameon": False,
        "lines.linewidth": 1.25,
        "lines.markersize": 4.0,
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


def theoretical_isotropic_matrix(
    youngs_modulus: float = YOUNGS_MODULUS,
    poisson_ratio: float = POISSON_RATIO,
) -> np.ndarray:
    """Return the isotropic 6 x 6 stiffness matrix in engineering notation."""
    if youngs_modulus <= 0.0:
        raise ValueError("Young's modulus must be positive")
    if not (-1.0 < poisson_ratio < 0.5):
        raise ValueError("Poisson's ratio must satisfy -1 < nu < 0.5")

    denominator = (1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio)
    normal = youngs_modulus * (1.0 - poisson_ratio) / denominator
    coupling = youngs_modulus * poisson_ratio / denominator
    shear = youngs_modulus / (2.0 * (1.0 + poisson_ratio))

    matrix = np.zeros((6, 6), dtype=float)
    matrix[:3, :3] = coupling
    np.fill_diagonal(matrix[:3, :3], normal)
    np.fill_diagonal(matrix[3:, 3:], shear)
    return matrix


def theoretical_components(matrix: np.ndarray) -> Dict[str, float]:
    """Extract the nine non-zero independent components used in the figures."""
    return {
        "C11": float(matrix[0, 0]),
        "C22": float(matrix[1, 1]),
        "C33": float(matrix[2, 2]),
        "C12": float(matrix[0, 1]),
        "C13": float(matrix[0, 2]),
        "C23": float(matrix[1, 2]),
        "C44": float(matrix[3, 3]),
        "C55": float(matrix[4, 4]),
        "C66": float(matrix[5, 5]),
    }


def _read_matrix(lines: List[str], heading_prefix: str, size: int) -> np.ndarray:
    """Read a dense numeric matrix immediately following a named heading."""
    try:
        heading_index = next(
            index for index, line in enumerate(lines) if line.strip().startswith(heading_prefix)
        )
    except StopIteration as exc:
        raise ValueError("Missing matrix heading: %s" % heading_prefix) from exc

    rows: List[List[float]] = []
    for line in lines[heading_index + 1 :]:
        if not line.strip():
            if rows:
                break
            continue
        try:
            values = [float(token) for token in line.split()]
        except ValueError:
            if rows:
                break
            continue
        if len(values) != size:
            if rows:
                break
            continue
        rows.append(values)
        if len(rows) == size:
            break

    if len(rows) != size:
        raise ValueError("Expected %d rows after %s" % (size, heading_prefix))
    matrix = np.asarray(rows, dtype=float)
    if not np.all(np.isfinite(matrix)):
        raise ValueError("Non-finite values found after %s" % heading_prefix)
    return matrix


def _exact_integer_cube_root(total_elements: int) -> int:
    """Return n for total_elements = n**3, rejecting inconsistent filenames."""
    if total_elements < 1:
        raise ValueError("Total element count must be positive")
    approximate = int(round(total_elements ** (1.0 / 3.0)))
    candidates = range(max(1, approximate - 2), approximate + 3)
    exact = [candidate for candidate in candidates if candidate**3 == total_elements]
    if len(exact) != 1:
        raise ValueError(
            "Total element count %d is not an exact integer cube" % total_elements
        )
    return exact[0]


def _zero_coupling_mask() -> np.ndarray:
    """Mask the constitutive entries that should vanish for isotropic elasticity."""
    theory = theoretical_isotropic_matrix()
    return np.isclose(theory, 0.0, atol=0.0, rtol=0.0)


def parse_result_file(path: Path) -> Dict[str, object]:
    """Parse one NIAH cube result and retain numerical-quality diagnostics."""
    match = re.fullmatch(r"CH-(Cube_E(\d+))-3D\.txt", path.name)
    if match is None:
        raise ValueError("Unexpected result filename: %s" % path.name)

    total_elements = int(match.group(2))
    elements_per_edge = _exact_integer_cube_root(total_elements)
    text = path.read_text(encoding="utf-8-sig")
    raw_matrix = _read_matrix(text.splitlines(), "EH (6x6", 6)

    # Energy-based elasticity is symmetric.  Average reciprocal entries for
    # the reported independent components, but retain the raw asymmetry below.
    matrix = 0.5 * (raw_matrix + raw_matrix.T)
    scale = float(np.max(np.abs(raw_matrix)))
    max_abs_asymmetry = float(np.max(np.abs(raw_matrix - raw_matrix.T)))
    relative_symmetry_residual = max_abs_asymmetry / scale if scale > 0.0 else 0.0
    max_abs_forbidden_coupling = float(np.max(np.abs(matrix[_zero_coupling_mask()])))

    record: Dict[str, object] = {
        "Case": match.group(1),
        "Total elements": total_elements,
        "Elements per edge": elements_per_edge,
        "Element edge length": CUBE_EDGE_LENGTH / float(elements_per_edge),
        "max_abs_asymmetry": max_abs_asymmetry,
        "relative_symmetry_residual": relative_symmetry_residual,
        "max_abs_forbidden_coupling": max_abs_forbidden_coupling,
        "source_txt": path.name,
    }
    record.update(
        {
            "C11": float(matrix[0, 0]),
            "C22": float(matrix[1, 1]),
            "C33": float(matrix[2, 2]),
            "C12": float(matrix[0, 1]),
            "C13": float(matrix[0, 2]),
            "C23": float(matrix[1, 2]),
            "C44": float(matrix[3, 3]),
            "C55": float(matrix[4, 4]),
            "C66": float(matrix[5, 5]),
        }
    )
    return record


def load_case_table() -> pd.DataFrame:
    """Load all cube result files and order them by increasing mesh density."""
    paths = list(ROOT.glob(TXT_GLOB))
    if not paths:
        raise FileNotFoundError("No result files matched %s" % TXT_GLOB)
    cases = pd.DataFrame.from_records(parse_result_file(path) for path in paths)
    if cases["Total elements"].duplicated().any():
        duplicates = cases.loc[cases["Total elements"].duplicated(), "Total elements"]
        raise ValueError("Duplicate element counts: %s" % duplicates.tolist())
    return cases.sort_values("Total elements").reset_index(drop=True)


def make_tidy_table(
    cases: pd.DataFrame, theory: Dict[str, float]
) -> pd.DataFrame:
    """Create long-form source data for plotting and independent verification."""
    rows: List[Dict[str, object]] = []
    for _, case in cases.iterrows():
        for family, components in COMPONENTS.items():
            for component in components:
                computed = float(case[component])
                reference = theory[component]
                signed_error = 100.0 * (computed / reference - 1.0)
                rows.append(
                    {
                        "Case": case["Case"],
                        "Total elements": int(case["Total elements"]),
                        "Elements per edge": int(case["Elements per edge"]),
                        "Element edge length": float(case["Element edge length"]),
                        "Family": family,
                        "Component": component,
                        "Computed (symmetrized)": computed,
                        "Theoretical": reference,
                        "Normalized stiffness": computed / reference,
                        "Signed relative error (%)": signed_error,
                        "Absolute relative error (%)": abs(signed_error),
                        "max_abs_asymmetry": float(case["max_abs_asymmetry"]),
                        "relative_symmetry_residual": float(
                            case["relative_symmetry_residual"]
                        ),
                        "max_abs_forbidden_coupling": float(
                            case["max_abs_forbidden_coupling"]
                        ),
                        "source_txt": case["source_txt"],
                    }
                )
    return pd.DataFrame(rows)


def _case_tick_labels(cases: pd.DataFrame) -> List[str]:
    return [
        "%d\n($%d^3$)" % (int(row["Total elements"]), int(row["Elements per edge"]))
        for _, row in cases.iterrows()
    ]


def _style_axis(axis: plt.Axes, panel_label: str, title: str) -> None:
    axis.text(
        -0.095,
        1.04,
        panel_label,
        transform=axis.transAxes,
        fontsize=8.5,
        fontweight="bold",
        va="bottom",
    )
    axis.set_title(title, loc="left", pad=4.0)
    axis.margins(x=0.02)


def _plot_family_lines(
    axis: plt.Axes,
    tidy: pd.DataFrame,
    family: str,
    y_column: str,
) -> None:
    for component in COMPONENTS[family]:
        subset = tidy.loc[tidy["Component"] == component].sort_values("Total elements")
        values = subset[y_column].to_numpy(dtype=float)
        if y_column == "Absolute relative error (%)":
            values = np.where(values > 0.0, values, np.nan)
        axis.plot(
            np.arange(len(subset)),
            values,
            color=COLORS[component],
            marker=MARKERS[component],
            markerfacecolor="white",
            markeredgewidth=0.9,
            label=component,
            zorder=3,
        )


def _set_ratio_limits(axis: plt.Axes, values: np.ndarray) -> None:
    """Apply honest, data-driven ratio limits while keeping theory visible."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("No finite normalized values available for plotting")
    lower = min(float(np.min(finite)), 1.0)
    upper = max(float(np.max(finite)), 1.0)
    span = upper - lower
    if span == 0.0:
        span = 2.0e-8
    axis.set_ylim(lower - 0.18 * span, upper + 0.18 * span)
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-3, 3))
    formatter.set_useOffset(True)
    axis.yaxis.set_major_formatter(formatter)


def plot_normalized_stiffness(
    cases: pd.DataFrame, tidy: pd.DataFrame, output_base: Path
) -> None:
    """Plot computed/theoretical ratios; the analytical target is one."""
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(180.0 / 25.4, 145.0 / 25.4),
        sharex=True,
        gridspec_kw={"hspace": 0.22},
    )
    panel_specs = (
        ("Normal", "a", "Normal stiffness"),
        ("Coupling", "b", "Normal coupling stiffness"),
        ("Shear", "c", "Shear stiffness"),
    )

    for axis, (family, panel_label, title) in zip(axes, panel_specs):
        _plot_family_lines(axis, tidy, family, "Normalized stiffness")
        axis.axhline(1.0, color="#333333", linewidth=0.9, linestyle=(0, (4, 2)), zorder=1)
        axis.set_ylabel(r"$C_{ij}^{\mathrm{NIAH}}/C_{ij}^{\mathrm{theory}}$")
        _style_axis(axis, panel_label, title)
        axis.legend(ncol=3, loc="best", handlelength=1.6)
        values = tidy.loc[
            tidy["Family"] == family, "Normalized stiffness"
        ].to_numpy(dtype=float)
        _set_ratio_limits(axis, values)

    maximum_error = float(tidy["Absolute relative error (%)"].max())
    axes[0].text(
        0.995,
        0.06,
        "Maximum error = %.2e%%" % maximum_error,
        transform=axes[0].transAxes,
        ha="right",
        va="bottom",
        fontsize=6.2,
        color="#444444",
    )

    axes[-1].set_xticks(np.arange(len(cases)))
    axes[-1].set_xticklabels(_case_tick_labels(cases), rotation=35, ha="right")
    axes[-1].set_xlabel("Total number of elements (per-edge count in parentheses)")

    fig.align_ylabels(axes)
    fig.subplots_adjust(left=0.125, right=0.985, top=0.975, bottom=0.19)
    save_publication_figure(fig, output_base)


def plot_absolute_error(
    cases: pd.DataFrame, tidy: pd.DataFrame, output_base: Path
) -> None:
    """Plot absolute relative errors against the isotropic analytical solution."""
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(180.0 / 25.4, 145.0 / 25.4),
        sharex=True,
        gridspec_kw={"hspace": 0.22},
    )
    panel_specs = (
        ("Normal", "a", "Normal stiffness"),
        ("Coupling", "b", "Normal coupling stiffness"),
        ("Shear", "c", "Shear stiffness"),
    )

    for axis, (family, panel_label, title) in zip(axes, panel_specs):
        _plot_family_lines(axis, tidy, family, "Absolute relative error (%)")
        family_values = tidy.loc[
            tidy["Family"] == family, "Absolute relative error (%)"
        ].to_numpy(dtype=float)
        positive = family_values[np.isfinite(family_values) & (family_values > 0.0)]
        if positive.size == 0:
            raise ValueError("All relative errors are exactly zero for %s" % family)
        lower = float(np.min(positive)) / 1.8
        upper = float(np.max(positive)) * 1.8
        if np.isclose(lower, upper):
            lower /= 2.0
            upper *= 2.0

        axis.set_yscale("log")
        axis.set_ylim(lower, upper)
        axis.yaxis.set_major_locator(LogLocator(base=10.0, numticks=5))
        axis.yaxis.set_minor_formatter(NullFormatter())
        axis.set_ylabel("Absolute relative error (%)")
        _style_axis(axis, panel_label, title)
        axis.legend(ncol=3, loc="best", handlelength=1.6)

    axes[-1].set_xticks(np.arange(len(cases)))
    axes[-1].set_xticklabels(_case_tick_labels(cases), rotation=35, ha="right")
    axes[-1].set_xlabel("Total number of elements (per-edge count in parentheses)")

    fig.align_ylabels(axes)
    fig.subplots_adjust(left=0.125, right=0.985, top=0.975, bottom=0.19)
    save_publication_figure(fig, output_base)


def save_publication_figure(fig: plt.Figure, output_base: Path) -> None:
    """Export editable vectors plus high-resolution submission and preview files."""
    fig.savefig(output_base.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(
        output_base.with_suffix(".tiff"),
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.02,
    )
    fig.savefig(
        output_base.with_suffix(".png"),
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.close(fig)


def write_theory_table(matrix: np.ndarray, output_path: Path) -> None:
    """Write the complete analytical constitutive matrix with explicit labels."""
    labels = ["1", "2", "3", "4", "5", "6"]
    table = pd.DataFrame(matrix, index=["C%s" % item for item in labels])
    table.columns = ["C%s" % item for item in labels]
    table.index.name = "row"
    table.to_csv(output_path)


def print_summary(cases: pd.DataFrame, tidy: pd.DataFrame) -> None:
    """Print the theory and the numerical range relevant to interpretation."""
    theory = theoretical_components(theoretical_isotropic_matrix())
    print("\nAnalytical isotropic stiffness for E = %.6g, nu = %.6g:" % (
        YOUNGS_MODULUS,
        POISSON_RATIO,
    ))
    print("  C11 = C22 = C33 = %.12e" % theory["C11"])
    print("  C12 = C13 = C23 = %.12e" % theory["C12"])
    print("  C44 = C55 = C66 = %.12e" % theory["C44"])
    print("\nMesh cases: %d^3 to %d^3 elements" % (
        int(cases["Elements per edge"].min()),
        int(cases["Elements per edge"].max()),
    ))
    print("Maximum absolute relative error: %.6e %%" % (
        float(tidy["Absolute relative error (%)"].max())
    ))
    print("Maximum raw-matrix relative symmetry residual: %.6e" % (
        float(cases["relative_symmetry_residual"].max())
    ))
    print("Maximum absolute forbidden coupling: %.6e" % (
        float(cases["max_abs_forbidden_coupling"].max())
    ))


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    theory_matrix = theoretical_isotropic_matrix()
    theory = theoretical_components(theory_matrix)
    cases = load_case_table()
    tidy = make_tidy_table(cases, theory)

    cases.to_csv(OUTPUT_DIR / "cube_3d_case_results.csv", index=False)
    tidy.to_csv(OUTPUT_DIR / "cube_3d_convergence_tidy_data.csv", index=False)
    write_theory_table(
        theory_matrix,
        OUTPUT_DIR / "cube_3d_theoretical_stiffness.csv",
    )

    plot_normalized_stiffness(
        cases,
        tidy,
        OUTPUT_DIR / "cube_3d_mesh_convergence_normalized",
    )
    plot_absolute_error(
        cases,
        tidy,
        OUTPUT_DIR / "cube_3d_mesh_convergence_absolute_error",
    )
    print_summary(cases, tidy)
    print("\nOutputs written to: %s" % OUTPUT_DIR)


if __name__ == "__main__":
    main()
