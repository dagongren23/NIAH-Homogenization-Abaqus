"""Mesh-convergence analysis for the homogeneous 2 x 2 x 0.1 plate.

The script reads:
  * ``Convergence analysis examples.xlsx`` for the through-thickness mesh count;
  * ``CH-Plate_T*-shell-nz.txt`` for the NIAH A, B, D and K results.

It writes publication-ready figures and traceable CSV tables to
``convergence_outputs``.  The classical reference is the unmodified
Reissner--Mindlin plate solution with kappa = 5/6.  A separate table also
reports the midpoint-integration correction eta_D = 1 - 1 / n_z**2 for D.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter, ScalarFormatter
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Inputs and physical constants (consistent units; E is normalized to 1)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
EXCEL_FILE = ROOT / "Convergence analysis examples.xlsx"
TXT_GLOB = "CH-Plate_T*-shell-nz.txt"
OUTPUT_DIR = ROOT / "convergence_outputs"

E = 1.0
NU = 0.3
THICKNESS = 0.1
SHEAR_CORRECTION = 5.0 / 6.0
REFERENCE_CASE = "Plate_T73926"

COMPONENTS = {
    "A": ("A11", "A22", "A12", "A66"),
    "D": ("D11", "D22", "D12", "D66"),
    "K": ("Kxz", "Kyz"),
}

COLORS = {
    "A11": "#0072B2",
    "A22": "#D55E00",
    "A12": "#009E73",
    "A66": "#CC79A7",
    "D11": "#0072B2",
    "D22": "#D55E00",
    "D12": "#009E73",
    "D66": "#CC79A7",
    "Kxz": "#0072B2",
    "Kyz": "#D55E00",
}
MARKERS = {
    "A11": "o",
    "A22": "s",
    "A12": "^",
    "A66": "D",
    "D11": "o",
    "D22": "s",
    "D12": "^",
    "D66": "D",
    "Kxz": "o",
    "Kyz": "s",
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


def theoretical_stiffnesses(
    youngs_modulus: float = E,
    poisson_ratio: float = NU,
    thickness: float = THICKNESS,
    shear_correction: float = SHEAR_CORRECTION,
) -> Dict[str, float]:
    """Return the non-zero isotropic plate stiffness components."""
    q11 = youngs_modulus / (1.0 - poisson_ratio**2)
    q12 = poisson_ratio * q11
    q66 = youngs_modulus / (2.0 * (1.0 + poisson_ratio))

    a_scale = thickness
    d_scale = thickness**3 / 12.0
    k_scale = shear_correction * thickness
    return {
        "A11": q11 * a_scale,
        "A22": q11 * a_scale,
        "A12": q12 * a_scale,
        "A66": q66 * a_scale,
        "D11": q11 * d_scale,
        "D22": q11 * d_scale,
        "D12": q12 * d_scale,
        "D66": q66 * d_scale,
        "Kxz": q66 * k_scale,
        "Kyz": q66 * k_scale,
    }


def _read_matrix(lines: List[str], heading_prefix: str, size: int) -> np.ndarray:
    """Read a dense matrix immediately following a named TXT heading."""
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
        values = [float(token) for token in line.split()]
        if len(values) != size:
            if rows:
                break
            continue
        rows.append(values)
        if len(rows) == size:
            break

    if len(rows) != size:
        raise ValueError("Expected %d rows after %s" % (size, heading_prefix))
    return np.asarray(rows, dtype=float)


def parse_result_file(path: Path) -> Dict[str, object]:
    """Parse one NIAH shell result text file."""
    match = re.fullmatch(r"CH-(Plate_T(\d+))-shell-nz\.txt", path.name)
    if match is None:
        raise ValueError("Unexpected result filename: %s" % path.name)

    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    a_matrix = _read_matrix(lines, "A (3x3", 3)
    b_matrix = _read_matrix(lines, "B (3x3", 3)
    d_matrix = _read_matrix(lines, "D (3x3", 3)

    kxz_match = re.search(r"^Kxz\s*=\s*([+\-0-9.eE]+)", text, re.MULTILINE)
    kyz_match = re.search(r"^Kyz\s*=\s*([+\-0-9.eE]+)", text, re.MULTILINE)
    if kxz_match is None or kyz_match is None:
        raise ValueError("Missing Kxz/Kyz in %s" % path.name)

    return {
        "Example Name": match.group(1),
        "Total elements": int(match.group(2)),
        "A11": a_matrix[0, 0],
        "A22": a_matrix[1, 1],
        "A12": a_matrix[0, 1],
        "A66": a_matrix[2, 2],
        "D11": d_matrix[0, 0],
        "D22": d_matrix[1, 1],
        "D12": d_matrix[0, 1],
        "D66": d_matrix[2, 2],
        "Kxz": float(kxz_match.group(1)),
        "Kyz": float(kyz_match.group(1)),
        "max_abs_B": float(np.max(np.abs(b_matrix))),
        "source_txt": path.name,
    }


def load_case_table() -> pd.DataFrame:
    """Join the TXT results to the Excel through-thickness mesh metadata."""
    mesh = pd.read_excel(EXCEL_FILE, sheet_name="Sheet1", engine="openpyxl")
    required = {"Example Name", "Number of thickness direction elements"}
    missing_columns = required.difference(mesh.columns)
    if missing_columns:
        raise ValueError("Excel file is missing columns: %s" % sorted(missing_columns))

    if mesh["Example Name"].duplicated().any():
        duplicates = mesh.loc[mesh["Example Name"].duplicated(), "Example Name"].tolist()
        raise ValueError("Duplicate Excel example names: %s" % duplicates)

    records = [parse_result_file(path) for path in sorted(ROOT.glob(TXT_GLOB))]
    if not records:
        raise FileNotFoundError("No result files matched %s" % TXT_GLOB)

    results = pd.DataFrame.from_records(records)
    joined = results.merge(mesh, on="Example Name", how="outer", indicator=True)
    unmatched = joined.loc[joined["_merge"] != "both", ["Example Name", "_merge"]]
    if not unmatched.empty:
        raise ValueError("TXT/Excel case mismatch:\n%s" % unmatched.to_string(index=False))

    nz_column = "Number of thickness direction elements"
    joined[nz_column] = pd.to_numeric(joined[nz_column], errors="raise").astype(int)
    if (joined[nz_column] < 1).any():
        raise ValueError("All through-thickness element counts must be positive")

    return joined.drop(columns="_merge").sort_values("Total elements").reset_index(drop=True)


def make_tidy_table(cases: pd.DataFrame, theory: Dict[str, float]) -> pd.DataFrame:
    """Create a traceable long-form convergence table."""
    nz_column = "Number of thickness direction elements"
    rows: List[Dict[str, object]] = []
    for _, case in cases.iterrows():
        for family, components in COMPONENTS.items():
            for component in components:
                computed = float(case[component])
                reference = theory[component]
                signed_error = 100.0 * (computed / reference - 1.0)
                rows.append(
                    {
                        "Example Name": case["Example Name"],
                        "Total elements": int(case["Total elements"]),
                        "n_z": int(case[nz_column]),
                        "Family": family,
                        "Component": component,
                        "Computed": computed,
                        "Classical theory": reference,
                        "Normalized stiffness": computed / reference,
                        "Signed relative error (%)": signed_error,
                        "Absolute relative error (%)": abs(signed_error),
                        "max_abs_B": float(case["max_abs_B"]),
                        "source_txt": case["source_txt"],
                    }
                )
    return pd.DataFrame(rows)


def make_reference_case_table(
    tidy: pd.DataFrame, theory: Dict[str, float], reference_case: str = REFERENCE_CASE
) -> pd.DataFrame:
    """Compare one case with both classical and D-discretized references."""
    selected = tidy.loc[tidy["Example Name"] == reference_case].copy()
    if selected.empty:
        raise ValueError("Reference case not found: %s" % reference_case)

    nz_values = selected["n_z"].unique()
    if len(nz_values) != 1:
        raise ValueError("Reference case has inconsistent n_z values")
    nz = int(nz_values[0])
    eta_d = 1.0 - 1.0 / float(nz**2)

    corrected_values: List[float] = []
    correction_factors: List[float] = []
    for component in selected["Component"]:
        factor = eta_d if component.startswith("D") else 1.0
        correction_factors.append(factor)
        corrected_values.append(theory[component] * factor)

    selected["D discretization factor"] = correction_factors
    selected["Discretized theory"] = corrected_values
    selected["Error vs discretized theory (%)"] = (
        100.0 * (selected["Computed"] / selected["Discretized theory"] - 1.0)
    )
    selected["Absolute error vs discretized theory (%)"] = selected[
        "Error vs discretized theory (%)"
    ].abs()

    columns = [
        "Example Name",
        "Total elements",
        "n_z",
        "Family",
        "Component",
        "Computed",
        "Classical theory",
        "Signed relative error (%)",
        "D discretization factor",
        "Discretized theory",
        "Error vs discretized theory (%)",
        "Absolute error vs discretized theory (%)",
        "max_abs_B",
    ]
    return selected[columns].reset_index(drop=True)


def _case_tick_labels(cases: pd.DataFrame) -> List[str]:
    nz_column = "Number of thickness direction elements"
    return [
        "%d\n($n_z$=%d)" % (int(row["Total elements"]), int(row[nz_column]))
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
        axis.plot(
            np.arange(len(subset)),
            subset[y_column],
            color=COLORS[component],
            marker=MARKERS[component],
            markerfacecolor="white",
            markeredgewidth=0.9,
            label=component,
            zorder=3,
        )


def plot_normalized_stiffness(
    cases: pd.DataFrame, tidy: pd.DataFrame, output_base: Path
) -> None:
    """Plot computed/classical-theory ratios; the convergence target is one."""
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(180.0 / 25.4, 145.0 / 25.4),
        sharex=True,
        gridspec_kw={"hspace": 0.22},
    )
    panel_specs = (
        ("A", "a", "Extensional stiffness"),
        ("D", "b", "Bending stiffness"),
        ("K", "c", "Transverse shear stiffness"),
    )

    for axis, (family, panel_label, title) in zip(axes, panel_specs):
        _plot_family_lines(axis, tidy, family, "Normalized stiffness")
        axis.axhline(1.0, color="#333333", linewidth=0.9, linestyle=(0, (4, 2)), zorder=1)
        axis.set_ylabel(r"$C^\mathrm{NIAH}/C^\mathrm{theory}$")
        _style_axis(axis, panel_label, title)
        axis.legend(ncol=len(COMPONENTS[family]), loc="best", handlelength=1.6)

    a_values = tidy.loc[tidy["Family"] == "A", "Normalized stiffness"].to_numpy()
    a_span = max(float(np.ptp(a_values)), 2.0e-8)
    a_mid = float(np.mean(a_values))
    axes[0].set_ylim(a_mid - 1.3 * a_span, a_mid + 1.3 * a_span)
    axes[0].ticklabel_format(axis="y", style="plain", useOffset=False)

    d_values = tidy.loc[tidy["Family"] == "D", "Normalized stiffness"].to_numpy()
    d_min, d_max = float(np.min(d_values)), float(np.max(d_values))
    d_pad = 0.08 * (d_max - d_min)
    axes[1].set_ylim(min(0.68, d_min - d_pad), max(1.08, d_max + d_pad))

    k_values = tidy.loc[tidy["Family"] == "K", "Normalized stiffness"].to_numpy()
    axes[2].set_yscale("log")
    axes[2].set_ylim(0.9, float(np.max(k_values)) * 1.25)
    axes[2].yaxis.set_major_locator(LogLocator(base=10.0, numticks=5))
    axes[2].yaxis.set_minor_formatter(NullFormatter())

    x = np.arange(len(cases))
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(_case_tick_labels(cases), rotation=35, ha="right")
    axes[-1].set_xlabel("Total number of elements (through-thickness count in parentheses)")

    fig.align_ylabels(axes)
    fig.subplots_adjust(left=0.105, right=0.985, top=0.975, bottom=0.19)
    save_publication_figure(fig, output_base)


def plot_absolute_error(cases: pd.DataFrame, tidy: pd.DataFrame, output_base: Path) -> None:
    """Plot absolute relative errors against the unmodified analytical solution."""
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(180.0 / 25.4, 145.0 / 25.4),
        sharex=True,
        gridspec_kw={"hspace": 0.22},
    )
    panel_specs = (
        ("A", "a", "Extensional stiffness"),
        ("D", "b", "Bending stiffness"),
        ("K", "c", "Transverse shear stiffness"),
    )

    for axis, (family, panel_label, title) in zip(axes, panel_specs):
        _plot_family_lines(axis, tidy, family, "Absolute relative error (%)")
        axis.set_yscale("log")
        axis.set_ylabel("Absolute relative error (%)")
        _style_axis(axis, panel_label, title)
        axis.legend(ncol=len(COMPONENTS[family]), loc="best", handlelength=1.6)

        family_values = tidy.loc[
            tidy["Family"] == family, "Absolute relative error (%)"
        ].to_numpy()
        positive_values = family_values[family_values > 0.0]
        lower = float(np.min(positive_values)) / 1.8
        upper = float(np.max(positive_values)) * 1.8
        axis.set_ylim(lower, upper)
        if lower <= 1.0 <= upper:
            axis.axhline(
                1.0,
                color="#777777",
                linewidth=0.8,
                linestyle=(0, (3, 2)),
                zorder=1,
            )
            axis.text(
                0.995,
                1.0,
                "1%",
                transform=axis.get_yaxis_transform(),
                ha="right",
                va="bottom",
                color="#666666",
                fontsize=6.2,
            )

    x = np.arange(len(cases))
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(_case_tick_labels(cases), rotation=35, ha="right")
    axes[-1].set_xlabel("Total number of elements (through-thickness count in parentheses)")

    fig.align_ylabels(axes)
    fig.subplots_adjust(left=0.115, right=0.985, top=0.975, bottom=0.19)
    save_publication_figure(fig, output_base)


def save_publication_figure(fig: plt.Figure, output_base: Path) -> None:
    """Export editable vector files plus high-resolution submission/preview files."""
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


def print_reference_summary(reference: pd.DataFrame) -> None:
    """Print the key reference-case comparison to the console."""
    display_columns = [
        "Component",
        "Computed",
        "Classical theory",
        "Signed relative error (%)",
        "Discretized theory",
        "Error vs discretized theory (%)",
    ]
    print("\n%s comparison:" % REFERENCE_CASE)
    print(
        reference[display_columns].to_string(
            index=False,
            float_format=lambda value: "%.8e" % value,
        )
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    theory = theoretical_stiffnesses()
    cases = load_case_table()
    tidy = make_tidy_table(cases, theory)
    reference = make_reference_case_table(tidy, theory)

    cases.to_csv(OUTPUT_DIR / "case_metadata_and_results.csv", index=False)
    tidy.to_csv(OUTPUT_DIR / "convergence_tidy_data.csv", index=False)
    reference.to_csv(OUTPUT_DIR / "Plate_T73926_error_summary.csv", index=False)

    plot_normalized_stiffness(
        cases,
        tidy,
        OUTPUT_DIR / "plate_mesh_convergence_normalized",
    )
    plot_absolute_error(
        cases,
        tidy,
        OUTPUT_DIR / "plate_mesh_convergence_absolute_error",
    )
    print_reference_summary(reference)
    print("\nOutputs written to: %s" % OUTPUT_DIR)


if __name__ == "__main__":
    main()
