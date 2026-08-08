"""Ordinary-Python checks for the static field-comparison mathematics."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


PLUGIN_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = (
    PLUGIN_ROOT.parent
    if PLUGIN_ROOT.name == "NIAH Abaqus Plugin"
    else PLUGIN_ROOT
)
SCRIPT = (
    REPOSITORY_ROOT
    / "examples"
    / "static_field_validation"
    / "compare_static_displacement_fields.py"
)
SPEC = importlib.util.spec_from_file_location("static_field_validation", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)

EXTRACTOR_SCRIPT = (
    REPOSITORY_ROOT
    / "examples"
    / "static_field_validation"
    / "extract_odb_displacement.py"
)
EXTRACTOR_SPEC = importlib.util.spec_from_file_location(
    "extract_odb_displacement", EXTRACTOR_SCRIPT
)
EXTRACTOR = importlib.util.module_from_spec(EXTRACTOR_SPEC)
assert EXTRACTOR_SPEC.loader is not None
EXTRACTOR_SPEC.loader.exec_module(EXTRACTOR)


class _MockNode:
    def __init__(self, label, coordinates):
        self.label = label
        self.coordinates = coordinates


def test_envelope_alignment_handles_translation_and_shell_midplane():
    homo = np.array([[0.0, 0.0, 0.0], [2.0, 2.0, 0.0]])
    fine = np.array([[-1.0, 3.0, -1.0], [1.0, 5.0, 1.0]])
    query, metadata = MODULE.map_homo_to_fine_envelope(homo, homo, fine)
    np.testing.assert_allclose(query, [[-1.0, 3.0, 0.0], [1.0, 5.0, 0.0]])
    assert metadata["maximum_span_mismatch_fraction"] == 0.0


def test_alignment_rejects_implicit_geometry_scaling():
    homo = np.array([[0.0, 0.0, 0.0], [2.0, 2.0, 0.0]])
    fine = np.array([[0.0, 0.0, -1.0], [3.0, 2.0, 1.0]])
    _, metadata = MODULE.map_homo_to_fine_envelope(homo, homo, fine)
    with pytest.raises(ValueError, match="geometry mismatch"):
        MODULE.validate_alignment(metadata, allow_envelope_scaling=False)
    MODULE.validate_alignment(metadata, allow_envelope_scaling=True)


def test_mls_exactly_recovers_an_affine_vector_field():
    grid = np.linspace(0.0, 1.0, 5)
    fine_coordinates = np.array(
        [(x, y, z) for x in grid for y in grid for z in grid], dtype=float
    )
    matrix = np.array(
        [[2.0, -0.5, 0.25], [0.1, 1.2, -0.3], [-0.2, 0.4, 0.8]]
    )
    intercept = np.array([0.3, -0.1, 0.7])
    fine_u = fine_coordinates @ matrix.T + intercept
    query = np.array([[0.17, 0.33, 0.61], [0.82, 0.44, 0.29]])
    expected = query @ matrix.T + intercept
    predicted, conditions, _ = MODULE.moving_least_squares(
        fine_coordinates, fine_u, query, neighbours=32
    )
    np.testing.assert_allclose(predicted, expected, atol=2.0e-14)
    assert np.isfinite(conditions).all()


def test_spatial_sampling_is_deterministic_and_bounded():
    grid = np.linspace(0.0, 1.0, 101)
    coordinates = np.array([(x, y, 0.0) for x in grid for y in grid])
    first = MODULE.sample_homogenized_nodes(coordinates, max_points=400)
    second = MODULE.sample_homogenized_nodes(coordinates, max_points=400)
    np.testing.assert_array_equal(first, second)
    assert 300 <= first.size <= 400


def test_abaqus_side_sampling_preserves_exact_envelope_and_limit():
    grid = np.linspace(-2.0, 3.0, 21)
    nodes = [
        _MockNode(index + 1, (x, y, z))
        for index, (x, y, z) in enumerate(
            (point for x in grid for y in grid[::2] for z in grid[::4]
             for point in [(x, y, z)])
        )
    ]
    selected = EXTRACTOR._spatially_sample_nodes(nodes, max_nodes=500)
    coordinates = np.asarray([record[1:] for record in selected], dtype=float)
    assert len(selected) <= 500
    np.testing.assert_allclose(coordinates.min(axis=0), [-2.0, -2.0, -2.0])
    np.testing.assert_allclose(coordinates.max(axis=0), [3.0, 3.0, 3.0])


def test_identical_fields_have_zero_error_and_unit_mac():
    field = np.array([[0.0, 0.0, 0.0], [0.1, -0.2, 0.3], [0.4, 0.2, -0.1]])
    metrics = MODULE.field_metrics(field, field.copy())
    assert metrics["vector_relative_L2"] == 0.0
    assert metrics["magnitude_relative_L2"] == 0.0
    np.testing.assert_allclose(metrics["displacement_field_MAC"], 1.0)
    assert metrics["peak_U_signed_relative_error"] == 0.0


def test_figures_are_split_and_export_only_svg_and_tiff(tmp_path):
    grid = np.linspace(0.0, 1.0, 8)
    coordinates = np.array([(x, y, 0.0) for x in grid for y in grid])
    fine_u = np.column_stack(
        (0.1 * coordinates[:, 0], -0.2 * coordinates[:, 1], 0.05 * coordinates[:, 0])
    )
    homo_u = fine_u * 1.01
    fine_umag = np.linalg.norm(fine_u, axis=1)
    homo_umag = np.linalg.norm(homo_u, axis=1)
    error = np.linalg.norm(homo_u - fine_u, axis=1)
    sampled = MODULE.pd.DataFrame(
        {
            "x": coordinates[:, 0],
            "y": coordinates[:, 1],
            "z": coordinates[:, 2],
            "fine_umag": fine_umag,
            "homo_umag": homo_umag,
            "pointwise_error_over_fine_peak": error / fine_umag.max(),
        }
    )
    metrics = MODULE.field_metrics(homo_u, fine_u)
    stem = tmp_path / "validation"
    MODULE.make_validation_figures("qa", sampled, metrics, stem)

    expected_suffixes = ("fine_field", "homo_field", "error_field", "parity")
    for suffix in expected_suffixes:
        assert (tmp_path / f"validation_{suffix}.svg").is_file()
        assert (tmp_path / f"validation_{suffix}.tiff").is_file()
    with Image.open(tmp_path / "validation_fine_field.tiff") as image:
        expected_width = round(86.0 / 25.4 * 600.0)
        expected_height = round(67.0 / 25.4 * 600.0)
        assert abs(image.width - expected_width) <= 1
        assert abs(image.height - expected_height) <= 1
    assert not list(tmp_path.glob("*.pdf"))
    assert not list(tmp_path.glob("*.png"))
