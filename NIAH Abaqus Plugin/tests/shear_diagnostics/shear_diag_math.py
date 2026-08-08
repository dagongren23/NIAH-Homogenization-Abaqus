# -*- coding: utf-8 -*-
"""Pure NumPy mathematics for read-only NIAH shear diagnostics.

The module has no Abaqus imports and does not alter the production stiffness
calculation.  It is compatible with both Python 2.7 (Abaqus 2020) and modern
Python 3 so the numerical operations can be unit-tested independently.
"""
from __future__ import division

import numpy as np


EPS = 1.0e-30


def _vector(values, name):
    arr = np.asarray(values, dtype=float).reshape(-1)
    if not np.all(np.isfinite(arr)):
        raise ValueError("%s contains a non-finite value." % name)
    return arr


def relative_residual(numerator, denominator, eps=EPS):
    """Return ``abs(numerator) / max(abs(denominator), eps)``."""
    return abs(float(numerator)) / max(abs(float(denominator)), float(eps))


def split_generalized_work(displacement, force, dof_per_node):
    """Split ``u dot f`` into translation and rotation contributions."""
    u = _vector(displacement, "displacement")
    f = _vector(force, "force")
    if u.shape != f.shape:
        raise ValueError("displacement and force must have identical shapes.")
    if dof_per_node not in (3, 6):
        raise ValueError("dof_per_node must be 3 or 6.")
    if u.size % dof_per_node:
        raise ValueError("Vector length is not divisible by dof_per_node.")

    if dof_per_node == 3:
        translation = float(np.dot(u, f))
        rotation = 0.0
    else:
        uv = u.reshape((-1, 6))
        fv = f.reshape((-1, 6))
        translation = float(np.sum(uv[:, :3] * fv[:, :3]))
        rotation = float(np.sum(uv[:, 3:] * fv[:, 3:]))
    return {
        "translation_raw": translation,
        "rotation_raw": rotation,
        "total_raw": translation + rotation,
        "translation_half": 0.5 * translation,
        "rotation_half": 0.5 * rotation,
        "total_half": 0.5 * (translation + rotation),
    }


def chain_energy_decomposition(u1, u2, f1, f2, dof_per_node):
    """Expand the four bilinear terms of a three-stage NIAH chain.

    Production code uses ``(u1 + u2) dot (f1 + f2)`` without a factor of
    one half.  Both that raw convention and the physical half-work convention
    are reported here.
    """
    u1 = _vector(u1, "u1")
    u2 = _vector(u2, "u2")
    f1 = _vector(f1, "f1")
    f2 = _vector(f2, "f2")
    shapes = set((u1.shape, u2.shape, f1.shape, f2.shape))
    if len(shapes) != 1:
        raise ValueError("u1, u2, f1 and f2 must have identical shapes.")

    terms = {
        "u1_f1_raw": float(np.dot(u1, f1)),
        "u1_f2_raw": float(np.dot(u1, f2)),
        "u2_f1_raw": float(np.dot(u2, f1)),
        "u2_f2_raw": float(np.dot(u2, f2)),
    }
    expanded = sum(terms.values())
    combined = float(np.dot(u1 + u2, f1 + f2))
    terms.update({
        "expanded_raw": expanded,
        "combined_raw": combined,
        "closure_abs": abs(expanded - combined),
        "closure_rel": relative_residual(expanded - combined, combined),
        "combined_half": 0.5 * combined,
        "combined_split": split_generalized_work(
            u1 + u2, f1 + f2, dof_per_node
        ),
    })
    return terms


def mutual_work_matrix(displacements, forces, scales=None):
    """Build raw, symmetric, and antisymmetric mutual-work matrices.

    Columns/states are scaled by ``scales`` before computing mutual work.
    Applying the same scale to displacement and force preserves the intended
    geometric normalization of each diagonal work term.
    """
    if len(displacements) != len(forces):
        raise ValueError("displacements and forces must have equal lengths.")
    count = len(displacements)
    if count == 0:
        raise ValueError("At least one state is required.")
    if scales is None:
        scales = np.ones(count, dtype=float)
    scales = _vector(scales, "scales")
    if scales.size != count:
        raise ValueError("scales must contain one value per state.")

    us = [_vector(value, "displacement") for value in displacements]
    fs = [_vector(value, "force") for value in forces]
    expected = us[0].shape
    if any(value.shape != expected for value in us + fs):
        raise ValueError("All states must have identical vector shapes.")

    raw = np.zeros((count, count), dtype=float)
    for i in range(count):
        for j in range(count):
            raw[i, j] = (
                scales[i] * scales[j] * float(np.dot(us[i], fs[j]))
            )
    symmetric = 0.5 * (raw + raw.T)
    antisymmetric = 0.5 * (raw - raw.T)
    denom = max(float(np.linalg.norm(symmetric)), EPS)
    return {
        "raw": raw,
        "symmetric": symmetric,
        "antisymmetric": antisymmetric,
        "reciprocity_relative": float(np.linalg.norm(antisymmetric)) / denom,
    }


def full_shear_energy_diagnostic(su_x, sf_x, su_y, sf_y, dbar, length, height,
                                 bending_work=None):
    """Construct a complete 2x2 *diagnostic* shear-energy matrix.

    This function deliberately does not replace the production K recovery.
    It uses reciprocal cross work, includes D12 in the default bending-work
    subtraction, and reports positivity/conditioning.  The optional
    ``bending_work`` argument can supply a more accurate matrix obtained from
    fitted curvature fields and numerical area integration.
    """
    dbar = np.asarray(dbar, dtype=float)
    if dbar.shape != (3, 3):
        raise ValueError("dbar must have shape (3, 3).")
    length = float(length)
    height = float(height)
    if length <= 0.0 or height <= 0.0:
        raise ValueError("length and height must be positive.")

    scales = np.array([
        np.sqrt(length / height),
        np.sqrt(height / length),
    ], dtype=float)
    work = mutual_work_matrix([su_x, su_y], [sf_x, sf_y], scales=scales)
    if bending_work is None:
        bending = np.array([
            [dbar[0, 0] * length * length / 12.0,
             dbar[0, 1] * length * height / 12.0],
            [dbar[1, 0] * length * height / 12.0,
             dbar[1, 1] * height * height / 12.0],
        ], dtype=float)
    else:
        bending = np.asarray(bending_work, dtype=float)
        if bending.shape != (2, 2):
            raise ValueError("bending_work must have shape (2, 2).")
        bending = 0.5 * (bending + bending.T)

    residual = work["symmetric"] - bending
    eigenvalues = np.linalg.eigvalsh(residual)
    result = {
        "work_raw": work["raw"],
        "work_symmetric": work["symmetric"],
        "work_antisymmetric": work["antisymmetric"],
        "reciprocity_relative": work["reciprocity_relative"],
        "bending_work": bending,
        "residual_energy": residual,
        "residual_eigenvalues": eigenvalues,
        "residual_positive_definite": bool(np.all(eigenvalues > 0.0)),
        "residual_condition": float(np.linalg.cond(residual)),
    }

    # Experimental full-matrix extension of the scalar legacy relation.
    # It is evidence only and must never replace production K automatically.
    transform = np.array([
        [dbar[0, 0] ** 2, dbar[0, 2] ** 2],
        [dbar[1, 2] ** 2, dbar[1, 1] ** 2],
    ], dtype=float)
    result["legacy_transform"] = transform
    try:
        inv_transform = np.linalg.inv(transform)
        compliance_candidate = np.dot(
            inv_transform, np.dot(residual, inv_transform.T)
        )
        compliance_candidate = 0.5 * (
            compliance_candidate + compliance_candidate.T
        )
        result["compliance_candidate"] = compliance_candidate
        result["stiffness_candidate"] = np.linalg.inv(compliance_candidate)
        result["candidate_valid"] = True
    except np.linalg.LinAlgError:
        result["compliance_candidate"] = None
        result["stiffness_candidate"] = None
        result["candidate_valid"] = False
    return result


def periodic_pair_residuals(values, pairs, dof_per_node, jumps=None):
    """Evaluate positive-minus-negative-minus-jump residuals by node label."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[1] != dof_per_node:
        raise ValueError("values must have shape (node_count, dof_per_node).")
    pairs = list(pairs)
    if jumps is None:
        jumps = np.zeros((len(pairs), dof_per_node), dtype=float)
    jumps = np.asarray(jumps, dtype=float)
    if jumps.shape != (len(pairs), dof_per_node):
        raise ValueError("jumps has an incompatible shape.")

    residuals = np.zeros_like(jumps)
    for row, pair in enumerate(pairs):
        positive, negative = int(pair[0]), int(pair[1])
        residuals[row, :] = (
            values[positive - 1, :] - values[negative - 1, :] - jumps[row, :]
        )
    norms = np.linalg.norm(residuals, axis=1) if len(pairs) else np.zeros(0)
    reference = np.linalg.norm(jumps, axis=1) if len(pairs) else np.zeros(0)
    return {
        "residuals": residuals,
        "norms": norms,
        "max_abs": float(np.max(np.abs(residuals))) if residuals.size else 0.0,
        "rms": float(np.sqrt(np.mean(residuals ** 2))) if residuals.size else 0.0,
        "relative_rms": (
            float(np.sqrt(np.mean(residuals ** 2))) /
            max(float(np.sqrt(np.mean(jumps ** 2))), EPS)
            if residuals.size else 0.0
        ),
        "reference_norms": reference,
    }


def traction_pair_residuals(forces, pairs):
    """Evaluate anti-periodicity residual ``f_positive + f_negative``."""
    forces = np.asarray(forces, dtype=float)
    if forces.ndim != 2:
        raise ValueError("forces must be a two-dimensional array.")
    residuals = []
    relative = []
    for positive, negative in pairs:
        fp = forces[int(positive) - 1, :]
        fm = forces[int(negative) - 1, :]
        residual = fp + fm
        residuals.append(residual)
        relative.append(
            float(np.linalg.norm(residual)) /
            max(float(np.linalg.norm(fp)), float(np.linalg.norm(fm)), EPS)
        )
    if residuals:
        residuals = np.asarray(residuals, dtype=float)
        relative = np.asarray(relative, dtype=float)
    else:
        residuals = np.zeros((0, forces.shape[1]), dtype=float)
        relative = np.zeros(0, dtype=float)
    return {
        "residuals": residuals,
        "relative": relative,
        "max_abs": float(np.max(np.abs(residuals))) if residuals.size else 0.0,
        "rms": float(np.sqrt(np.mean(residuals ** 2))) if residuals.size else 0.0,
        "relative_max": float(np.max(relative)) if relative.size else 0.0,
        "relative_rms": float(np.sqrt(np.mean(relative ** 2))) if relative.size else 0.0,
    }


def fit_transverse_cubic(coords, generalized_displacement, dof_per_node):
    """Fit the transverse field and report desired/spurious curvatures.

    ``w(x,y)`` is fitted with a complete cubic polynomial.  Curvatures use
    ``kxx=-d2w/dx2``, ``kyy=-d2w/dy2`` and engineering
    ``kxy=-2*d2w/(dx*dy)``.  For six-DOF states, the Reissner-Mindlin shear
    consistency residuals ``theta_y + dw/dx`` and ``theta_x - dw/dy`` are
    reported as well.
    """
    coords = np.asarray(coords, dtype=float)
    vector = _vector(generalized_displacement, "generalized_displacement")
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError("coords must have shape (node_count, at least 2).")
    if vector.size != coords.shape[0] * dof_per_node:
        raise ValueError("Vector length does not match coordinates.")
    if dof_per_node not in (3, 6):
        raise ValueError("dof_per_node must be 3 or 6.")

    x = coords[:, 0] - 0.5 * (coords[:, 0].max() + coords[:, 0].min())
    y = coords[:, 1] - 0.5 * (coords[:, 1].max() + coords[:, 1].min())
    design = np.column_stack((
        np.ones_like(x), x, y, x*x, x*y, y*y,
        x*x*x, x*x*y, x*y*y, y*y*y,
    ))
    nodal = vector.reshape((-1, dof_per_node))
    w = nodal[:, 2]
    coeff, _, rank, singular = np.linalg.lstsq(design, w, rcond=None)
    fitted = np.dot(design, coeff)
    residual = w - fitted

    kxx = -(2.0 * coeff[3] + 6.0 * coeff[6] * x + 2.0 * coeff[7] * y)
    kyy = -(2.0 * coeff[5] + 2.0 * coeff[8] * x + 6.0 * coeff[9] * y)
    kxy = -2.0 * (coeff[4] + 2.0 * coeff[7] * x + 2.0 * coeff[8] * y)
    result = {
        "coefficients": coeff,
        "rank": int(rank),
        "condition": (
            float(singular[0] / singular[-1]) if singular[-1] > 0.0 else np.inf
        ),
        "w_rmse": float(np.sqrt(np.mean(residual ** 2))),
        "w_relative_rmse": (
            float(np.sqrt(np.mean(residual ** 2))) /
            max(float(np.sqrt(np.mean(w ** 2))), EPS)
        ),
        "kxx_rms": float(np.sqrt(np.mean(kxx ** 2))),
        "kyy_rms": float(np.sqrt(np.mean(kyy ** 2))),
        "kxy_rms": float(np.sqrt(np.mean(kxy ** 2))),
        "kxx_mean": float(np.mean(kxx)),
        "kyy_mean": float(np.mean(kyy)),
        "kxy_mean": float(np.mean(kxy)),
    }
    if dof_per_node == 6:
        dw_dx = (
            coeff[1] + 2.0*coeff[3]*x + coeff[4]*y +
            3.0*coeff[6]*x*x + 2.0*coeff[7]*x*y + coeff[8]*y*y
        )
        dw_dy = (
            coeff[2] + coeff[4]*x + 2.0*coeff[5]*y +
            coeff[7]*x*x + 2.0*coeff[8]*x*y + 3.0*coeff[9]*y*y
        )
        theta_x = nodal[:, 3]
        theta_y = nodal[:, 4]
        gamma_xz = theta_y + dw_dx
        gamma_yz = theta_x - dw_dy
        result.update({
            "gamma_xz_rms": float(np.sqrt(np.mean(gamma_xz ** 2))),
            "gamma_yz_rms": float(np.sqrt(np.mean(gamma_yz ** 2))),
            "gamma_xz_mean": float(np.mean(gamma_xz)),
            "gamma_yz_mean": float(np.mean(gamma_yz)),
        })
    return result


def hill_mandel_residual(micro_energy, boundary_energy):
    """Return absolute and normalized Hill-Mandel energy residuals."""
    micro = float(micro_energy)
    boundary = float(boundary_energy)
    difference = micro - boundary
    return {
        "micro_energy": micro,
        "boundary_energy": boundary,
        "difference": difference,
        "relative": abs(difference) / max(abs(micro), abs(boundary), EPS),
    }
