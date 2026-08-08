# -*- coding: utf-8 -*-
from __future__ import division

import unittest

import numpy as np

from tests.shear_diagnostics.shear_diag_math import (
    chain_energy_decomposition,
    fit_transverse_cubic,
    full_shear_energy_diagnostic,
    hill_mandel_residual,
    mutual_work_matrix,
    periodic_pair_residuals,
    split_generalized_work,
    traction_pair_residuals,
)


class ShearDiagnosticMathTests(unittest.TestCase):
    def test_chain_energy_expansion_closes(self):
        u1 = np.array([1.0, 2.0, 3.0])
        u2 = np.array([-0.5, 0.25, 1.0])
        f1 = np.array([4.0, -2.0, 1.0])
        f2 = np.array([0.5, 1.5, -3.0])
        result = chain_energy_decomposition(u1, u2, f1, f2, 3)
        self.assertAlmostEqual(result["closure_abs"], 0.0, places=14)
        self.assertAlmostEqual(
            result["combined_half"], 0.5 * result["combined_raw"]
        )

    def test_six_dof_work_is_split_without_losing_terms(self):
        u = np.array([1, 2, 3, 4, 5, 6], dtype=float)
        f = np.array([6, 5, 4, 3, 2, 1], dtype=float)
        result = split_generalized_work(u, f, 6)
        self.assertAlmostEqual(result["translation_raw"], 28.0)
        self.assertAlmostEqual(result["rotation_raw"], 28.0)
        self.assertAlmostEqual(result["total_raw"], np.dot(u, f))

    def test_mutual_work_exposes_reciprocity_error(self):
        u1 = np.array([1.0, 0.0])
        u2 = np.array([0.0, 1.0])
        f1 = np.array([2.0, 3.0])
        f2 = np.array([4.0, 5.0])
        result = mutual_work_matrix([u1, u2], [f1, f2])
        self.assertAlmostEqual(result["raw"][0, 1], 4.0)
        self.assertAlmostEqual(result["raw"][1, 0], 3.0)
        self.assertGreater(result["reciprocity_relative"], 0.0)

    def test_full_matrix_uses_d12_bending_cross_term(self):
        dbar = np.array([
            [2.0, 0.75, 0.0],
            [0.75, 3.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        result = full_shear_energy_diagnostic(
            np.array([2.0, 0.0]), np.array([4.0, 0.0]),
            np.array([0.0, 3.0]), np.array([0.0, 5.0]),
            dbar, 2.0, 4.0,
        )
        self.assertAlmostEqual(
            result["bending_work"][0, 1], 0.75 * 2.0 * 4.0 / 12.0
        )
        np.testing.assert_allclose(
            result["work_symmetric"], result["work_symmetric"].T
        )

    def test_periodic_and_traction_pair_residuals(self):
        values = np.array([[2.0, 3.0], [1.0, 1.0]])
        kinematic = periodic_pair_residuals(
            values, [(1, 2)], 2, jumps=np.array([[1.0, 2.0]])
        )
        self.assertAlmostEqual(kinematic["max_abs"], 0.0)
        traction = traction_pair_residuals(
            np.array([[2.0, -3.0], [-2.0, 3.0]]), [(1, 2)]
        )
        self.assertAlmostEqual(traction["max_abs"], 0.0)

    def test_cubic_fit_recovers_lcxz_curvature_without_spurious_kyy(self):
        xs = np.linspace(-1.0, 1.0, 7)
        ys = np.linspace(-0.5, 0.5, 5)
        coords = np.array([(x, y, 0.0) for x in xs for y in ys])
        length = 2.0
        vector = np.zeros(coords.shape[0] * 6)
        vector[2::6] = -(coords[:, 0] ** 3) / (6.0 * length)
        vector[4::6] = (coords[:, 0] ** 2) / (2.0 * length)
        result = fit_transverse_cubic(coords, vector, 6)
        self.assertLess(result["w_relative_rmse"], 1.0e-12)
        self.assertGreater(result["kxx_rms"], 0.0)
        self.assertLess(result["kyy_rms"], 1.0e-12)
        self.assertLess(result["kxy_rms"], 1.0e-12)
        self.assertLess(result["gamma_xz_rms"], 1.0e-12)

    def test_hill_mandel_residual(self):
        result = hill_mandel_residual(10.0, 9.0)
        self.assertAlmostEqual(result["relative"], 0.1)


if __name__ == "__main__":
    unittest.main()
