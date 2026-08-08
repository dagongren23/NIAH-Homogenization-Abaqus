import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "niah_core"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))


class HomogenizationMathTests(unittest.TestCase):
    def test_chain_assembly_uses_total_macro_plus_correction_fields(self):
        from homogenization_math import assemble_effective_stiffness_from_chain

        x1 = np.array([[1.0, 2.0], [0.5, 0.0]])
        x2 = np.array([[-0.25, 0.5], [0.0, 0.25]])
        f1 = np.array([[10.0, 4.0], [2.0, 1.0]])
        f2 = np.array([[-1.0, 3.0], [0.5, -0.5]])

        actual = assemble_effective_stiffness_from_chain(x1, x2, f1, f2, measure=2.0)
        expected = np.dot(x1 + x2, (f1 + f2).T) / 2.0

        np.testing.assert_allclose(actual, expected)

    def test_chain_assembly_rejects_zero_measure(self):
        from homogenization_math import assemble_effective_stiffness_from_chain

        with self.assertRaises(ValueError):
            assemble_effective_stiffness_from_chain(
                np.zeros((1, 1)),
                np.zeros((1, 1)),
                np.zeros((1, 1)),
                np.zeros((1, 1)),
                measure=0.0,
            )

    def test_chain_assembly_includes_all_beam_6dof_components(self):
        from homogenization_math import assemble_effective_stiffness_from_chain

        x1 = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 2.0]])
        x2 = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.5]])
        f1 = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 4.0]])
        f2 = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])

        actual = assemble_effective_stiffness_from_chain(x1, x2, f1, f2, measure=2.0)
        expected = np.array([[6.25]])

        np.testing.assert_allclose(actual, expected)

    def test_chain_assembly_has_one_correction_field_convention(self):
        import inspect
        from homogenization_math import assemble_effective_stiffness_from_chain

        parameters = inspect.signature(assemble_effective_stiffness_from_chain).parameters
        self.assertNotIn("correction_sign", parameters)


if __name__ == "__main__":
    unittest.main()
