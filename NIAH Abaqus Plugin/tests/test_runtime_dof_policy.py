import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "niah_core"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))


class RuntimeDofPolicyTests(unittest.TestCase):
    def test_all_preprocessing_shell_types_are_six_dof(self):
        from input_data import _make_prefix_and_dof

        for element_type in ("S3", "S3R", "S4", "S4R"):
            prefix, condition, dof = _make_prefix_and_dof("shell", element_type)
            self.assertEqual((prefix, condition, dof), ("shell_6dof_", 4, 6))

    def test_b31_remains_six_dof(self):
        from input_data import _make_prefix_and_dof

        self.assertEqual(
            _make_prefix_and_dof("shell", "B31"),
            ("shell_6dof_", 4, 6),
        )

    def test_solid_element_remains_three_dof(self):
        from input_data import _make_prefix_and_dof

        self.assertEqual(
            _make_prefix_and_dof("shell", "C3D8R"),
            ("shell_3dof_", 3, 3),
        )

    def test_six_dof_requires_ur_and_rm_fields(self):
        from boundary_ordering import require_rotational_field

        require_rotational_field({"U": object()}, 3, "UR")
        require_rotational_field({"RF": object()}, 3, "RM")

        with self.assertRaisesRegex(RuntimeError, "UR"):
            require_rotational_field({"U": object()}, 6, "UR")
        with self.assertRaisesRegex(RuntimeError, "RM"):
            require_rotational_field({"RF": object()}, 6, "RM")

    def test_aborted_or_locked_job_is_rejected_before_odb_open(self):
        from boundary_ordering import ensure_job_output_ready

        ensure_job_output_ready("OK_JOB", "COMPLETED", True, False)

        with self.assertRaisesRegex(RuntimeError, "ABORTED"):
            ensure_job_output_ready("BAD_JOB", "ABORTED", True, True)
        with self.assertRaisesRegex(RuntimeError, "lock file"):
            ensure_job_output_ready("LOCKED_JOB", None, True, True)
        with self.assertRaisesRegex(RuntimeError, "not found"):
            ensure_job_output_ready("NO_ODB_JOB", None, False, False)


if __name__ == "__main__":
    unittest.main()
