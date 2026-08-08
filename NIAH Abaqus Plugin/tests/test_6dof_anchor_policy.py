from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "niah_core"


class SixDofAnchorPolicyTests(unittest.TestCase):
    def test_6dof_periodic_equations_include_rotations(self):
        source = (CORE / "PBC_constraint_builder.py").read_text(encoding="utf-8")

        self.assertIn(
            "for d in (1, 2, 3, 4, 5, 6):",
            source,
            msg="6DOF periodic equations must include UR1, UR2, and UR3",
        )

    def test_rigid_anchors_do_not_clamp_rotational_dofs(self):
        paths = [
            CORE / "def_run_pre_processing.py",
            CORE / "def_run_pre_processing_model.py",
            CORE / "Utility_function.py",
        ]

        for path in paths:
            source = path.read_text(encoding="utf-8")
            self.assertNotIn(
                "ur1=0.0, ur2=0.0, ur3=0.0",
                source,
                msg="%s clamps rotational DOFs at the rigid anchor" % path.name,
            )


if __name__ == "__main__":
    unittest.main()
