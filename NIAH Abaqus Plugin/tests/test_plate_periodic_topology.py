import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "niah_core"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))


class PlatePeriodicTopologyTests(unittest.TestCase):
    def test_homogeneous_free_surface_edge_pairs_close_both_z_planes(self):
        from boundary_ordering import plate_homogeneous_free_surface_edge_pairs

        groups = plate_homogeneous_free_surface_edge_pairs(
            fledge=[1], bledge=[2], bredge=[3], fredge=[4],
            ltedge=[5], rtedge=[6], rbedge=[7], lbedge=[8],
        )

        names = [g[0] for g in groups]
        self.assertEqual(
            names,
            ["FLEDGE-BLEDGE", "FREDGE-BREDGE", "LTEDGE-LBEDGE", "RTEDGE-RBEDGE"],
        )
        self.assertEqual(groups[0][1:], ("fledge", [1], "bledge", [2]))
        self.assertEqual(groups[3][1:], ("rtedge", [6], "rbedge", [7]))

    def test_shear_pure_edge_pairs_use_reduced_nonredundant_edges(self):
        from boundary_ordering import plate_shear_pure_edge_pairs

        groups_xz = plate_shear_pure_edge_pairs(
            "SHXZ",
            ftedge=[1], btedge=[2], fbedge=[3], bbedge=[4],
            fledge=[5], bledge=[6], bredge=[7], fredge=[8],
            ltedge=[9], rtedge=[10], rbedge=[11], lbedge=[12],
        )
        groups_yz = plate_shear_pure_edge_pairs(
            "SHYZ",
            ftedge=[1], btedge=[2], fbedge=[3], bbedge=[4],
            fledge=[5], bledge=[6], bredge=[7], fredge=[8],
            ltedge=[9], rtedge=[10], rbedge=[11], lbedge=[12],
        )

        self.assertEqual(
            [g[0] for g in groups_xz],
            ["BTEDGE-BBEDGE", "LTEDGE-LBEDGE", "RTEDGE-RBEDGE"],
        )
        self.assertEqual(
            [g[0] for g in groups_yz],
            ["FBEDGE-BBEDGE", "FLEDGE-BLEDGE", "FREDGE-BREDGE"],
        )

    def test_shear_corner_pairs_are_split_by_nonhomogeneous_direction(self):
        from boundary_ordering import plate_shear_corner_pairs

        self.assertEqual(
            plate_shear_corner_pairs("SHXZ"),
            [("c2", "c6"), ("c3", "c7")],
        )
        self.assertEqual(
            plate_shear_corner_pairs("SHYZ"),
            [("c5", "c6"), ("c8", "c7")],
        )


if __name__ == "__main__":
    unittest.main()
