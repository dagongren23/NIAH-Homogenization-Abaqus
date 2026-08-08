import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "niah_core"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))


def _read_inp_nodes(path):
    nodes = []
    in_node_block = False
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line.lower().startswith("*node"):
            in_node_block = True
            continue
        if in_node_block and line.startswith("*"):
            break
        if in_node_block and line and not line.startswith("**"):
            fields = [field.strip() for field in line.split(",")]
            nodes.append((
                int(fields[0]),
                tuple(float(value) for value in fields[1:4]),
            ))
    return nodes


class PeriodicMeshTests(unittest.TestCase):
    def test_cube_boundary_sets_are_complete_and_paired(self):
        from periodic_mesh import build_ordered_boundary_sets

        result = build_ordered_boundary_sets(
            _read_inp_nodes(ROOT / "cube_mesh.inp"),
            tolerance=1.0e-6,
        )
        ordered = result["ordered_sets"]
        for index in range(1, 9):
            self.assertEqual(len(ordered["c%d" % index]), 1)
        for positive, negative in (
            ("fronts", "backs"), ("tops", "bots"),
            ("lefts", "rights"), ("frontbc", "backbc"),
            ("topbc", "botbc"), ("leftbc", "rightbc"),
        ):
            self.assertEqual(len(ordered[positive]), len(ordered[negative]))

    def test_honeycomb_keeps_optional_corners_and_vertical_edges_empty(self):
        from periodic_mesh import build_ordered_boundary_sets

        result = build_ordered_boundary_sets(
            _read_inp_nodes(ROOT / "Hexagonal_honeycomb_shell.inp"),
            tolerance=1.0e-3,
        )
        ordered = result["ordered_sets"]
        for name in tuple("c%d" % index for index in range(1, 9)) + (
                "ftedge", "fbedge", "btedge", "bbedge"):
            self.assertEqual(ordered[name], [], name)
        expected_counts = {
            "fronts": 9, "backs": 9,
            "tops": 189, "bots": 189,
            "fledge": 1, "bledge": 1,
            "fredge": 1, "bredge": 1,
            "ltedge": 21, "lbedge": 21,
            "rtedge": 21, "rbedge": 21,
        }
        for name, expected in expected_counts.items():
            self.assertEqual(len(ordered[name]), expected, name)

    def test_coordinate_matching_is_deterministic(self):
        from periodic_mesh import pair_regions

        positive = {
            20: (1.0, 1.0, 0.0),
            10: (1.0, 0.0, 0.0),
        }
        negative = {
            2: (0.0, 1.0, 0.0),
            1: (0.0, 0.0, 0.0),
        }
        self.assertEqual(
            pair_regions(positive, negative, (1, 2), 1.0e-6),
            ([10, 20], [1, 2]),
        )

    def test_count_mismatch_raises_clear_error(self):
        from periodic_mesh import PeriodicMeshError, pair_regions

        with self.assertRaisesRegex(PeriodicMeshError, "node-count mismatch"):
            pair_regions(
                {1: (1.0, 0.0, 0.0)},
                {},
                (1, 2),
                1.0e-6,
                "front", "back",
            )


if __name__ == "__main__":
    unittest.main()
