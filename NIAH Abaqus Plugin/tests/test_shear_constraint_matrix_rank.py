import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "niah_core"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))


def _node_name(prefix, label):
    return "%s%s" % (prefix, label)


def _group_relations(groups):
    relations = []
    for _, prefix_a, labels_a, prefix_b, labels_b in groups:
        relations.extend(
            (_node_name(prefix_a, label_a), _node_name(prefix_b, label_b))
            for label_a, label_b in zip(labels_a, labels_b)
        )
    return relations


def _constraint_matrix(relations, dof_per_node):
    nodes = sorted(set(node for pair in relations for node in pair))
    columns = {
        (node, dof): index
        for index, (node, dof) in enumerate(
            (node, dof)
            for node in nodes
            for dof in range(dof_per_node)
        )
    }
    matrix = np.zeros((len(relations) * dof_per_node, len(columns)))
    row = 0
    for node_a, node_b in relations:
        for dof in range(dof_per_node):
            matrix[row, columns[(node_a, dof)]] = 1.0
            matrix[row, columns[(node_b, dof)]] = -1.0
            row += 1
    return matrix


def _representative_relations(mode):
    from boundary_ordering import (
        plate_shear_corner_pairs,
        plate_shear_pure_edge_pairs,
    )

    edge_groups = plate_shear_pure_edge_pairs(
        mode,
        ftedge=[101, 105, 109],
        btedge=[102, 106, 110],
        fbedge=[103, 107, 111],
        bbedge=[104, 108, 112],
        fledge=[201], bledge=[202], bredge=[203], fredge=[204],
        ltedge=[301], rtedge=[302], rbedge=[303], lbedge=[304],
    )

    if mode == "SHXZ":
        # The non-homogeneous x-face pairs include vertical edges and corners.
        primary = [
            ("fledge201", "bledge202"),
            ("fredge204", "bredge203"),
            ("c1", "c2"), ("c5", "c6"),
            ("c4", "c3"), ("c8", "c7"),
            ("front_face", "back_face"),
        ]
        primary[0:0] = [
            ("ftedge%d" % front, "btedge%d" % back)
            for front, back in zip([101, 105, 109], [102, 106, 110])
        ]
        primary[3:3] = [
            ("fbedge%d" % front, "bbedge%d" % back)
            for front, back in zip([103, 107, 111], [104, 108, 112])
        ]
        complementary_face = [("top_face", "bottom_face")]
    else:
        # The non-homogeneous y-face pairs include vertical edges and corners.
        primary = [
            ("ltedge301", "lbedge304"),
            ("rtedge302", "rbedge303"),
            ("c1", "c5"), ("c2", "c6"),
            ("c4", "c8"), ("c3", "c7"),
            ("top_face", "bottom_face"),
        ]
        primary[0:0] = [
            ("ftedge%d" % top, "fbedge%d" % bottom)
            for top, bottom in zip([101, 105, 109], [103, 107, 111])
        ]
        primary[3:3] = [
            ("btedge%d" % top, "bbedge%d" % bottom)
            for top, bottom in zip([102, 106, 110], [104, 108, 112])
        ]
        complementary_face = [("front_face", "back_face")]

    return (
        primary
        + complementary_face
        + _group_relations(edge_groups)
        + plate_shear_corner_pairs(mode)
    )


class ShearConstraintMatrixRankTests(unittest.TestCase):
    def test_full_shxz_and_shyz_constraint_matrices_have_full_row_rank(self):
        for mode in ("SHXZ", "SHYZ"):
            relations = _representative_relations(mode)
            for dof_per_node in (3, 6):
                matrix = _constraint_matrix(relations, dof_per_node)
                self.assertEqual(
                    np.linalg.matrix_rank(matrix),
                    matrix.shape[0],
                    "%s %d-DOF equations contain a redundant row" %
                    (mode, dof_per_node),
                )

    def test_equation_order_never_reuses_an_eliminated_first_dof(self):
        for mode in ("SHXZ", "SHYZ"):
            eliminated_nodes = set()
            for node_a, node_b in _representative_relations(mode):
                self.assertNotIn(
                    node_a,
                    eliminated_nodes,
                    "%s reuses eliminated node %s as a later equation pivot" %
                    (mode, node_a),
                )
                self.assertNotIn(
                    node_b,
                    eliminated_nodes,
                    "%s references previously eliminated node %s" %
                    (mode, node_b),
                )
                eliminated_nodes.add(node_a)

    def test_previous_closed_corner_loop_is_rank_deficient(self):
        closed_loop = [
            ("c1", "c2"),
            ("c2", "c6"),
            ("c6", "c5"),
            ("c5", "c1"),
        ]
        matrix = _constraint_matrix(closed_loop, 3)
        self.assertLess(np.linalg.matrix_rank(matrix), matrix.shape[0])

    def test_old_shxz_positive_side_pivots_reproduce_five_missing_dofs(self):
        current_relations = _representative_relations("SHXZ")
        primary_count = 13
        primary_eliminated = set(
            node_a for node_a, _ in current_relations[:primary_count]
        )
        old_positive_side_relations = [
            ("ftedge101", "fbedge103"),
            ("ftedge105", "fbedge107"),
            ("ftedge109", "fbedge111"),
            ("c1", "c5"),
            ("c4", "c8"),
        ]
        repeated = [
            node_a
            for node_a, _ in old_positive_side_relations
            if node_a in primary_eliminated
        ]
        self.assertEqual(len(repeated), 5)


if __name__ == "__main__":
    unittest.main()
