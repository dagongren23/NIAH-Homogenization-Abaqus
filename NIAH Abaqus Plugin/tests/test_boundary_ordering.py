import os
import sys
import unittest


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NIAH_CORE = os.path.join(ROOT, "niah_core")
if NIAH_CORE not in sys.path:
    sys.path.insert(0, NIAH_CORE)


class FakeValue(object):
    def __init__(self, node_label, data):
        self.nodeLabel = node_label
        self.data = data


class BoundaryOrderingTests(unittest.TestCase):
    def test_sort_pairs_by_negative_label(self):
        from boundary_ordering import sort_pairs_by_negative_label

        positive, negative = sort_pairs_by_negative_label(
            [101, 102, 103, 104],
            [4, 1, 3, 2],
        )

        self.assertEqual(positive, [102, 104, 103, 101])
        self.assertEqual(negative, [1, 2, 3, 4])

    def test_flatten_values_by_node_label_fills_translation_and_rotation(self):
        from boundary_ordering import flatten_values_by_node_label

        u_values = [
            FakeValue(3, (30.0, 31.0, 32.0)),
            FakeValue(1, (10.0, 11.0, 12.0)),
            FakeValue(2, (20.0, 21.0, 22.0)),
        ]
        ur_values = [
            FakeValue(2, (200.0, 201.0, 202.0)),
            FakeValue(1, (100.0, 101.0, 102.0)),
            FakeValue(3, (300.0, 301.0, 302.0)),
        ]

        result = flatten_values_by_node_label(
            u_values,
            labels=[1, 2, 3],
            dof_per_node=6,
            rotation_values=ur_values,
        )

        self.assertEqual(
            result.tolist(),
            [
                10.0, 11.0, 12.0, 100.0, 101.0, 102.0,
                20.0, 21.0, 22.0, 200.0, 201.0, 202.0,
                30.0, 31.0, 32.0, 300.0, 301.0, 302.0,
            ],
        )

    def test_flatten_values_by_node_label_uses_sorted_labels_by_default(self):
        from boundary_ordering import flatten_values_by_node_label

        u_values = [
            FakeValue(3, (30.0, 31.0, 32.0)),
            FakeValue(1, (10.0, 11.0, 12.0)),
            FakeValue(2, (20.0, 21.0, 22.0)),
        ]

        result = flatten_values_by_node_label(u_values, dof_per_node=3)

        self.assertEqual(
            result.tolist(),
            [
                10.0, 11.0, 12.0,
                20.0, 21.0, 22.0,
                30.0, 31.0, 32.0,
            ],
        )


if __name__ == "__main__":
    unittest.main()
