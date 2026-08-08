import ast
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "niah_core"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))


def _matrix_from_physical_relations(relations, dof_per_node):
    nodes = sorted(set(label for pair in relations for label in pair))
    columns = {
        (label, dof): index
        for index, (label, dof) in enumerate(
            (label, dof)
            for label in nodes
            for dof in range(dof_per_node)
        )
    }
    matrix = np.zeros((len(relations) * dof_per_node, len(columns)))
    row = 0
    for label_a, label_b in relations:
        for dof in range(dof_per_node):
            matrix[row, columns[(label_a, dof)]] = 1.0
            matrix[row, columns[(label_b, dof)]] = -1.0
            row += 1
    return matrix


def _read_inp_nodes(path):
    nodes = {}
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
            nodes[int(fields[0])] = tuple(float(value) for value in fields[1:4])
    return nodes


def _classify_honeycomb_boundaries(nodes, tolerance=1.0e-3):
    coordinates = list(nodes.values())
    mins = tuple(min(point[index] for point in coordinates) for index in range(3))
    maxs = tuple(max(point[index] for point in coordinates) for index in range(3))

    def near(value, target):
        return abs(value - target) <= tolerance

    def far(value, target):
        return abs(value - target) > tolerance

    xmin, ymin, zmin = mins
    xmax, ymax, zmax = maxs
    predicates = {
        "corners": lambda x, y, z: (
            (near(x, xmin) or near(x, xmax)) and
            (near(y, ymin) or near(y, ymax)) and
            (near(z, zmin) or near(z, zmax))
        ),
        "ftedge": lambda x, y, z: near(x, xmax) and near(y, ymax) and far(z, zmax) and far(z, zmin),
        "fbedge": lambda x, y, z: near(x, xmax) and near(y, ymin) and far(z, zmax) and far(z, zmin),
        "btedge": lambda x, y, z: near(x, xmin) and near(y, ymax) and far(z, zmax) and far(z, zmin),
        "bbedge": lambda x, y, z: near(x, xmin) and near(y, ymin) and far(z, zmax) and far(z, zmin),
        "fledge": lambda x, y, z: near(x, xmax) and near(z, zmax) and far(y, ymax) and far(y, ymin),
        "fredge": lambda x, y, z: near(x, xmax) and near(z, zmin) and far(y, ymax) and far(y, ymin),
        "bledge": lambda x, y, z: near(x, xmin) and near(z, zmax) and far(y, ymax) and far(y, ymin),
        "bredge": lambda x, y, z: near(x, xmin) and near(z, zmin) and far(y, ymax) and far(y, ymin),
        "ltedge": lambda x, y, z: near(z, zmax) and near(y, ymax) and far(x, xmax) and far(x, xmin),
        "lbedge": lambda x, y, z: near(z, zmax) and near(y, ymin) and far(x, xmax) and far(x, xmin),
        "rtedge": lambda x, y, z: near(z, zmin) and near(y, ymax) and far(x, xmax) and far(x, xmin),
        "rbedge": lambda x, y, z: near(z, zmin) and near(y, ymin) and far(x, xmax) and far(x, xmin),
        "fronts": lambda x, y, z: near(x, xmax) and far(y, ymax) and far(y, ymin) and far(z, zmax) and far(z, zmin),
        "backs": lambda x, y, z: near(x, xmin) and far(y, ymax) and far(y, ymin) and far(z, zmax) and far(z, zmin),
        "tops": lambda x, y, z: near(y, ymax) and far(x, xmax) and far(x, xmin) and far(z, zmax) and far(z, zmin),
        "bots": lambda x, y, z: near(y, ymin) and far(x, xmax) and far(x, xmin) and far(z, zmax) and far(z, zmin),
        "frontbc": lambda x, y, z: near(x, xmax),
        "backbc": lambda x, y, z: near(x, xmin),
        "topbc": lambda x, y, z: near(y, ymax),
        "botbc": lambda x, y, z: near(y, ymin),
    }
    return {
        name: [
            label for label, point in nodes.items()
            if predicate(*point)
        ]
        for name, predicate in predicates.items()
    }


def _pair_labels_by_coordinates(nodes, positive, negative, coordinate_indices, tolerance=1.0e-3):
    negative_by_key = {}
    scale = 1.0 / tolerance
    for label in negative:
        key = tuple(int(round(nodes[label][index] * scale)) for index in coordinate_indices)
        negative_by_key[key] = label

    relations = []
    for label in positive:
        key = tuple(int(round(nodes[label][index] * scale)) for index in coordinate_indices)
        relations.append((label, negative_by_key[key]))
    return relations


def _load_source_function(path, function_name):
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    function_node = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == function_name
    )
    function_module = ast.Module(body=[function_node], type_ignores=[])
    namespace = {}
    exec(compile(ast.fix_missing_locations(function_module),
                 str(path), "exec"), namespace)
    return namespace[function_name]


class OptionalBoundaryTopologyTests(unittest.TestCase):
    def test_homogeneous_plate_accepts_no_geometric_corners(self):
        from boundary_ordering import plate_homogeneous_corner_pairs

        corners = {"c%d" % index: [] for index in range(1, 9)}
        self.assertEqual(plate_homogeneous_corner_pairs(corners), [])

    def test_solid_plate_keeps_independent_corner_trees_on_both_z_surfaces(self):
        from boundary_ordering import plate_homogeneous_corner_pairs

        corners = {
            "c1": [1], "c2": [2], "c3": [3], "c4": [4],
            "c5": [5], "c6": [6], "c7": [7], "c8": [8],
        }
        relations = plate_homogeneous_corner_pairs(corners)

        self.assertEqual(len(relations), 6)
        self.assertEqual([item[0] for item in relations], ["ZMAX"] * 3 + ["ZMIN"] * 3)

    def test_shell_corner_aliases_are_deduplicated_by_physical_node(self):
        from boundary_ordering import plate_homogeneous_corner_pairs

        corners = {
            "c1": [11], "c2": [12], "c3": [12], "c4": [11],
            "c5": [15], "c6": [16], "c7": [16], "c8": [15],
        }
        relations = plate_homogeneous_corner_pairs(corners)
        physical_relations = [(item[3], item[4]) for item in relations]

        self.assertEqual(physical_relations, [(11, 12), (12, 16), (16, 15)])
        for dof_per_node in (3, 6):
            matrix = _matrix_from_physical_relations(
                physical_relations, dof_per_node
            )
            self.assertEqual(np.linalg.matrix_rank(matrix), matrix.shape[0])

    def test_missing_intermediate_corners_form_tree_only_from_existing_nodes(self):
        from boundary_ordering import plate_homogeneous_corner_pairs

        corners = {
            "c1": [21], "c2": [], "c6": [26], "c5": [],
            "c4": [], "c3": [], "c7": [], "c8": [],
        }
        relations = plate_homogeneous_corner_pairs(corners)

        self.assertEqual(
            relations,
            [("ZMAX", "c1", "c6", 21, 26)],
        )

    def test_3d_corner_chain_uses_only_existing_physical_nodes(self):
        from boundary_ordering import existing_corner_chain_pairs

        corners = {
            "c1": [21], "c2": [], "c3": [23], "c4": [],
            "c5": [25], "c6": [], "c7": [], "c8": [28],
        }
        chain = ("c1", "c5", "c4", "c8", "c2", "c6", "c3", "c7")
        relations = existing_corner_chain_pairs(chain, corners)

        self.assertEqual(
            relations,
            [
                ("c1", "c5", 21, 25),
                ("c5", "c8", 25, 28),
                ("c8", "c3", 28, 23),
            ],
        )
        matrix = _matrix_from_physical_relations(
            [(item[2], item[3]) for item in relations], 3
        )
        self.assertEqual(np.linalg.matrix_rank(matrix), matrix.shape[0])

    def test_3d_corner_chain_accepts_no_geometric_corners(self):
        from boundary_ordering import existing_corner_chain_pairs

        corners = {"c%d" % index: [] for index in range(1, 9)}
        chain = ("c1", "c5", "c4", "c8", "c2", "c6", "c3", "c7")

        self.assertEqual(existing_corner_chain_pairs(chain, corners), [])

    def test_ambiguous_corner_with_multiple_nodes_is_rejected(self):
        from boundary_ordering import plate_homogeneous_corner_pairs

        corners = {"c%d" % index: [] for index in range(1, 9)}
        corners["c1"] = [1, 2]
        with self.assertRaisesRegex(ValueError, "contains 2 nodes"):
            plate_homogeneous_corner_pairs(corners)

    def test_shear_skips_pairs_when_both_corners_are_absent(self):
        from boundary_ordering import plate_existing_shear_corner_pairs

        corners = {"c%d" % index: [] for index in range(1, 9)}
        for mode in ("SHXZ", "SHYZ"):
            self.assertEqual(
                plate_existing_shear_corner_pairs(mode, corners),
                [],
            )

    def test_shear_shell_aliases_do_not_duplicate_physical_equations(self):
        from boundary_ordering import plate_existing_shear_corner_pairs

        corners = {
            "c1": [11], "c2": [12], "c3": [12], "c4": [11],
            "c5": [15], "c6": [16], "c7": [16], "c8": [15],
        }
        relations = plate_existing_shear_corner_pairs("SHXZ", corners)

        self.assertEqual(relations, [("c2", "c6", 12, 16)])

    def test_one_sided_shear_corner_pair_is_a_periodicity_error(self):
        from boundary_ordering import plate_existing_shear_corner_pairs

        corners = {"c%d" % index: [] for index in range(1, 9)}
        corners["c2"] = [12]
        with self.assertRaisesRegex(ValueError, "incomplete"):
            plate_existing_shear_corner_pairs("SHXZ", corners)

    def test_coincident_shell_edge_aliases_are_deduplicated(self):
        from boundary_ordering import unique_periodic_pair_relations

        groups = [
            ("ZMAX-X", "fledge", [31, 32], "bledge", [41, 42]),
            ("ZMIN-X", "fredge", [31, 32], "bredge", [41, 42]),
            ("EMPTY", "a", [], "b", []),
        ]
        relations = unique_periodic_pair_relations(groups)

        self.assertEqual(
            relations,
            [
                ("ZMAX-X", "fledge", 31, "bledge", 41),
                ("ZMAX-X", "fledge", 32, "bledge", 42),
            ],
        )

    def test_pair_length_mismatch_is_not_silently_truncated(self):
        from boundary_ordering import unique_periodic_pair_relations

        groups = [("BAD", "left", [1, 2], "right", [3])]
        with self.assertRaisesRegex(ValueError, "length mismatch"):
            unique_periodic_pair_relations(groups)

    def test_correspondence_check_executes_with_all_four_vertical_edges(self):
        check = _load_source_function(
            CORE / "PBC_constraint_builder.py",
            "check_point_correspondence",
        )
        argument_names = (
            "frontsxyz", "backsxyz", "topsxyz", "botsxyz",
            "leftsxyz", "rightsxyz",
            "ftedgexyz", "btedgexyz", "fbedgexyz", "bbedgexyz",
            "fledgexyz", "bledgexyz", "bredgexyz", "fredgexyz",
            "ltedgexyz", "rtedgexyz", "rbedgexyz", "lbedgexyz",
            "frontbcxyz", "backbcxyz", "topbcxyz", "botbcxyz",
            "leftbcxyz", "rightbcxyz",
        )
        empty_groups = {name: {} for name in argument_names}

        self.assertFalse(check(False, **empty_groups))

        mismatched_groups = dict(empty_groups)
        mismatched_groups["fbedgexyz"] = {17: (0.0, 0.0, 0.5)}
        self.assertTrue(check(False, **mismatched_groups))

    def test_main_delegates_boundary_work_to_pure_python_builder(self):
        source = (CORE / "PBC_constraint_builder.py").read_text(
            encoding="utf-8"
        )
        tree = ast.parse(source)
        main_function = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and
            node.name == "main_apply_pbc_constraint"
        )
        called_names = {
            node.func.id
            for node in ast.walk(main_function)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        self.assertIn("build_ordered_boundary_sets", called_names)
        self.assertNotIn("check_point_correspondence", called_names)
        self.assertNotIn("check_correspondence_transform_lable", called_names)

    def test_builder_passes_actual_corner_topology_to_all_pbc_paths(self):
        source = (CORE / "PBC_constraint_builder.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        definitions = {
            node.name: node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
        }
        pbc_functions = (
            "create_3D_periodic_constraints_equation",
            "create_3D_periodic_constraints_equation_6dof",
            "create_mindlin_periodic_constrains_equation_3dof",
            "create_mindlin_periodic_constrains_equation_6dof",
            "create_mindlin_nonhomogeneous_periodic_constraints_equation",
            "create_mindlin_nonhomogeneous_periodic_constraints_equation_6dof",
        )

        for function_name in pbc_functions:
            parameter_names = [
                item.arg for item in definitions[function_name].args.args
            ]
            self.assertIn("corner_labels", parameter_names, function_name)
        self.assertIn("arguments.append(ordered_sets)", source)
        self.assertEqual(source.count("corner_labels=ordered_sets"), 2)
        self.assertIn("if node_labels:", source)
        self.assertIn("optional empty boundary sets were skipped", source)
        self.assertNotIn("'E-1-c7-RP1'", source)
        self.assertNotIn("add_eq('E-c7-RP1'", source)
        self.assertNotIn(
            "_tie_corner_plane_6dof(('c1', 'c2', 'c6', 'c5')",
            source,
        )

    def test_gui_selected_node_is_the_real_translational_anchor(self):
        preprocessing_paths = (
            CORE / "def_run_pre_processing.py",
            CORE / "def_run_pre_processing_model.py",
            CORE / "Utility_function.py",
        )
        for path in preprocessing_paths:
            source = path.read_text(encoding="utf-8")
            self.assertIn(
                "_build_rigid_anchor_region(",
                source,
                path.name,
            )
            self.assertIn("u1=0.0, u2=0.0, u3=0.0", source, path.name)

        utility_source = (CORE / "Utility_function.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("nodes.sequenceFromLabels(labels=(node_label,))",
                      utility_source)
        self.assertNotIn("_nodes[node_bd - 1:node_bd]", utility_source)
        self.assertIn("Check the GUI rigid-node coordinates/tolerance",
                      utility_source)

    def test_hexagonal_honeycomb_real_mesh_has_full_rank_without_corners(self):
        nodes = _read_inp_nodes(ROOT / "Hexagonal_honeycomb_shell.inp")
        boundary = _classify_honeycomb_boundaries(nodes)

        self.assertEqual(len(nodes), 1540)
        self.assertEqual(len(boundary["corners"]), 0)
        for name in ("ftedge", "fbedge", "btedge", "bbedge"):
            self.assertEqual(len(boundary[name]), 0)

        expected_counts = {
            "fronts": 9, "backs": 9,
            "tops": 189, "bots": 189,
            "fledge": 1, "bledge": 1,
            "fredge": 1, "bredge": 1,
            "ltedge": 21, "lbedge": 21,
            "rtedge": 21, "rbedge": 21,
        }
        for name, expected in expected_counts.items():
            self.assertEqual(len(boundary[name]), expected, name)

        homogeneous_relations = []
        for positive, negative, coordinate_indices in (
            ("tops", "bots", (0, 2)),
            ("fronts", "backs", (1, 2)),
            ("fledge", "bledge", (1,)),
            ("fredge", "bredge", (1,)),
            ("ltedge", "lbedge", (0,)),
            ("rtedge", "rbedge", (0,)),
        ):
            homogeneous_relations.extend(
                _pair_labels_by_coordinates(
                    nodes,
                    boundary[positive],
                    boundary[negative],
                    coordinate_indices,
                )
            )

        self.assertEqual(len(homogeneous_relations), 242)
        matrix = _matrix_from_physical_relations(homogeneous_relations, 1)
        self.assertEqual(np.linalg.matrix_rank(matrix), matrix.shape[0])
        self.assertEqual(
            np.linalg.matrix_rank(matrix) * 6,
            len(homogeneous_relations) * 6,
            "The corresponding 6-DOF equation matrix must also have full row rank",
        )

        shear_specs = {
            "SHXZ": (
                ("frontbc", "backbc", (1, 2)),
                ("tops", "bots", (0, 2)),
                ("ltedge", "lbedge", (0,)),
                ("rtedge", "rbedge", (0,)),
            ),
            "SHYZ": (
                ("topbc", "botbc", (0, 2)),
                ("fronts", "backs", (1, 2)),
                ("fledge", "bledge", (1,)),
                ("fredge", "bredge", (1,)),
            ),
        }
        for mode, pair_specs in shear_specs.items():
            shear_relations = []
            for positive, negative, coordinate_indices in pair_specs:
                shear_relations.extend(
                    _pair_labels_by_coordinates(
                        nodes,
                        boundary[positive],
                        boundary[negative],
                        coordinate_indices,
                    )
                )

            self.assertEqual(len(shear_relations), 242, mode)
            shear_matrix = _matrix_from_physical_relations(
                shear_relations, 1
            )
            self.assertEqual(
                np.linalg.matrix_rank(shear_matrix),
                shear_matrix.shape[0],
                "%s relations from the real honeycomb mesh are redundant" % mode,
            )
            self.assertEqual(
                np.linalg.matrix_rank(shear_matrix) * 6,
                len(shear_relations) * 6,
            )


if __name__ == "__main__":
    unittest.main()
