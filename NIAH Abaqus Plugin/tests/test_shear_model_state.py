# -*- coding: UTF-8 -*-
"""Tests for idempotent reconstruction of imported shear base models."""
from __future__ import print_function

import io
import os
import sys
import unittest


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORE = os.path.join(ROOT, 'niah_core')
if CORE not in sys.path:
    sys.path.insert(0, CORE)

from shear_model_state import reset_imported_shear_base_model


class _Assembly(object):
    def __init__(self):
        self.sets = {'OLD-SET': object(), 'USER-SET': object()}
        self.features = {
            'RP-1': object(),
            'RP-27': object(),
            'PINGBAN_SOLID-1': object(),
        }


class _Model(object):
    def __init__(self):
        self.rootAssembly = _Assembly()
        self.constraints = {
            'OLD-EQUATION': object(),
            'OLD-COUPLING': object(),
        }
        self.boundaryConditions = {
            'BC-c7': object(),
            'Disp-BC-1': object(),
            'Disp-BC-1215': object(),
        }


class ShearModelStateTests(unittest.TestCase):
    def test_reset_removes_accumulated_solver_state(self):
        model = _Model()

        removed = reset_imported_shear_base_model(model)

        self.assertEqual(model.rootAssembly.sets, {})
        self.assertEqual(model.constraints, {})
        self.assertEqual(model.boundaryConditions, {})
        self.assertNotIn('RP-1', model.rootAssembly.features)
        self.assertNotIn('RP-27', model.rootAssembly.features)
        self.assertIn('PINGBAN_SOLID-1', model.rootAssembly.features)
        self.assertEqual(
            removed,
            {
                'sets': 2,
                'constraints': 2,
                'boundary_conditions': 3,
                'reference_point_features': 2,
            }
        )

    def test_reset_is_idempotent(self):
        model = _Model()
        reset_imported_shear_base_model(model)

        removed_again = reset_imported_shear_base_model(model)

        self.assertEqual(
            removed_again,
            {
                'sets': 0,
                'constraints': 0,
                'boundary_conditions': 0,
                'reference_point_features': 0,
            }
        )

    def test_solver_writes_pbc_to_derived_input_not_base_input(self):
        utility_path = os.path.join(CORE, 'Utility_function.py')
        with io.open(utility_path, 'r', encoding='utf-8') as stream:
            source = stream.read()

        self.assertIn(
            "shear_pbc_job_name = BASE_JOB_REALNOPBC + '_SHEAR_PBC'",
            source
        )
        self.assertIn(
            "_inject_block_into_step(shear_pbc_inp, run_inp5_path",
            source
        )
        self.assertNotIn(
            "_write_job_input_from_model(BASE_JOB_REALNOPBC, "
            "model_name_sh1",
            source
        )


if __name__ == '__main__':
    unittest.main()
