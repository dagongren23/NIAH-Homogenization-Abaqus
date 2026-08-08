# -*- coding: UTF-8 -*-
"""Regression tests for clean shear bases and chain-level fast resume."""
from __future__ import print_function

import io
import os
import sys
import tempfile
import unittest

import numpy as np


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORE = os.path.join(ROOT, 'niah_core')
if CORE not in sys.path:
    sys.path.insert(0, CORE)

from shear_resume import (
    build_shear_dependency_fingerprint,
    find_active_clean_shear_base,
    find_shear_chain_checkpoint,
    preserve_clean_shear_base,
    record_shear_chain_checkpoint,
)


def _write(path, content):
    with open(path, 'wb') as stream:
        if not isinstance(content, bytes):
            content = content.encode('utf-8')
        stream.write(content)


class ShearResumeTests(unittest.TestCase):
    def _work_place(self):
        # Intentionally retain test artifacts: project policy forbids deletion.
        return tempfile.mkdtemp(prefix='niah_shear_resume_')

    def test_dependency_fingerprint_covers_files_metadata_and_arrays(self):
        work_place = self._work_place()
        base_path = os.path.join(work_place, 'BASE.inp')
        _write(base_path, '*Heading\nBASE-A\n')
        metadata = {'mode': 'SHXZ', 'dof': 3}
        files = {'base': base_path}
        arrays = {'disp': np.array([1.0, 2.0], dtype=float)}

        original = build_shear_dependency_fingerprint(
            metadata, files, arrays
        )
        self.assertEqual(
            original,
            build_shear_dependency_fingerprint(metadata, files, arrays)
        )

        changed_array = build_shear_dependency_fingerprint(
            metadata,
            files,
            {'disp': np.array([1.0, 2.1], dtype=float)}
        )
        self.assertNotEqual(original, changed_array)

        _write(base_path, '*Heading\nBASE-B\n')
        changed_file = build_shear_dependency_fingerprint(
            metadata, files, arrays
        )
        self.assertNotEqual(original, changed_file)

        changed_metadata = build_shear_dependency_fingerprint(
            {'mode': 'SHYZ', 'dof': 3},
            files,
            arrays
        )
        self.assertNotEqual(changed_file, changed_metadata)

    def test_preprocessing_preserves_and_activates_immutable_clean_base(self):
        work_place = self._work_place()
        source = os.path.join(work_place, 'BASE_NOPBCX.inp')
        _write(source, '*Heading\nCLEAN-A\n')

        first_clean = preserve_clean_shear_base(
            work_place,
            'BASE_NOPBCX',
            source
        )
        self.assertTrue(os.path.isfile(first_clean))
        self.assertEqual(
            find_active_clean_shear_base(work_place, 'BASE_NOPBCX'),
            first_clean
        )

        _write(source, '*Heading\nCLEAN-B\n')
        second_clean = preserve_clean_shear_base(
            work_place,
            'BASE_NOPBCX',
            source
        )
        self.assertNotEqual(first_clean, second_clean)
        self.assertEqual(
            find_active_clean_shear_base(work_place, 'BASE_NOPBCX'),
            second_clean
        )
        with open(first_clean, 'rb') as stream:
            self.assertIn(b'CLEAN-A', stream.read())

        # A damaged latest immutable copy is rejected; the prior valid
        # activation remains usable without deleting any historical file.
        _write(second_clean, '*Heading\nDAMAGED\n')
        self.assertEqual(
            find_active_clean_shear_base(work_place, 'BASE_NOPBCX'),
            first_clean
        )

    def test_chain_checkpoint_reports_each_stage_without_rebuilding_input(self):
        work_place = self._work_place()
        odb_paths = []
        for stage in ('SH1', 'SH2', 'SH3'):
            job_name = 'CASE_' + stage + '_retry001'
            odb_path = os.path.join(work_place, job_name + '.odb')
            sta_path = os.path.join(work_place, job_name + '.sta')
            _write(odb_path, 'ODB-' + stage)
            _write(sta_path, 'THE ANALYSIS HAS COMPLETED SUCCESSFULLY\n')
            odb_paths.append(odb_path)

        record_shear_chain_checkpoint(
            work_place,
            'CASE',
            'abc123',
            odb_paths
        )
        found = find_shear_chain_checkpoint(
            work_place,
            'CASE',
            'abc123'
        )
        self.assertIsNotNone(found)
        self.assertTrue(found['stages']['SH1']['artifact_valid'])
        self.assertTrue(found['stages']['SH2']['artifact_valid'])
        self.assertTrue(found['stages']['SH3']['artifact_valid'])
        self.assertEqual(
            found['stages']['SH2']['actual_job_name'],
            'CASE_SH2_retry001'
        )

        _write(
            os.path.join(work_place, 'CASE_SH2_retry001.lck'),
            'locked'
        )
        found_locked = find_shear_chain_checkpoint(
            work_place,
            'CASE',
            'abc123'
        )
        self.assertFalse(found_locked['stages']['SH2']['artifact_valid'])
        self.assertTrue(found_locked['stages']['SH3']['artifact_valid'])

    def test_solver_fast_path_precedes_any_cae_model_reconstruction(self):
        utility_path = os.path.join(CORE, 'Utility_function.py')
        with io.open(utility_path, 'r', encoding='utf-8') as stream:
            source = stream.read()

        function_start = source.index('def _niah_share_one_case_inp(')
        function_end = source.index(
            '\ndef _compute_E_Dbar_from_EH6',
            function_start
        )
        shear_source = source[function_start:function_end]
        fast_path = shear_source.index('if all(reusable_stages.values()):')
        model_import = shear_source.index(
            'mdb.ModelFromInputFile('
        )
        self.assertLess(fast_path, model_import)
        self.assertIn(
            'skip CAE import, PBC reconstruction and INP generation',
            shear_source
        )
        self.assertIn(
            "if reusable_stages['SH2']:",
            shear_source
        )
        clean_lookup = shear_source.index(
            'find_active_clean_shear_base('
        )
        resume_guard = shear_source.rfind(
            'if resume_solver:',
            0,
            clean_lookup
        )
        self.assertGreaterEqual(resume_guard, 0)

    def test_both_preprocessing_modes_preserve_clean_bases(self):
        for relative_path in (
                'niah_core/def_run_pre_processing.py',
                'niah_core/def_run_pre_processing_model.py'):
            path = os.path.join(ROOT, relative_path)
            with io.open(path, 'r', encoding='utf-8') as stream:
                source = stream.read()
            self.assertIn('preserve_clean_shear_base(', source)
            self.assertIn(
                "runtime_input.get('resume_solver', False)",
                source
            )


if __name__ == '__main__':
    unittest.main()
