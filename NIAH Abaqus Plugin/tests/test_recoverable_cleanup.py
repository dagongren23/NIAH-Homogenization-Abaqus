"""Tests for the recoverable Abaqus-output cleanup policy."""

from __future__ import absolute_import

import os
import uuid
import unittest

from niah_core.def_run_correct import run_correct


class RecoverableCleanupTests(unittest.TestCase):
    def _new_workbench(self):
        root = os.path.abspath(os.path.join(
            os.path.dirname(__file__),
            os.pardir,
            'niah_workbench',
            'test_artifacts',
            'cleanup_%s' % uuid.uuid4().hex
        ))
        os.makedirs(root)
        return root

    def _write_text(self, path, value):
        parent = os.path.dirname(path)
        if not os.path.isdir(parent):
            os.makedirs(parent)
        with open(path, 'w') as stream:
            stream.write(value)

    def test_recursive_mode_archives_generated_files_and_preserves_odb(self):
        workbench = self._new_workbench()
        generated = os.path.join(workbench, 'nested', 'case.inp')
        retained = os.path.join(workbench, 'nested', 'case.odb')
        self._write_text(generated, 'input')
        self._write_text(retained, 'output database placeholder')

        archived = run_correct(1, {
            'workbench_path': workbench,
            'plugin_root': os.path.dirname(workbench)
        })

        self.assertEqual(1, len(archived))
        self.assertFalse(os.path.exists(generated))
        self.assertTrue(os.path.isfile(archived[0]))
        self.assertTrue(os.path.isfile(retained))
        self.assertIn(os.path.join('delete', 'run_outputs'), archived[0])

    def test_top_level_mode_does_not_archive_nested_files(self):
        workbench = self._new_workbench()
        top_level = os.path.join(workbench, 'job.log')
        nested = os.path.join(workbench, 'nested', 'job.log')
        self._write_text(top_level, 'top')
        self._write_text(nested, 'nested')

        archived = run_correct(2, {
            'workbench_path': workbench,
            'plugin_root': os.path.dirname(workbench)
        })

        self.assertEqual(1, len(archived))
        self.assertFalse(os.path.exists(top_level))
        self.assertTrue(os.path.isfile(nested))

    def test_resume_mode_preserves_input_and_status_checkpoints(self):
        workbench = self._new_workbench()
        input_path = os.path.join(workbench, 'job.inp')
        status_path = os.path.join(workbench, 'job.sta')
        message_path = os.path.join(workbench, 'job.msg')
        self._write_text(input_path, 'input')
        self._write_text(status_path, 'status')
        self._write_text(message_path, 'message')

        archived = run_correct(1, {
            'workbench_path': workbench,
            'plugin_root': os.path.dirname(workbench),
            'resume_solver': True
        })

        self.assertEqual(1, len(archived))
        self.assertTrue(os.path.isfile(input_path))
        self.assertTrue(os.path.isfile(status_path))
        self.assertFalse(os.path.exists(message_path))


if __name__ == '__main__':
    unittest.main()
