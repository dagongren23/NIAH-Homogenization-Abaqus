# -*- coding: UTF-8 -*-
"""Ensure filesystem paths are converted at every Abaqus API boundary."""
from __future__ import print_function

import ast
import io
import os
import sys
import unittest
from unittest import mock


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORE = os.path.join(ROOT, 'niah_core')
if CORE not in sys.path:
    sys.path.insert(0, CORE)

import job_resume


PRODUCTION_PATH_FILES = (
    'niah_core/Utility_function.py',
    'niah_core/def_run_pre_processing.py',
    'niah_core/def_run_pre_processing_model.py',
    'niah_core/def_run_homogenization.py',
)


class AbaqusApiPathBoundaryTests(unittest.TestCase):
    def test_python2_unicode_is_encoded_only_at_api_boundary(self):
        class FakeUnicode(str):
            pass

        original_unicode = getattr(job_resume, 'unicode', None)
        had_unicode = hasattr(job_resume, 'unicode')
        job_resume.unicode = FakeUnicode
        try:
            with mock.patch.object(job_resume.sys, 'version_info', (2, 7)):
                with mock.patch.object(
                        job_resume.sys,
                        'getfilesystemencoding',
                        return_value='utf-8'):
                    converted = job_resume.abaqus_api_path(
                        FakeUnicode('C:/\u6a21\u578b/BASE.inp')
                    )
        finally:
            if had_unicode:
                job_resume.unicode = original_unicode
            else:
                del job_resume.unicode

        self.assertIsInstance(converted, bytes)
        self.assertEqual(
            converted.decode('utf-8'),
            'C:/\u6a21\u578b/BASE.inp'
        )

    def test_every_input_filename_keyword_uses_path_adapter(self):
        checked_calls = []
        for relative_path in PRODUCTION_PATH_FILES:
            path = os.path.join(ROOT, relative_path)
            with io.open(path, 'r', encoding='utf-8') as stream:
                tree = ast.parse(stream.read(), filename=relative_path)
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                for keyword in node.keywords:
                    if keyword.arg != 'inputFileName':
                        continue
                    checked_calls.append(
                        (relative_path, getattr(node, 'lineno', None))
                    )
                    self.assertIsInstance(
                        keyword.value,
                        ast.Call,
                        msg='%s:%s inputFileName is not adapted' %
                        (relative_path, node.lineno)
                    )
                    self.assertIsInstance(keyword.value.func, ast.Name)
                    self.assertEqual(
                        keyword.value.func.id,
                        'abaqus_api_path',
                        msg='%s:%s bypasses abaqus_api_path' %
                        (relative_path, node.lineno)
                    )

        self.assertEqual(len(checked_calls), 5)

    def test_open_odb_paths_use_same_adapter(self):
        path = os.path.join(ROOT, 'niah_core', 'Utility_function.py')
        with io.open(path, 'r', encoding='utf-8') as stream:
            source = stream.read()
        self.assertEqual(
            source.count('session.openOdb(name=abaqus_api_path(path))'),
            2
        )


if __name__ == '__main__':
    unittest.main()
