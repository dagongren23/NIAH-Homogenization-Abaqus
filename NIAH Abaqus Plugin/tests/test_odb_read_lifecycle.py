# -*- coding: UTF-8 -*-
"""Regression tests for cached ODB acquisition and production read paths."""
from __future__ import print_function

import ast
import io
import os
import sys
import unittest


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORE = os.path.join(ROOT, 'niah_core')
if CORE not in sys.path:
    sys.path.insert(0, CORE)

from job_resume import acquire_odb_for_read


class _FakeOdb(object):
    def __init__(self, path):
        self.path = path


class OdbAcquisitionTests(unittest.TestCase):
    def test_existing_session_odb_is_reused_without_opening(self):
        odb_path = r"C:\work\case.odb"
        existing = _FakeOdb(r"c:/WORK/CASE.odb")
        open_calls = []

        def opener(path):
            open_calls.append(path)
            return _FakeOdb(path)

        odb_obj, opened_here = acquire_odb_for_read(
            {r"c:/work/case.odb": existing},
            odb_path,
            opener
        )

        self.assertIs(odb_obj, existing)
        self.assertFalse(opened_here)
        self.assertEqual(open_calls, [])

    def test_missing_session_odb_is_opened_with_original_path(self):
        odb_path = u"C:\\work\\case.odb"
        opened = _FakeOdb(odb_path)
        open_calls = []

        def opener(path):
            open_calls.append(path)
            return opened

        odb_obj, opened_here = acquire_odb_for_read(
            {},
            odb_path,
            opener
        )

        self.assertIs(odb_obj, opened)
        self.assertTrue(opened_here)
        self.assertEqual(open_calls, [odb_path])


class ProductionOdbReadSourceTests(unittest.TestCase):
    TARGET_FUNCTIONS = (
        '_extract_reaction_vector_from_odb',
        '_extract_displacement_vector_from_odb',
        '_extract_reaction_vector6dof_from_odb2stat4',
        '_extract_displacement_vector6dof_from_odb2stat2',
    )

    @classmethod
    def setUpClass(cls):
        cls.utility_path = os.path.join(CORE, 'Utility_function.py')
        with io.open(cls.utility_path, 'r', encoding='utf-8') as stream:
            cls.source = stream.read()
        cls.tree = ast.parse(cls.source, filename=cls.utility_path)
        cls.functions = {
            node.name: node
            for node in cls.tree.body
            if isinstance(node, ast.FunctionDef)
        }

    def _called_names(self, function_node):
        names = []
        for node in ast.walk(function_node):
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Name):
                names.append(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                names.append(node.func.attr)
        return names

    def test_all_extractors_use_shared_acquire_and_owned_close(self):
        for function_name in self.TARGET_FUNCTIONS:
            function_node = self.functions[function_name]
            called_names = self._called_names(function_node)
            self.assertIn(
                '_acquire_odb_for_read',
                called_names,
                msg=function_name
            )
            self.assertIn(
                '_close_odb_if_owned',
                called_names,
                msg=function_name
            )
            self.assertNotIn('openOdb', called_names, msg=function_name)

    def test_no_raw_checkpoint_path_indexes_session_repository(self):
        self.assertNotIn('session.odbs[odb_path]', self.source)
        self.assertNotIn('session.openOdb(name=odb_path)', self.source)

    def test_every_direct_openodb_call_converts_legacy_path_type(self):
        direct_calls = []
        for node in ast.walk(self.tree):
            if (isinstance(node, ast.Call) and
                    isinstance(node.func, ast.Attribute) and
                    node.func.attr == 'openOdb'):
                direct_calls.append(node)

        self.assertGreater(len(direct_calls), 0)
        for call in direct_calls:
            name_keywords = [
                keyword.value
                for keyword in call.keywords
                if keyword.arg == 'name'
            ]
            self.assertEqual(len(name_keywords), 1)
            value = name_keywords[0]
            self.assertIsInstance(value, ast.Call)
            self.assertIsInstance(value.func, ast.Name)
            self.assertEqual(value.func.id, 'abaqus_api_path')


if __name__ == '__main__':
    unittest.main()
