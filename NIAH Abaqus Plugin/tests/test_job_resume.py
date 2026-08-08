import sys
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "niah_core"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))


class JobResumeHelperTests(unittest.TestCase):
    class _FakeStep:
        def __init__(self, frame_count=1):
            self.frames = [object()] * frame_count

    class _FakeOdb:
        def __init__(self, path, frame_count=1, has_step=True):
            self.path = path
            self.name = path
            self.steps = {}
            if has_step:
                self.steps["Step-1"] = JobResumeHelperTests._FakeStep(
                    frame_count
                )
            self.close_count = 0

        def close(self):
            self.close_count += 1

    def test_user_supplied_successful_sta_is_recognized(self):
        from job_resume import sta_text_completed_successfully

        sta_text = (
            " Abaqus/Standard 2020 DATE 29-7\u6708-2026 TIME 12:52:01\n"
            " SUMMARY OF JOB INFORMATION:\n"
            " THE ANALYSIS HAS COMPLETED SUCCESSFULLY\n"
        )
        self.assertTrue(sta_text_completed_successfully(sta_text))
        self.assertTrue(sta_text_completed_successfully(sta_text.encode("utf-8")))

    def test_incomplete_or_failed_sta_is_rejected(self):
        from job_resume import sta_text_completed_successfully

        self.assertFalse(sta_text_completed_successfully(""))
        self.assertFalse(
            sta_text_completed_successfully(
                "THE ANALYSIS HAS NOT COMPLETED SUCCESSFULLY"
            )
        )

    def test_input_fingerprint_changes_with_dependency_content(self):
        from job_resume import sha256_bytes

        original = "*Cload\nINSTANCE.1, 1, 1.0\n"
        changed = "*Cload\nINSTANCE.1, 1, 1.1\n"
        self.assertNotEqual(sha256_bytes(original), sha256_bytes(changed))
        self.assertEqual(sha256_bytes(original), sha256_bytes(original))

    def test_cache_requires_matching_hash_sta_odb_and_no_lock(self):
        from job_resume import (
            CHECKPOINT_SCHEMA,
            cache_candidate_is_valid,
        )

        record = {
            "schema": CHECKPOINT_SCHEMA,
            "input_sha256": "expected",
        }
        success = "THE ANALYSIS HAS COMPLETED SUCCESSFULLY"
        self.assertTrue(
            cache_candidate_is_valid(
                record, "expected", success, odb_exists=True, lock_exists=False
            )
        )
        self.assertFalse(
            cache_candidate_is_valid(
                record, "changed", success, odb_exists=True, lock_exists=False
            )
        )
        self.assertFalse(
            cache_candidate_is_valid(
                record, "expected", success, odb_exists=False, lock_exists=False
            )
        )
        self.assertFalse(
            cache_candidate_is_valid(
                record, "expected", success, odb_exists=True, lock_exists=True
            )
        )
        self.assertFalse(
            cache_candidate_is_valid(
                record, "expected", "ABORTED", odb_exists=True, lock_exists=False
            )
        )

    def test_retry_names_do_not_overwrite_existing_artifacts(self):
        from job_resume import (
            choose_available_input_path,
            choose_available_job_name,
        )

        existing = {
            str(Path("work") / "CASE.odb"),
            str(Path("work") / "CASE_retry001.sta"),
            str(Path("work") / "CASE.inp"),
            str(Path("work") / "CASE_input001.inp"),
        }

        def fake_exists(path):
            return str(Path(path)) in existing

        with mock.patch("job_resume.os.path.exists", side_effect=fake_exists):
            self.assertEqual(
                choose_available_job_name("work", "CASE"),
                "CASE_retry002",
            )
            self.assertEqual(
                choose_available_input_path("work", "CASE"),
                str(Path("work") / "CASE_input002.inp"),
            )

    def test_open_session_odb_is_reused_without_open_or_close(self):
        from job_resume import validate_completed_odb

        odb = self._FakeOdb(r"C:\Work\CASE.odb")

        def unexpected_open(_path):
            self.fail("open_odb must not be called for an existing session ODB")

        valid, error, warning = validate_completed_odb(
            {r"c:/work/case.odb": odb},
            r"C:\WORK\CASE.ODB",
            unexpected_open,
        )
        self.assertTrue(valid)
        self.assertEqual(error, "")
        self.assertEqual(warning, "")
        self.assertEqual(odb.close_count, 0)

    def test_odb_object_path_can_match_when_repository_key_differs(self):
        from job_resume import validate_completed_odb

        odb = self._FakeOdb(r"C:\Work\CASE.odb")
        valid, error, warning = validate_completed_odb(
            {"display-name": odb},
            r"c:/work/case.odb",
            lambda _path: self.fail("existing ODB should be reused"),
        )
        self.assertEqual((valid, error, warning), (True, "", ""))
        self.assertEqual(odb.close_count, 0)

    def test_newly_opened_odb_is_validated_and_closed_once(self):
        from job_resume import validate_completed_odb

        odb = self._FakeOdb(r"C:\Work\CASE.odb")
        opened_paths = []

        def open_odb(path):
            opened_paths.append(path)
            return odb

        valid, error, warning = validate_completed_odb(
            {},
            r"C:\Work\CASE.odb",
            open_odb,
        )
        self.assertEqual((valid, error, warning), (True, "", ""))
        self.assertEqual(opened_paths, [r"C:\Work\CASE.odb"])
        self.assertEqual(odb.close_count, 1)

    def test_missing_step_and_empty_frames_report_specific_reason(self):
        from job_resume import validate_completed_odb

        missing_step = self._FakeOdb(
            r"C:\Work\MISSING.odb",
            has_step=False,
        )
        valid, error, warning = validate_completed_odb(
            {"missing": missing_step},
            r"C:\Work\MISSING.odb",
            lambda _path: missing_step,
        )
        self.assertFalse(valid)
        self.assertIn("no Step-1", error)
        self.assertEqual(warning, "")

        empty_step = self._FakeOdb(
            r"C:\Work\EMPTY.odb",
            frame_count=0,
        )
        valid, error, warning = validate_completed_odb(
            {"empty": empty_step},
            r"C:\Work\EMPTY.odb",
            lambda _path: empty_step,
        )
        self.assertFalse(valid)
        self.assertIn("no frames", error)
        self.assertEqual(warning, "")

    def test_open_error_preserves_exception_type_and_message(self):
        from job_resume import validate_completed_odb

        class OdbError(Exception):
            pass

        def failing_open(_path):
            raise OdbError("database is already registered")

        valid, error, warning = validate_completed_odb(
            {},
            r"C:\Work\CASE.odb",
            failing_open,
        )
        self.assertFalse(valid)
        self.assertIn("OdbError", error)
        self.assertIn("already registered", error)
        self.assertEqual(warning, "")

    def test_python3_abaqus_path_remains_text(self):
        from job_resume import abaqus_api_path

        path = r"C:\Work\CASE.odb"
        self.assertIs(abaqus_api_path(path), path)


class ResumeSolverPropagationTests(unittest.TestCase):
    @staticmethod
    def _read(relative_path):
        return (ROOT / relative_path).read_text(encoding="utf-8")

    def test_gui_and_runtime_expose_resume_switch(self):
        input_source = self._read("niah_core/input_data.py")
        form_source = self._read("niahForm.py")
        dialog_source = self._read("niahDB.py")
        kernel_source = self._read("niah_kernel.py")
        cleanup_source = self._read("niah_core/def_run_correct.py")

        self.assertIn("resume_solver=True", input_source)
        self.assertIn("'resume_solver'", form_source)
        self.assertIn("keyword='resume_solver'", dialog_source)
        self.assertIn("resume_solver=bool(resume_solver)", kernel_source)
        self.assertIn("_run_correct(runtime)", kernel_source)
        self.assertIn("if runtime_input.get('resume_solver')", cleanup_source)
        self.assertIn("if extension not in ('.inp', '.sta')", cleanup_source)

    def test_actual_x2_odb_path_is_forwarded_to_dependent_stage(self):
        utility_source = self._read("niah_core/Utility_function.py")
        solver_source = self._read("niah_core/def_run_homogenization.py")

        self.assertIn(
            "return modif1, modiu2, modif2, odb_path2",
            utility_source,
        )
        self.assertIn("x2_odb_path=mem_x2_odb_path", solver_source)
        self.assertIn("x2_odb_path=ben_x2_odb_path", solver_source)

    def test_runner_uses_hash_checkpoint_and_non_overwriting_retry(self):
        utility_source = self._read("niah_core/Utility_function.py")

        self.assertIn("find_reusable_job(", utility_source)
        self.assertIn("choose_available_job_name(", utility_source)
        self.assertIn("record_success_checkpoint(", utility_source)
        self.assertNotIn("os.remove(", utility_source)


if __name__ == "__main__":
    unittest.main()
