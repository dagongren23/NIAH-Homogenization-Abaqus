# -*- coding: UTF-8 -*-
"""Pure-Python helpers for NIAH job-level resume checkpoints.

This module intentionally avoids Abaqus imports so fingerprint, STA and retry
logic can be tested in a normal Python interpreter.
"""
from __future__ import print_function

import hashlib
import json
import os
import sys
import time


SUCCESS_MARKER = 'THE ANALYSIS HAS COMPLETED SUCCESSFULLY'
CHECKPOINT_SCHEMA = 1
CHECKPOINT_DIRNAME = '.niah_resume'
JOB_OUTPUT_EXTENSIONS = (
    '.odb', '.lck', '.dat', '.msg', '.sta', '.com', '.prt', '.sim',
    '.log', '.ipm', '.stt', '.res'
)


def sha256_bytes(data):
    """Return a stable SHA-256 hex digest for bytes or text."""
    if not isinstance(data, bytes):
        data = data.encode('utf-8')
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def sta_text_completed_successfully(text):
    """Recognize the Abaqus/Standard successful-completion STA marker."""
    upper_text = text.upper()
    marker = SUCCESS_MARKER
    try:
        return marker in upper_text
    except TypeError:
        return marker.encode('ascii') in upper_text


def sta_file_completed_successfully(sta_path):
    if not os.path.isfile(sta_path):
        return False
    with open(sta_path, 'rb') as stream:
        return sta_text_completed_successfully(stream.read())


def normalize_odb_path(path):
    """Normalize an ODB repository key or file path for comparison."""
    if not path:
        return ''
    return os.path.normcase(
        os.path.abspath(os.path.normpath(path))
    )


def abaqus_api_path(path):
    """Return a path type accepted by legacy Abaqus/Python 2 APIs."""
    if sys.version_info[0] < 3:
        try:
            unicode_type = unicode
        except NameError:
            unicode_type = None
        if unicode_type is not None and isinstance(path, unicode_type):
            encoding = sys.getfilesystemencoding() or 'mbcs'
            return path.encode(encoding)
    return path


def find_open_odb(session_odbs, odb_path):
    """Return an already-open ODB matching odb_path, or None.

    Abaqus repository keys and ODB name/path attributes can use different
    slash direction or path case.  Compare all available representations
    without opening or closing any database.
    """
    target = normalize_odb_path(odb_path)
    for key in list(session_odbs.keys()):
        odb_obj = session_odbs[key]
        candidates = [key]
        for attr_name in ('path', 'name'):
            try:
                attr_value = getattr(odb_obj, attr_name, None)
            except Exception:
                attr_value = None
            if attr_value:
                candidates.append(attr_value)
        for candidate in candidates:
            if normalize_odb_path(candidate) == target:
                return odb_obj
    return None


def acquire_odb_for_read(session_odbs, odb_path, open_odb):
    """Return ``(odb, opened_here)`` without reopening a session ODB.

    ``open_odb`` receives the original path so the caller can perform any
    Abaqus-version-specific path conversion at the API boundary.
    """
    odb_obj = find_open_odb(session_odbs, odb_path)
    if odb_obj is not None:
        return odb_obj, False
    return open_odb(odb_path), True


def validate_completed_odb(session_odbs, odb_path, open_odb):
    """Validate Step-1 and frames while respecting ODB ownership.

    open_odb is called only when the target is not already present in the
    Abaqus session repository.  Only an ODB returned by open_odb is closed.
    Return (is_valid, error_detail, close_warning).
    """
    odb_obj = None
    opened_here = False
    is_valid = False
    error_detail = ''
    close_warning = ''
    try:
        odb_obj, opened_here = acquire_odb_for_read(
            session_odbs,
            odb_path,
            open_odb
        )
        if 'Step-1' not in odb_obj.steps:
            error_detail = "ODB has no Step-1."
        elif len(odb_obj.steps['Step-1'].frames) == 0:
            error_detail = "ODB Step-1 has no frames."
        else:
            is_valid = True
    except Exception as exc:
        error_detail = "%s: %s" % (exc.__class__.__name__, exc)
    finally:
        if opened_here and odb_obj is not None:
            try:
                odb_obj.close()
            except Exception as close_exc:
                close_warning = "%s: %s" % (
                    close_exc.__class__.__name__,
                    close_exc
                )
    return is_valid, error_detail, close_warning


def cache_candidate_is_valid(record, expected_fingerprint, sta_text,
                             odb_exists, lock_exists):
    """Validate non-Abaqus parts of one checkpoint candidate."""
    if not isinstance(record, dict):
        return False
    if record.get('schema') != CHECKPOINT_SCHEMA:
        return False
    if record.get('input_sha256') != expected_fingerprint:
        return False
    if not odb_exists or lock_exists:
        return False
    return sta_text_completed_successfully(sta_text)


def _safe_name(name):
    chars = []
    for char in str(name):
        if char.isalnum() or char in ('-', '_', '.'):
            chars.append(char)
        else:
            chars.append('_')
    return ''.join(chars)


def _checkpoint_dir(work_place):
    return os.path.join(work_place, CHECKPOINT_DIRNAME)


def _record_artifact_path(work_place, record, key):
    value = record.get(key, '')
    if not value:
        return ''
    if os.path.isabs(value):
        return value
    return os.path.join(work_place, value)


def find_reusable_job(work_place, logical_job_name, inp_path):
    """Return a matching successful checkpoint, or None."""
    fingerprint = sha256_file(inp_path)
    checkpoint_dir = _checkpoint_dir(work_place)
    if not os.path.isdir(checkpoint_dir):
        return None

    prefix = '%s__%s__' % (_safe_name(logical_job_name), fingerprint)
    filenames = sorted(os.listdir(checkpoint_dir), reverse=True)
    for filename in filenames:
        if not filename.startswith(prefix) or not filename.endswith('.json'):
            continue
        checkpoint_path = os.path.join(checkpoint_dir, filename)
        try:
            with open(checkpoint_path, 'r') as stream:
                record = json.load(stream)
        except (IOError, OSError, ValueError):
            continue

        odb_path = _record_artifact_path(work_place, record, 'odb_file')
        sta_path = _record_artifact_path(work_place, record, 'sta_file')
        lock_path = _record_artifact_path(work_place, record, 'lock_file')
        try:
            with open(sta_path, 'rb') as stream:
                sta_text = stream.read()
        except (IOError, OSError):
            sta_text = b''

        if cache_candidate_is_valid(
                record,
                fingerprint,
                sta_text,
                os.path.isfile(odb_path),
                bool(lock_path and os.path.exists(lock_path))):
            result = dict(record)
            result['checkpoint_path'] = checkpoint_path
            result['odb_path'] = odb_path
            result['sta_path'] = sta_path
            result['input_sha256'] = fingerprint
            return result
    return None


def record_success_checkpoint(work_place, logical_job_name, actual_job_name,
                              inp_path, odb_path, sta_path):
    """Append one immutable success checkpoint and return its path."""
    fingerprint = sha256_file(inp_path)
    checkpoint_dir = _checkpoint_dir(work_place)
    if not os.path.isdir(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    filename = '%s__%s__%s.json' % (
        _safe_name(logical_job_name),
        fingerprint,
        _safe_name(actual_job_name)
    )
    checkpoint_path = os.path.join(checkpoint_dir, filename)
    if os.path.exists(checkpoint_path):
        return checkpoint_path

    record = {
        'schema': CHECKPOINT_SCHEMA,
        'logical_job_name': logical_job_name,
        'actual_job_name': actual_job_name,
        'input_sha256': fingerprint,
        'odb_file': os.path.basename(odb_path),
        'sta_file': os.path.basename(sta_path),
        'lock_file': actual_job_name + '.lck',
        'created_at_epoch': time.time(),
    }
    with open(checkpoint_path, 'w') as stream:
        json.dump(record, stream, sort_keys=True, indent=2)
        stream.write('\n')
    return checkpoint_path


def _name_has_output_artifacts(work_place, job_name):
    for extension in JOB_OUTPUT_EXTENSIONS:
        if os.path.exists(os.path.join(work_place, job_name + extension)):
            return True
    return False


def choose_available_job_name(work_place, logical_job_name,
                              occupied_job_names=None):
    """Choose a non-overwriting logical or retryNNN Abaqus job name."""
    occupied = set(occupied_job_names or ())
    if (logical_job_name not in occupied and
            not _name_has_output_artifacts(work_place, logical_job_name)):
        return logical_job_name

    for attempt in range(1, 10000):
        candidate = '%s_retry%03d' % (logical_job_name, attempt)
        if candidate in occupied:
            continue
        if not _name_has_output_artifacts(work_place, candidate):
            return candidate
    raise RuntimeError(
        "No available retry job name for %s after 9999 attempts." %
        logical_job_name
    )


def choose_available_input_path(work_place, logical_job_name):
    """Return a new INP path without overwriting an earlier logical input."""
    primary = os.path.join(work_place, logical_job_name + '.inp')
    if not os.path.exists(primary):
        return primary
    for attempt in range(1, 10000):
        candidate = os.path.join(
            work_place,
            '%s_input%03d.inp' % (logical_job_name, attempt)
        )
        if not os.path.exists(candidate):
            return candidate
    raise RuntimeError(
        "No available input filename for %s after 9999 attempts." %
        logical_job_name
    )
