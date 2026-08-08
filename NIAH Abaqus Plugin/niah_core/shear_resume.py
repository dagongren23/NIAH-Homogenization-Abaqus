# -*- coding: UTF-8 -*-
"""Pure helpers for fast and dependency-safe shear-chain resume.

This module deliberately avoids Abaqus imports so its cache and clean-base
rules can be verified with ordinary Python tests.
"""
from __future__ import print_function

import hashlib
import json
import os
import shutil
import time

from job_resume import (
    CHECKPOINT_DIRNAME,
    sha256_file,
    sta_file_completed_successfully,
)


SHEAR_CHAIN_SCHEMA = 1
SHEAR_CLEAN_BASE_SCHEMA = 1
SHEAR_CHAIN_ALGORITHM = 'plate-shear-chain-20260730-v1'
SHEAR_STAGE_NAMES = ('SH1', 'SH2', 'SH3')


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


def _ensure_checkpoint_dir(work_place):
    checkpoint_dir = _checkpoint_dir(work_place)
    if not os.path.isdir(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    return checkpoint_dir


def _as_bytes(value):
    if isinstance(value, bytes):
        return value
    return str(value).encode('utf-8')


def _update_framed(digest, label, payload):
    label_bytes = _as_bytes(label)
    payload_bytes = _as_bytes(payload)
    digest.update(_as_bytes(len(label_bytes)))
    digest.update(b':')
    digest.update(label_bytes)
    digest.update(b':')
    digest.update(_as_bytes(len(payload_bytes)))
    digest.update(b':')
    digest.update(payload_bytes)
    digest.update(b'\n')


def _array_bytes(array_value):
    if array_value is None:
        return b'<NONE>'
    if hasattr(array_value, 'tobytes'):
        return array_value.tobytes(order='C')
    if hasattr(array_value, 'tostring'):
        return array_value.tostring(order='C')
    return _as_bytes(repr(list(array_value)))


def build_shear_dependency_fingerprint(metadata, file_paths, arrays):
    """Hash every dependency needed to reuse an SH1/SH2/SH3 result chain."""
    digest = hashlib.sha256()
    _update_framed(digest, 'algorithm', SHEAR_CHAIN_ALGORITHM)

    for key in sorted(metadata.keys()):
        _update_framed(digest, 'meta:' + str(key), repr(metadata[key]))

    for key in sorted(file_paths.keys()):
        path = file_paths[key]
        if not os.path.isfile(path):
            raise IOError("Shear dependency file does not exist: %s" % path)
        _update_framed(digest, 'file:' + str(key), sha256_file(path))

    for key in sorted(arrays.keys()):
        value = arrays[key]
        shape = getattr(value, 'shape', None)
        dtype = getattr(value, 'dtype', None)
        _update_framed(digest, 'array-shape:' + str(key), repr(shape))
        _update_framed(digest, 'array-dtype:' + str(key), repr(dtype))
        _update_framed(digest, 'array-data:' + str(key), _array_bytes(value))

    return digest.hexdigest()


def preserve_clean_shear_base(work_place, logical_base_name, source_path):
    """Preserve one immutable clean shear base and append an activation record."""
    if not os.path.isfile(source_path):
        raise IOError("Cannot preserve missing shear base: %s" % source_path)

    source_hash = sha256_file(source_path)
    source_stem = os.path.splitext(os.path.basename(source_path))[0]
    clean_filename = '%s_SHEAR_CLEAN_%s.inp' % (
        source_stem,
        source_hash[:16],
    )
    clean_path = os.path.join(work_place, clean_filename)
    if not os.path.exists(clean_path):
        shutil.copyfile(source_path, clean_path)
    elif sha256_file(clean_path) != source_hash:
        raise RuntimeError(
            "Existing clean shear base has unexpected content: %s" %
            clean_path
        )

    checkpoint_dir = _ensure_checkpoint_dir(work_place)
    created_at = time.time()
    stamp = int(created_at * 1000000)
    record = {
        'schema': SHEAR_CLEAN_BASE_SCHEMA,
        'kind': 'shear_clean_base',
        'logical_base_name': logical_base_name,
        'clean_file': clean_filename,
        'sha256': source_hash,
        'created_at_epoch': created_at,
    }
    prefix = 'shear_clean_base__%s__%020d' % (
        _safe_name(logical_base_name),
        stamp,
    )
    record_path = os.path.join(checkpoint_dir, prefix + '.json')
    attempt = 0
    while os.path.exists(record_path):
        attempt += 1
        record_path = os.path.join(
            checkpoint_dir,
            '%s__%03d.json' % (prefix, attempt)
        )
    with open(record_path, 'w') as stream:
        json.dump(record, stream, sort_keys=True, indent=2)
        stream.write('\n')
    return clean_path


def find_active_clean_shear_base(work_place, logical_base_name):
    """Return the newest valid clean base activated by preprocessing."""
    checkpoint_dir = _checkpoint_dir(work_place)
    if not os.path.isdir(checkpoint_dir):
        return None
    prefix = 'shear_clean_base__%s__' % _safe_name(logical_base_name)
    candidates = []
    for filename in os.listdir(checkpoint_dir):
        if not filename.startswith(prefix) or not filename.endswith('.json'):
            continue
        path = os.path.join(checkpoint_dir, filename)
        try:
            with open(path, 'r') as stream:
                record = json.load(stream)
        except (IOError, OSError, ValueError):
            continue
        if record.get('schema') != SHEAR_CLEAN_BASE_SCHEMA:
            continue
        if record.get('kind') != 'shear_clean_base':
            continue
        if record.get('logical_base_name') != logical_base_name:
            continue
        candidates.append((record.get('created_at_epoch', 0.0), record))

    candidates.sort(key=lambda item: item[0], reverse=True)
    for _created_at, record in candidates:
        clean_path = os.path.join(work_place, record.get('clean_file', ''))
        if not os.path.isfile(clean_path):
            continue
        try:
            if sha256_file(clean_path) == record.get('sha256'):
                return clean_path
        except (IOError, OSError):
            continue
    return None


def _stage_record(work_place, odb_path):
    odb_name = os.path.basename(odb_path)
    actual_job_name = os.path.splitext(odb_name)[0]
    return {
        'actual_job_name': actual_job_name,
        'odb_file': odb_name,
        'sta_file': actual_job_name + '.sta',
        'lock_file': actual_job_name + '.lck',
    }


def record_shear_chain_checkpoint(work_place, case_prefix,
                                  dependency_fingerprint, odb_paths):
    """Append an immutable checkpoint for a completed three-stage shear chain."""
    if len(odb_paths) != len(SHEAR_STAGE_NAMES):
        raise ValueError("A shear chain checkpoint requires three ODB paths.")
    checkpoint_dir = _ensure_checkpoint_dir(work_place)
    created_at = time.time()
    record = {
        'schema': SHEAR_CHAIN_SCHEMA,
        'kind': 'shear_chain',
        'case_prefix': case_prefix,
        'dependency_sha256': dependency_fingerprint,
        'algorithm': SHEAR_CHAIN_ALGORITHM,
        'created_at_epoch': created_at,
        'stages': {},
    }
    for stage_name, odb_path in zip(SHEAR_STAGE_NAMES, odb_paths):
        record['stages'][stage_name] = _stage_record(work_place, odb_path)

    filename_prefix = 'shear_chain__%s__%s__%020d' % (
        _safe_name(case_prefix),
        dependency_fingerprint,
        int(created_at * 1000000),
    )
    checkpoint_path = os.path.join(
        checkpoint_dir,
        filename_prefix + '.json'
    )
    attempt = 0
    while os.path.exists(checkpoint_path):
        attempt += 1
        checkpoint_path = os.path.join(
            checkpoint_dir,
            '%s__%03d.json' % (filename_prefix, attempt)
        )
    with open(checkpoint_path, 'w') as stream:
        json.dump(record, stream, sort_keys=True, indent=2)
        stream.write('\n')
    return checkpoint_path


def _resolve_stage(work_place, stage_record):
    result = dict(stage_record)
    odb_path = os.path.join(work_place, stage_record.get('odb_file', ''))
    sta_path = os.path.join(work_place, stage_record.get('sta_file', ''))
    lock_path = os.path.join(work_place, stage_record.get('lock_file', ''))
    result['odb_path'] = odb_path
    result['sta_path'] = sta_path
    result['lock_path'] = lock_path
    result['artifact_valid'] = bool(
        os.path.isfile(odb_path) and
        sta_file_completed_successfully(sta_path) and
        not (lock_path and os.path.exists(lock_path))
    )
    return result


def find_shear_chain_checkpoint(work_place, case_prefix,
                                dependency_fingerprint):
    """Return newest matching chain record with per-stage artifact status."""
    checkpoint_dir = _checkpoint_dir(work_place)
    if not os.path.isdir(checkpoint_dir):
        return None
    prefix = 'shear_chain__%s__%s__' % (
        _safe_name(case_prefix),
        dependency_fingerprint,
    )
    candidates = []
    for filename in os.listdir(checkpoint_dir):
        if not filename.startswith(prefix) or not filename.endswith('.json'):
            continue
        path = os.path.join(checkpoint_dir, filename)
        try:
            with open(path, 'r') as stream:
                record = json.load(stream)
        except (IOError, OSError, ValueError):
            continue
        if record.get('schema') != SHEAR_CHAIN_SCHEMA:
            continue
        if record.get('kind') != 'shear_chain':
            continue
        if record.get('case_prefix') != case_prefix:
            continue
        if record.get('dependency_sha256') != dependency_fingerprint:
            continue
        candidates.append((record.get('created_at_epoch', 0.0), path, record))

    candidates.sort(key=lambda item: item[0], reverse=True)
    for _created_at, path, record in candidates:
        stages = record.get('stages', {})
        if any(stage not in stages for stage in SHEAR_STAGE_NAMES):
            continue
        result = dict(record)
        result['checkpoint_path'] = path
        result['stages'] = {}
        for stage_name in SHEAR_STAGE_NAMES:
            result['stages'][stage_name] = _resolve_stage(
                work_place,
                stages[stage_name]
            )
        return result
    return None
