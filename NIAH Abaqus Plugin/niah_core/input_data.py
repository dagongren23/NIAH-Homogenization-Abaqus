# -*- coding: UTF-8 -*-
"""
Refactored input layer for NIAH.
This module replaces the original hard-coded getinput() editing workflow.
"""

from __future__ import print_function

import os
import time
from datetime import datetime


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def get_default_params():
    return dict(
        partname='honeycomb_plane_shell_homo',
        stru_type='shell',              # '3D' or 'shell'
        element_type='S4R',              # 'C3D8R' or 'S4R' or 'C3D4R'...
        model_name='Model-1',
        numCpus=8,
        meshsens=1.0e-3,
        material_name='Material-1',
        material_E=1.0,
        material_v=0.3,
        shell_thickness=0.01,
        beam_radius=0.01,
        beam_n1=(0.1, 0.0, -1.0),
        periodicity_ch=3,
        rigid_node=(2.0, 0.0, 0.5),              # must be correct
        rigid_node_tol=0.05,              # must be correct
        data_dir='',
        pre_mode='Part',
        run_pre=True,
        run_solver=True,
        resume_solver=True,
    )


def validate_and_build_runtime_input(params=None, plugin_anchor_dir=None):
    """
    Return a normalized runtime dictionary.

    Parameters
    ----------
    params : dict
        User-defined parameters from GUI or script.
    plugin_anchor_dir : str or None
        Absolute path to the plugin root directory. This should normally be
        the folder containing niah_plugin.py, niah_kernel.py, and niah_core/.

    Notes
    -----
    This deliberately avoids using os.getcwd() as the primary workbench anchor,
    because Abaqus may change cwd between GUI launch, job submission, and post-processing.
    """
    cfg = get_default_params()
    if params:
        cfg.update(params)

    _validate_scalar_inputs(cfg)
    _validate_vector_inputs(cfg)

    plugin_root = _resolve_plugin_root(plugin_anchor_dir)
    # 1. Default Abaqus working directory (for os.chdir and odb/dat files)
    workbench_path = os.path.join(plugin_root, 'niah_workbench')

    # 2. Determine whether the user has entered a custom data path
    # If the user enters a valid path, use the user's path; otherwise, fall back to using workbench_path
    if cfg.get('data_dir') and cfg['data_dir'].strip():
        raw_path = cfg['data_dir'].strip()
        # Clean up the original Python string literals (such as r'...' or r"...") brought from GUI copying
        if raw_path.startswith("r'") and raw_path.endswith("'"):
            raw_path = raw_path[2:-1]
        elif raw_path.startswith('r"') and raw_path.endswith('"'):
            raw_path = raw_path[2:-1]
        elif (raw_path.startswith("'") and raw_path.endswith("'")) or (
                raw_path.startswith('"') and raw_path.endswith('"')):
            raw_path = raw_path[1:-1]

        data_out_path = os.path.abspath(raw_path)
    else:
        data_out_path = workbench_path

    # 3. Generate txt and log files into a separate data_out_path
    txt_path = os.path.join(data_out_path, 'NIAH_CH_txt')
    log_path = os.path.join(
        data_out_path,
        'NIAH_ABAQUS_Progress_{}_{}.txt'.format(
            cfg['partname'], datetime.now().strftime('%Y%m%d')
        )
    )  # .log or None

    _ensure_dir(workbench_path)
    _ensure_dir(data_out_path)  # Ensure that the custom output directory exists
    _ensure_dir(txt_path)

    suffix = _make_suffix(cfg['stru_type'], cfg['periodicity_ch'])
    prefix, condition_ch, dof_per_node = _make_prefix_and_dof(cfg['stru_type'], cfg['element_type'])

    instance_name = cfg['partname'] + '-1'
    outtxt = os.path.join(txt_path, 'CH-{}-{}.txt'.format(cfg['partname'], suffix))
    t0 = time.time()

    runtime = dict(cfg)
    runtime.update(dict(
        plugin_root=plugin_root,
        workbench_path=workbench_path,
        data_out_path=data_out_path,  # The downstream can be used to store special output files or inp
        txt_path=txt_path,
        outtxt=outtxt,
        LOG_PATH=log_path,
        _t0=t0,
        instance_name=instance_name,
        suffix=suffix,
        prefix=prefix,
        condition_ch=condition_ch,
        dof_per_node=dof_per_node,
        model_name_NOPBC=prefix + 'Model-NOPBC',
        model_name_PBC=prefix + 'Model-PBC',
        model_name_NOPBCX=prefix + 'Model-NOPBCX',
        model_name_NOPBCY=prefix + 'Model-NOPBCY',
        BASE_JOB_REALNOPBCX=prefix + 'BASE_NOPBCX',
        BASE_JOB_REALNOPBCY=prefix + 'BASE_NOPBCY',
        BASE_JOB_NOPBC=prefix + 'BASE_NOPBC',
        BASE_JOB_PBC=prefix + 'BASE_PBC',
    ))
    return runtime


def build_legacy_getinput_tuple(params=None, plugin_anchor_dir=None):
    """
    Compatibility layer for legacy downstream scripts that still expect the old
    positional tuple returned by getinput().
    """
    rt = validate_and_build_runtime_input(params=params, plugin_anchor_dir=plugin_anchor_dir)
    return (
        rt['partname'], rt['stru_type'], rt['element_type'], rt['condition_ch'], rt['dof_per_node'],
        rt['model_name'], rt['instance_name'], rt['numCpus'], rt['meshsens'],
        rt['material_name'], rt['material_E'], rt['material_v'], rt['shell_thickness'],
        rt['beam_radius'], rt['beam_n1'], rt['periodicity_ch'], rt['rigid_node'],
        rt['rigid_node_tol'], rt['workbench_path'], rt['outtxt'], rt['suffix'],
        rt['LOG_PATH'], rt['_t0'], rt['model_name_NOPBC'], rt['model_name_PBC'],
        rt['model_name_NOPBCX'], rt['model_name_NOPBCY'], rt['BASE_JOB_REALNOPBCX'],
        rt['BASE_JOB_REALNOPBCY'], rt['BASE_JOB_NOPBC'], rt['BASE_JOB_PBC']
    )


def getinput(plugin_anchor_dir=None, overrides=None):
    """
    Drop-in replacement for your original getinput().
    It preserves the old return type while internally using the new robust path logic.
    """
    return build_legacy_getinput_tuple(params=overrides, plugin_anchor_dir=plugin_anchor_dir)


# -----------------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------------

def _resolve_plugin_root(plugin_anchor_dir=None):
    if plugin_anchor_dir:
        root = os.path.abspath(plugin_anchor_dir)
    else:
        # niah_core/input_data.py -> niah_core -> plugin_root
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    if not os.path.isdir(root):
        raise RuntimeError('Plugin root does not exist: {}'.format(root))
    return root


def _ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def _validate_scalar_inputs(cfg):
    if cfg['stru_type'] not in ('3D', 'shell'):
        raise ValueError("stru_type must be '3D' or 'shell'.")

    # Validate the selected preprocessing mode.
    if cfg['pre_mode'] not in ('Part', 'Model'):
        raise ValueError("pre_mode must be 'Part' or 'Model'.")

    if not isinstance(cfg['numCpus'], (int, long)) if _is_py2() else not isinstance(cfg['numCpus'], int):
        raise ValueError('numCpus must be an integer.')
    if cfg['numCpus'] < 1:
        raise ValueError('numCpus must be >= 1.')

    if cfg['meshsens'] <= 0.0:
        raise ValueError('meshsens must be > 0.')
    if cfg['rigid_node_tol'] < 0.0:
        raise ValueError('rigid_node_tol must be >= 0.')

    if cfg['stru_type'] == 'shell' and cfg['periodicity_ch'] not in (1, 2, 3):
        raise ValueError('For shell, periodicity_ch must be 1, 2, or 3.')



def _validate_vector_inputs(cfg):
    if len(cfg['beam_n1']) != 3:
        raise ValueError('beam_n1 must have length 3.')
    if len(cfg['rigid_node']) != 3:
        raise ValueError('rigid_node must have length 3.')



def _make_suffix(stru_type, periodicity_ch):
    if stru_type == '3D':
        return '3D'
    if periodicity_ch == 3:
        return 'shell-nz'
    if periodicity_ch == 2:
        return 'shell-ny'
    if periodicity_ch == 1:
        return 'shell-nx'
    raise RuntimeError('periodicity_ch has not been assigned correctly.')



def _make_prefix_and_dof(stru_type, element_type):
    beam_shell_types = ('B31', 'S3', 'S3R', 'S4', 'S4R')
    if stru_type == '3D':
        if element_type in beam_shell_types:
            return '3D_6dof_', 2, 6
        return '3D_3dof_', 1, 3
    else:
        if element_type in beam_shell_types:
            return 'shell_6dof_', 4, 6
        return 'shell_3dof_', 3, 3



def _is_py2():
    import sys
    return sys.version_info[0] == 2
