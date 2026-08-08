# -*- coding: UTF-8 -*-
"""
Kernel entry for NIAH plug-in.
All heavy Abaqus logic should stay in niah_core or submodules beneath the same plugin root.
"""

from __future__ import print_function

import os
import sys
import traceback

# Ensure the plugin root and its niah_core package are importable even if Abaqus changes cwd.
_THIS_FILE = os.path.abspath(__file__)
_PLUGIN_DIR = os.path.dirname(_THIS_FILE)
_CORE_DIR = os.path.join(_PLUGIN_DIR, "niah_core")
for _p in [_PLUGIN_DIR, _CORE_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from niah_core.input_data import get_default_params, validate_and_build_runtime_input


def _log(msg):
    print('[NIAH]', msg)


def run_niah_plugin(partname='honeycomb_plane_shell_homo',
                    stru_type='shell',
                    element_type='S4R',
                    model_name='Model-1',
                    numCpus=8,
                    meshsens=1.0e-3,
                    material_E=1.0,
                    material_v=0.3,
                    shell_thickness=0.01,
                    beam_radius=0.01,
                    beam_n1x=0.1,
                    beam_n1y=0.0,
                    beam_n1z=-1.0,
                    periodicity_ch=3,
                    rigid_nodex=2.0,
                    rigid_nodey=0.0,
                    rigid_nodez=0.5,
                    rigid_node_tol=0.05,
                    data_dir='',           # <--- data_dir
                    pre_mode='Part',      # <--- pre_mode
                    run_pre=True,
                    run_solver=True,
                    resume_solver=True,
                    run_vis=None):
    """
    Function called directly by the RSG dialog.

    ``run_vis`` is retained only so older scripts using this keyword continue
    to run. Visualization is provided separately by the Python 3 utility.
    """
    params = get_default_params()
    params.update(dict(
        partname=str(partname),
        stru_type=str(stru_type),
        element_type=str(element_type),
        model_name=str(model_name),
        numCpus=int(numCpus),
        meshsens=float(meshsens),
        material_E=float(material_E),
        material_v=float(material_v),
        shell_thickness=float(shell_thickness),
        beam_radius=float(beam_radius),
        beam_n1=(float(beam_n1x), float(beam_n1y), float(beam_n1z)),
        periodicity_ch=int(periodicity_ch),
        rigid_node=(float(rigid_nodex), float(rigid_nodey), float(rigid_nodez)),
        data_dir=str(data_dir),
        pre_mode=str(pre_mode).strip().capitalize(),
        rigid_node_tol=float(rigid_node_tol),
        run_pre=bool(run_pre),
        run_solver=bool(run_solver),
        resume_solver=bool(resume_solver),
    ))
    return run_pipeline(params)


def run_pipeline(params):
    """
    Safe wrapper for the whole workflow.
    Replace the placeholder calls with your real NIAH implementation.
    """
    runtime = validate_and_build_runtime_input(params, plugin_anchor_dir=_PLUGIN_DIR)
    _log('Plugin root  : {}'.format(runtime['plugin_root']))
    _log('Workbench    : {}'.format(runtime['workbench_path']))
    _log('TXT path     : {}'.format(runtime['txt_path']))
    _log('Structure    : {} / {}'.format(runtime['stru_type'], runtime['element_type']))
    _log('Pre-mode     : {}'.format(runtime['pre_mode']))
    _log('Resume solver: {}'.format(runtime['resume_solver']))

    try:
        if runtime['run_pre']:
            if runtime['pre_mode'] == 'Model':
                _run_pre_processing_model(runtime)
            else:
                _run_pre_processing(runtime)
            # ---------------------------------------
        if runtime['run_solver']:
            _run_homogenization_solver(runtime)
        _run_correct(runtime)
        _log('Workflow completed successfully.')
        return runtime
    except Exception:
        _log('Workflow failed. See traceback below:')
        traceback.print_exc()
        raise


def _run_pre_processing(runtime):
    _log('Running 1. Pre-processing (Part mode) ...')
    # Example of robust local import under plugin root.
    from niah_core.def_run_pre_processing import run_pre_processing
    return run_pre_processing(runtime)

def _run_pre_processing_model(runtime):
    _log('Running 1. Pre-processing (Model mode) ...')
    # Example of robust local import under plugin root.
    from niah_core.def_run_pre_processing_model import run_pre_processing_model
    return run_pre_processing_model(runtime)


def _run_homogenization_solver(runtime):
    _log('Running 2. Homogenization solver ...')
    from niah_core.def_run_homogenization import run_homogenization
    run_homogenization(runtime)
    return None


def _run_correct(runtime):
    _log('Running 3. recoverable output cleanup ...')
    from niah_core.def_run_correct import run_correct
    return run_correct(1, runtime)
