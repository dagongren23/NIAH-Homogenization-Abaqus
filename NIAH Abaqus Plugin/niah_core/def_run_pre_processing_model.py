# -*-coding:UTF-8-*-
"""
NIAH-ABAQUS: 1. Pre-processing
Author: Zhihui Liu
Date:   2026/05/09
Email:  dutme_lzh@163.com
"""
from caeModules import *
from abaqusConstants import *
from abaqus import *
def run_pre_processing_model(runtime_input):
    from driverUtils import executeOnCaeStartup
    from datetime import datetime
    import numpy as np
    import os
    executeOnCaeStartup()
    Mdb()

    # Input
    # (partname, stru_type, element_type, condition_ch, dof_per_node, model_name, instance_name, numCpus, meshsens,
    #  material_name, material_E, material_v, shell_thickness, beam_radius, beam_n1, periodicity_ch,
    #  rigid_node, rigid_node_tol, work_place, outtxt, suffix, LOG_PATH, _t0,
    #  model_name_NOPBC, model_name_PBC, model_name_NOPBCX, model_name_NOPBCY,
    #  BASE_JOB_REALNOPBCX, BASE_JOB_REALNOPBCY, BASE_JOB_NOPBC, BASE_JOB_PBC) = runtime_input
    partname = runtime_input['partname']
    stru_type = runtime_input['stru_type']
    element_type = runtime_input['element_type']
    condition_ch = runtime_input['condition_ch']
    dof_per_node = runtime_input['dof_per_node']
    model_name = runtime_input['model_name']
    instance_name = runtime_input['instance_name']
    numCpus = runtime_input['numCpus']
    meshsens = runtime_input['meshsens']
    periodicity_ch = runtime_input['periodicity_ch']
    rigid_node = runtime_input['rigid_node']
    rigid_node_tol = runtime_input['rigid_node_tol']
    work_place = runtime_input['workbench_path']
    data_out_path = runtime_input['data_out_path']
    LOG_PATH = runtime_input['LOG_PATH']
    _t0 = runtime_input['_t0']
    model_name_NOPBC = runtime_input['model_name_NOPBC']
    model_name_PBC = runtime_input['model_name_PBC']
    model_name_NOPBCX = runtime_input['model_name_NOPBCX']
    model_name_NOPBCY = runtime_input['model_name_NOPBCY']
    BASE_JOB_REALNOPBCX = runtime_input['BASE_JOB_REALNOPBCX']
    BASE_JOB_REALNOPBCY = runtime_input['BASE_JOB_REALNOPBCY']
    BASE_JOB_NOPBC = runtime_input['BASE_JOB_NOPBC']
    BASE_JOB_PBC = runtime_input['BASE_JOB_PBC']
    element_type = eval(element_type)
    os.chdir(work_place)
    from Utility_function import (log, _get_nodes_coordinates,
                                  _build_rigid_anchor_region,
                                  _write_job_input_from_model)
    from PBC_constraint_builder import main_apply_pbc_constraint
    from job_resume import abaqus_api_path
    from shear_resume import preserve_clean_shear_base

    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_PATH, "a") as f:
        f.write("-" * 50 + "\n")
        f.write("new NIAH job - {}\n".format(current_time))
        f.flush()

    log("Starting the NIAH-ABAQUS workflow.", LOG_PATH, _t0)
    log("NIAH-ABAQUS: Pre-processing", LOG_PATH, _t0)

    """
    1. Initialization module:
    """

    # import inp
    inp_name = '{}.inp'.format(partname)
    mdb.ModelFromInputFile(
        name=partname,
        inputFileName=abaqus_api_path(
            os.path.join(data_out_path, inp_name)
        )
    )
    del mdb.models[model_name]
    mdb.models.changeKey(fromName=partname, toName=model_name)
    # assemble
    a = mdb.models[model_name].rootAssembly
    a.regenerate()
    parts_name = mdb.models[model_name].parts.keys()
    mdb.models[model_name].rootAssembly.features.changeKey(fromName=parts_name[0] + '-1', toName=instance_name)
    a.regenerate()
    parts_name = mdb.models[model_name].parts.keys()
    mdb.models[model_name].parts.changeKey(fromName=parts_name[0], toName=partname)
    p = mdb.models[model_name].parts[partname]
    p.renumberNode(startLabel=1, increment=1)
    p.renumberElement(startLabel=1, increment=1)

    if stru_type == 'shell':
        if periodicity_ch == 2:
            a.rotate(instanceList=(instance_name,), axisPoint=(0.0, 0.0, 0.0),
                     axisDirection=(10.0, 0.0, 0.0), angle=90.0)
            log('y is no periodicity!', LOG_PATH, _t0)
        elif periodicity_ch == 1:
            a.rotate(instanceList=(instance_name,), axisPoint=(0.0, 0.0, 0.0),
                     axisDirection=(0.0, 10.0, 0.0), angle=90.0)
            log('x is no periodicity!', LOG_PATH, _t0)
        else:
            log('z is no periodicity!', LOG_PATH, _t0)

    a.regenerate()

    elem_type = p.elements[0].type  # Abaqus return: 'S4R','B31','C3D8'
    if elem_type == element_type:
        log("Detected element type: %s" % elem_type, LOG_PATH, _t0)
    else:
        log("READ:{}, INPUT:{}".format(elem_type, element_type), LOG_PATH, _t0)
        raise RuntimeError("element type has not been assigned correctly.")

    """
    2. Preprocessing module:
    """
    # Step
    if 'Step-1' not in mdb.models[model_name].steps.keys():
        mdb.models[model_name].StaticStep(name='Step-1', previous='Initial')

    # Output
    if 'F-Output-1' in mdb.models[model_name].fieldOutputRequests.keys():
        mdb.models[model_name].fieldOutputRequests['F-Output-1'].setValues(
            variables=('U', 'UT', 'RF', 'RT', 'RM', 'S', 'E', 'UR')
        )

    if model_name_NOPBC in mdb.models:
        del mdb.models[model_name_NOPBC]

    mdb.Model(name=model_name_NOPBC, objectToCopy=mdb.models[model_name])
    # Create sets in Model-NOPBC
    _, _, _, _, error = main_apply_pbc_constraint(model_name_NOPBC, instance_name, meshsens, numCpus, 0)
    if error:
        raise RuntimeError("Sets creation failed.")

    if model_name_PBC in mdb.models:
        del mdb.models[model_name_PBC]

    mdb.Model(name=model_name_PBC, objectToCopy=mdb.models[model_name])
    _, _, _, _, error = main_apply_pbc_constraint(model_name_PBC, instance_name, meshsens, numCpus, condition_ch)
    if error:
        raise RuntimeError("PBC creation failed.")

    PBC_a = mdb.models[model_name_PBC].rootAssembly
    _nodes, _ = _get_nodes_coordinates(PBC_a, instance_name)
    region_rp, node_bd = _build_rigid_anchor_region(
        _nodes, rigid_node, rigid_node_tol
    )
    if 'BC-c7' in mdb.models[model_name_PBC].boundaryConditions:
        del mdb.models[model_name_PBC].boundaryConditions['BC-c7']

    # A translational anchor removes rigid translation while rotations
    # remain governed by the periodic equations.
    mdb.models[model_name_PBC].DisplacementBC(
        name='BC-c7', createStepName='Initial',
        region=region_rp, u1=0.0, u2=0.0, u3=0.0,
        ur1=UNSET, ur2=UNSET, ur3=UNSET
    )

    if model_name_NOPBCX in mdb.models:
        del mdb.models[model_name_NOPBCX]

    mdb.Model(name=model_name_NOPBCX, objectToCopy=mdb.models[model_name])

    if model_name_NOPBCY in mdb.models:
        del mdb.models[model_name_NOPBCY]

    mdb.Model(name=model_name_NOPBCY, objectToCopy=mdb.models[model_name])

    BASE_INP_NOPBC = _write_job_input_from_model(BASE_JOB_NOPBC, model_name_NOPBC, work_place, numCpus)
    BASE_INP_PBC = _write_job_input_from_model(BASE_JOB_PBC, model_name_PBC, work_place, numCpus)
    BASE_INP_NOPBCX = None
    BASE_INP_NOPBCY = None
    if stru_type == 'shell':
        BASE_INP_NOPBCX = _write_job_input_from_model(BASE_JOB_REALNOPBCX, model_name_NOPBCX, work_place, numCpus)
        BASE_INP_NOPBCY = _write_job_input_from_model(BASE_JOB_REALNOPBCY, model_name_NOPBCY, work_place, numCpus)
        if bool(runtime_input.get('resume_solver', False)):
            clean_x = preserve_clean_shear_base(
                work_place,
                BASE_JOB_REALNOPBCX,
                BASE_INP_NOPBCX
            )
            clean_y = preserve_clean_shear_base(
                work_place,
                BASE_JOB_REALNOPBCY,
                BASE_INP_NOPBCY
            )
            log(
                "Resume: preserved clean shear bases: %s, %s" %
                (os.path.basename(clean_x), os.path.basename(clean_y)),
                LOG_PATH,
                _t0
            )

    # log("Base inp written:", LOG_PATH, _t0)
    # log("  NOPBC: %s" % BASE_INP_NOPBC, LOG_PATH, _t0)
    # log("  PBC  : %s" % BASE_INP_PBC, LOG_PATH, _t0)
    # if stru_type == 'shell':
    #     log(" NOPBCX: %s" % BASE_INP_NOPBCX, LOG_PATH, _t0)
    #     log(" NOPBCY: %s" % BASE_INP_NOPBCY, LOG_PATH, _t0)

    log("Preprocessing completed.", LOG_PATH, _t0)
    pass
