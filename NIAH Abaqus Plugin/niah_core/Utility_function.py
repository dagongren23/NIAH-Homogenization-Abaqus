# -*-coding:UTF-8-*-
"""
NIAH-ABAQUS: Utility Function
Author: Zhihui Liu
Date:   2026/04/07
Email:  dutme_lzh@163.com
"""
import numpy as np
import time
import sys
from abaqus import *
from abaqusConstants import *
from caeModules import *
import os
from PBC_constraint_builder import main_apply_pbc_constraint
from boundary_ordering import (ensure_job_output_ready,
                               flatten_values_by_node_label,
                               require_rotational_field)
from shear_model_state import reset_imported_shear_base_model
from shear_resume import (build_shear_dependency_fingerprint,
                          find_active_clean_shear_base,
                          find_shear_chain_checkpoint,
                          record_shear_chain_checkpoint)
from job_resume import (abaqus_api_path,
                        acquire_odb_for_read,
                        choose_available_input_path,
                        choose_available_job_name,
                        find_reusable_job,
                        record_success_checkpoint,
                        sta_file_completed_successfully,
                        validate_completed_odb)
# ----------------------------- utility function -----------------------------
def log(msg, LOG_PATH, _t0):
    dt = time.time() - _t0
    line = "[%8.1fs] %s" % (dt, msg)
    print(line)
    sys.stdout.flush()
    # noinspection PyBroadException
    try:
        with open(LOG_PATH, "a") as f:
            f.write(line + "\n")
            f.flush()
    except Exception:
        pass


def _get_nodes_coordinates(assem_name, inst_name):
    """
    Returns
    -------
        nodes        : Abaqus nodes object
        coordinates       : (n,3) numpy array
    """
    nodes = assem_name.instances[inst_name].nodes
    n = len(nodes)
    coords = np.zeros((n, 3), dtype=float)
    for i, nd in enumerate(nodes):
        x, y, z = nd.coordinates
        coords[i, :] = (x, y, z)
    return nodes, coords


def _build_rigid_anchor_region(nodes, rigid_node, rigid_node_tol):
    """Return the exact GUI-selected node region and its Abaqus label.

    Abaqus node labels are identifiers, not zero-based array positions.  Using
    ``label - 1`` as an array index can select the wrong node when an imported
    mesh has non-contiguous labels.
    """
    try:
        closest_node = nodes.getClosest(
            rigid_node, 1, rigid_node_tol
        )
        node_label = closest_node.label
    except Exception as exc:
        raise ValueError(
            "No mesh node was found for rigid-node coordinate %s within "
            "tolerance %s. Check the GUI rigid-node coordinates/tolerance. "
            "Abaqus detail: %s" %
            (tuple(rigid_node), rigid_node_tol, exc)
        )

    anchor_nodes = nodes.sequenceFromLabels(labels=(node_label,))
    if len(anchor_nodes) != 1:
        raise ValueError(
            "Rigid-node label %s could not be resolved to exactly one mesh "
            "node. Check the imported node labels." % node_label
        )
    return regionToolset.Region(nodes=anchor_nodes), node_label


def _section2partition(fp, ele_size, fmeshsens, LOG_PATH, _t0):
    """
    The microstructure, composed of beam or shell elements,
    is divided into different regions according to the section properties
    assigned to each element.

    :param fp: object part
    :param ele_size: mm
    :param fmeshsens: tolerance
    :return: None
    """
    all_elems = fp.elements
    all_nodes = fp.nodes
    # -----------------------------
    # 0) calculate geometric bounding box
    # -----------------------------
    aal_x = []
    aal_y = []
    aal_z = []
    for nd in all_nodes:
        aal_x.append(nd.coordinates[0])
        aal_y.append(nd.coordinates[1])
        aal_z.append(nd.coordinates[2])
    Max = max(aal_x); May = max(aal_y); Maz = max(aal_z)
    Mnx = min(aal_x); Mny = min(aal_y); Mnz = min(aal_z)
    # -----------------------------
    # 1) coarse selection of boundary elements: union of six bounding boxes
    # -----------------------------
    bb_tol = 0.1 * ele_size  # or 0.1*ele_size
    boundary_maz = all_elems.getByBoundingBox(
        xMin=Mnx-bb_tol, yMin=Mny-bb_tol, zMin=Maz-bb_tol,
        xMax=Max+bb_tol, yMax=May+bb_tol, zMax=Maz+bb_tol)
    boundary_mnz = all_elems.getByBoundingBox(
        xMin=Mnx-bb_tol, yMin=Mny-bb_tol, zMin=Mnz-bb_tol,
        xMax=Max+bb_tol, yMax=May+bb_tol, zMax=Mnz+bb_tol)
    boundary_may = all_elems.getByBoundingBox(
        xMin=Mnx-bb_tol, yMin=May-bb_tol, zMin=Mnz-bb_tol,
        xMax=Max+bb_tol, yMax=May+bb_tol, zMax=Maz+bb_tol)
    boundary_mny = all_elems.getByBoundingBox(
        xMin=Mnx-bb_tol, yMin=Mny-bb_tol, zMin=Mnz-bb_tol,
        xMax=Max+bb_tol, yMax=Mny+bb_tol, zMax=Maz+bb_tol)
    boundary_max = all_elems.getByBoundingBox(
        xMin=Max-bb_tol, yMin=Mny-bb_tol, zMin=Mnz-bb_tol,
        xMax=Max+bb_tol, yMax=May+bb_tol, zMax=Maz+bb_tol)
    boundary_mnx = all_elems.getByBoundingBox(
        xMin=Mnx-bb_tol, yMin=Mny-bb_tol, zMin=Mnz-bb_tol,
        xMax=Mnx+bb_tol, yMax=May+bb_tol, zMax=Maz+bb_tol)
    boundary_labels = set()
    for es in (boundary_maz, boundary_mnz, boundary_may, boundary_mny, boundary_max, boundary_mnx):
        for ele in es:
            boundary_labels.add(ele.label)
    # label->element mapping (to avoid the risk of all_elements[lab-1])
    elem_by_label = {}
    for ele in all_elems:
        elem_by_label[ele.label] = ele
    # -----------------------------
    # 2) utility function: center point & boundary attachment determination
    # -----------------------------
    def elem_centroid(elem):
        xs = 0.0; ys = 0.0; zs = 0.0
        n = len(elem.connectivity)
        for nl in elem.connectivity:
            x, y, z = fp.nodes[nl].coordinates
            xs += x; ys += y; zs += z
        return xs / n, ys / n, zs / n
    def near(a, b, tol):
        return abs(a - b) <= tol
    def on_x_boundary(x):
        return near(x, Mnx, fmeshsens) or near(x, Max, fmeshsens)
    def on_y_boundary(y):
        return near(y, Mny, fmeshsens) or near(y, May, fmeshsens)
    def on_z_boundary(z):
        return near(z, Mnz, fmeshsens) or near(z, Maz, fmeshsens)
    # -----------------------------
    # 3) only judge "on the edge":
    # -----------------------------
    edge_labels = set()
    for lab in boundary_labels:
        ele = elem_by_label[lab]
        cx, cy, cz = elem_centroid(ele)
        onx = on_x_boundary(cx)
        ony = on_y_boundary(cy)
        onz = on_z_boundary(cz)
        hit = (1 if onx else 0) + (1 if ony else 0) + (1 if onz else 0)
        # "on the edge"= hit>=2
        if hit >= 2:
            edge_labels.add(lab)
    # In-plane
    face_labels = boundary_labels - edge_labels
    # inner-body
    all_labels = set([ele.label for ele in all_elems])
    inner_labels = all_labels - boundary_labels
    # -----------------------------
    # 4) Establish element sets at the Part level.
    # -----------------------------
    def elems_from_labels(labels_set):
        if len(labels_set) == 0:
            return None
        return all_elems.sequenceFromLabels(labels=tuple(sorted(list(labels_set))))
    for sname in ['ELEM_BOUNDARY', 'ELEM_FACE', 'ELEM_EDGE', 'ELEM_INNER']:
        if fp.sets.has_key(sname):
            del fp.sets[sname]
    boundary_elems = elems_from_labels(boundary_labels)
    face_elems     = elems_from_labels(face_labels)
    edge_elems     = elems_from_labels(edge_labels)
    inner_elems    = elems_from_labels(inner_labels)
    if boundary_elems: fp.Set(name='ELEM_BOUNDARY', elements=boundary_elems)
    if face_elems:     fp.Set(name='ELEM_FACE', elements=face_elems)
    if edge_elems:     fp.Set(name='ELEM_EDGE', elements=edge_elems)
    if inner_elems:    fp.Set(name='ELEM_INNER', elements=inner_elems)
    log('Boundary elems: %d' % len(boundary_labels), LOG_PATH, _t0)
    log('Face elems    : %d' % len(face_labels), LOG_PATH, _t0)
    log('Edge elems    : %d' % len(edge_labels), LOG_PATH, _t0)
    log('Inner elems   : %d' % len(inner_labels), LOG_PATH, _t0)


def _write_job_input_from_model(job_name, fmodel_name, work_place, numCpus):
    """
    create inp file

    :param job_name:
    :param fmodel_name:
    :return: job name
    """
    mdb.Job(
        name=job_name, model=fmodel_name, description='', type=ANALYSIS,
        memory=90, memoryUnits=PERCENTAGE, getMemoryFromAnalysis=True,
        explicitPrecision=SINGLE, nodalOutputPrecision=SINGLE,
        echoPrint=OFF, modelPrint=OFF, contactPrint=OFF, historyPrint=OFF,
        resultsFormat=ODB, multiprocessingMode=DEFAULT,
        numCpus=numCpus, numDomains=numCpus, numGPUs=0
    )
    mdb.jobs[job_name].writeInput(consistencyChecking=OFF)
    return os.path.join(work_place, job_name + '.inp')


def _build_3D_X1(nodes_coords, dof_per_node = 3):
    x = nodes_coords[:, 0].copy()
    y = nodes_coords[:, 1].copy()
    z = nodes_coords[:, 2].copy()
    x -= 0.5 * (x.max() + x.min())
    y -= 0.5 * (y.max() + y.min())
    z -= 0.5 * (z.max() + z.min())
    nnode = nodes_coords.shape[0]
    X1 = np.zeros((6, nnode * dof_per_node))
    if dof_per_node == 6:
        # 0: unit xx normal strain -> u = x
        X1[0, 0::6] = x
        # 1: unit yy normal strain -> v = y
        X1[1, 1::6] = y
        # 2: unit zz normal strain -> w = z
        X1[2, 2::6] = z
        # 3: unit yz engineering shear strain -> v = z/2, w = y/2
        X1[3, 1::6] = z / 2.0
        X1[3, 2::6] = y / 2.0
        # 4: unit xz engineering shear strain -> u = z/2, w = x/2
        X1[4, 0::6] = z / 2.0
        X1[4, 2::6] = x / 2.0
        # 5: unit xy engineering shear strain -> u = y/2, v = x/2
        X1[5, 0::6] = y / 2.0
        X1[5, 1::6] = x / 2.0
    elif dof_per_node == 3:
        X1[0, 0::3] = x
        # 1: unit yy normal strain -> v = y
        X1[1, 1::3] = y
        # 2: unit zz normal strain -> w = z
        X1[2, 2::3] = z
        # 3: unit yz engineering shear strain -> v = z/2, w = y/2
        X1[3, 1::3] = z / 2.0
        X1[3, 2::3] = y / 2.0
        # 4: unit xz engineering shear strain -> u = z/2, w = x/2
        X1[4, 0::3] = z / 2.0
        X1[4, 2::3] = x / 2.0
        # 5: unit xy engineering shear strain -> u = y/2, v = x/2
        X1[5, 0::3] = y / 2.0
        X1[5, 1::3] = x / 2.0
    else:
        raise ValueError('dof_per_node be 6 or 3')

    return X1


def _build_v0_vstar_vectors(nodes_coords, kl_tag, dof_per_node = 3):
    """
    For Shell homogenization, dof_per_node = 3 or 6.
    """
    x = nodes_coords[:, 0].copy()
    y = nodes_coords[:, 1].copy()
    z = nodes_coords[:, 2].copy()
    x -= 0.5 * (x.max() + x.min())
    y -= 0.5 * (y.max() + y.min())
    z -= 0.5 * (z.max() + z.min())
    nnode = nodes_coords.shape[0]
    v0 = np.zeros((nnode * dof_per_node,), dtype=float)
    vs = np.zeros((nnode * dof_per_node,), dtype=float)
    if dof_per_node == 6:
        kxx = kyy = kxy = 1.0
        if kl_tag == '11':
            v0[0::6] = x
            vs[0::6] = kxx * z * x
            vs[2::6] = -0.5 * kxx * x * x
            vs[4::6] = kxx * x  # UR2
        elif kl_tag == '22':
            v0[1::6] = y
            vs[1::6] = kyy * z * y
            vs[2::6] = -0.5 * kyy * y * y
            vs[3::6] = -kyy * y  # UR1
        elif kl_tag == '12':
            v0[0::6] = 0.5 * y
            v0[1::6] = 0.5 * x
            vs[0::6] = 0.5 * kxy * z * y
            vs[1::6] = 0.5 * kxy * z * x
            vs[2::6] = -0.5 * kxy * x * y
            vs[3::6] = -0.5 * kxy * x
            vs[4::6] = 0.5 * kxy * y
        else:
            raise ValueError('kl_tag must be one of: 11, 22, 12')
    elif dof_per_node == 3:
        if kl_tag == '11':
            v0[0::3] = x
            vs[0::3] = z * x
            vs[2::3] = -0.5 * x * x
        elif kl_tag == '22':
            v0[1::3] = y
            vs[1::3] = z * y
            vs[2::3] = -0.5 * y * y
        elif kl_tag == '12':
            v0[0::3] = 0.5 * y
            v0[1::3] = 0.5 * x
            vs[0::3] = 0.5 * z * y
            vs[1::3] = 0.5 * z * x
            vs[2::3] = -0.5 * x * y
        else:
            raise ValueError('kl_tag must be one of: 11, 22, 12')

    return v0, vs


def _odb_has_completed_step(odb_path, job_name, raise_error):
    def _open_odb(path):
        return session.openOdb(name=abaqus_api_path(path))

    is_valid, error_detail, close_warning = validate_completed_odb(
        session.odbs,
        odb_path,
        _open_odb
    )
    if close_warning:
        print(
            "NIAH resume: warning: cannot close ODB opened for validation "
            "of job %s: %s" % (job_name, close_warning)
        )
    if is_valid:
        return True
    message = "Cannot validate ODB for job %s at %s: %s" % (
        job_name,
        odb_path,
        error_detail or 'unknown validation failure'
    )
    if raise_error:
        raise RuntimeError(message)
    print("NIAH resume: %s" % message)
    return False


def _acquire_odb_for_read(odb_path):
    """Open or reuse an ODB and report whether this function owns it."""
    def _open_odb(path):
        return session.openOdb(name=abaqus_api_path(path))

    return acquire_odb_for_read(session.odbs, odb_path, _open_odb)


def _close_odb_if_owned(odb_obj, opened_here):
    """Close only ODBs opened by the matching acquire operation."""
    if opened_here and odb_obj is not None:
        odb_obj.close()


def _run_abaqus_input(work_place, job_name, inp_path, cpus,
                      resume_solver=False):
    """Run or reuse one logical Abaqus job and return its actual ODB path."""
    if resume_solver:
        cached = find_reusable_job(work_place, job_name, inp_path)
        if cached is not None:
            if _odb_has_completed_step(
                    cached['odb_path'],
                    cached['actual_job_name'],
                    raise_error=False):
                print(
                    "NIAH resume: reuse completed job %s from %s" %
                    (job_name, cached['actual_job_name'])
                )
                return cached['odb_path']
            print(
                "NIAH resume: checkpoint for %s could not be validated; "
                "see the ODB validation error above. A retry will be "
                "submitted." % job_name
            )

    actual_job_name = choose_available_job_name(
        work_place,
        job_name,
        occupied_job_names=list(mdb.jobs.keys())
    )
    if actual_job_name != job_name:
        print(
            "NIAH resume: preserve existing artifacts; submit %s as %s." %
            (job_name, actual_job_name)
        )

    mdb.JobFromInputFile(
        name=actual_job_name,
        inputFileName=abaqus_api_path(inp_path),
        type=ANALYSIS,
        numCpus=cpus,
        numDomains=cpus,
        multiprocessingMode=DEFAULT,
        memory=90,
        memoryUnits=PERCENTAGE,
        getMemoryFromAnalysis=True
    )
    mdb.jobs[actual_job_name].submit(consistencyChecking=OFF)
    mdb.jobs[actual_job_name].waitForCompletion()

    odb_path = os.path.join(work_place, actual_job_name + '.odb')
    sta_path = os.path.join(work_place, actual_job_name + '.sta')
    lock_path = os.path.join(work_place, actual_job_name + '.lck')
    ensure_job_output_ready(
        actual_job_name,
        mdb.jobs[actual_job_name].status,
        os.path.exists(odb_path),
        os.path.exists(lock_path)
    )
    if not sta_file_completed_successfully(sta_path):
        raise RuntimeError(
            "Job %s has no successful-completion marker in its STA file. "
            "Expected: THE ANALYSIS HAS COMPLETED SUCCESSFULLY. "
            "Check %s.msg/.dat/.sta." %
            (actual_job_name, actual_job_name)
        )
    _odb_has_completed_step(odb_path, actual_job_name, raise_error=True)
    record_success_checkpoint(
        work_place,
        job_name,
        actual_job_name,
        inp_path,
        odb_path,
        sta_path
    )
    return odb_path


def _inject_block_into_step(inp_in, inp_out, step_name, block_lines):
    """
    Inject block_lines after *Step and before *End Step (try not to disrupt other segments).
    For most standard input parameters (inp), this injection method is sufficient.
    """
    with open(inp_in, 'r') as f:
        lines = f.readlines()
    step_start = None
    step_end = None
    for i, ln in enumerate(lines):
        if ln.strip().lower().startswith('*step'):
            step_start = i
        if step_start is not None and ln.strip().lower().startswith('*end step'):
            step_end = i
            break
    if step_start is None or step_end is None:
        raise RuntimeError('Cannot find *Step / *End Step in inp: %s' % inp_in)
    new_lines = []
    new_lines.extend(lines[:step_end])  # until the *End Step
    # Insert our block before the end
    new_lines.append('** --- injected by python ---\n')
    for bl in block_lines:
        if not bl.endswith('\n'):
            bl += '\n'
        new_lines.append(bl)
    new_lines.append('** --- end injected ---\n')
    new_lines.extend(lines[step_end:])
    with open(inp_out, 'w') as f:
        f.writelines(new_lines)


def _make_boundary_block_from_disp(disp_vec, all_nodes_num, instance_name, dof_per_node=3):
    """
    Generate *BOUNDARY blocks (node-by-node)

    For beam/shell elements:
      - DOF 1-3 represent nodal translation (corresponding to U1/2/3)
      - DOF 4-6 represent node rotation (corresponding to UR1/2/3)

    For solid elements: only DOF 1-3.
    """

    blk = ['*BOUNDARY']
    for nlab in range(1, all_nodes_num + 1):
        node_key = '%s.%d' % (instance_name, nlab)   # instance prefix
        base = dof_per_node * (nlab - 1)
        if dof_per_node == 3:
            # translations (U1,U2,U3)
            u1 = float(disp_vec[base + 0])
            u2 = float(disp_vec[base + 1])
            u3 = float(disp_vec[base + 2])
            blk.append('%s, 1, 1, %.16e' % (node_key, u1))
            blk.append('%s, 2, 2, %.16e' % (node_key, u2))
            blk.append('%s, 3, 3, %.16e' % (node_key, u3))
        elif dof_per_node == 6:
            # translations (U1,U2,U3)
            u1 = float(disp_vec[base + 0])
            u2 = float(disp_vec[base + 1])
            u3 = float(disp_vec[base + 2])
            blk.append('%s, 1, 1, %.16e' % (node_key, u1))
            blk.append('%s, 2, 2, %.16e' % (node_key, u2))
            blk.append('%s, 3, 3, %.16e' % (node_key, u3))
            # rotations (UR1,UR2,UR3)
            ur1 = float(disp_vec[base + 3])
            ur2 = float(disp_vec[base + 4])
            ur3 = float(disp_vec[base + 5])
            blk.append('%s, 4, 4, %.16e' % (node_key, ur1))
            blk.append('%s, 5, 5, %.16e' % (node_key, ur2))
            blk.append('%s, 6, 6, %.16e' % (node_key, ur3))
        else:
            raise RuntimeError("dof_per_node has not been assigned correctly.")

    return blk


def _make_cload_block_from_force(force_vec, all_nodes_num, instance_name, dof_per_node=3, eps_load=0.0):
    """
    Generate *CLOAD blocks (node-by-node)

    For beam/shell elements:
      - DOF 1-3 represent nodal forces (corresponding to RF1/2/3)
      - DOF 4-6 represent node moments (corresponding to RM1/2/3)

    For solid elements: only DOF 1-3.
    """
    blk = ['*CLOAD']
    for nlab in range(1, all_nodes_num + 1):
        node_key = '%s.%d' % (instance_name, nlab)   # instance prefix
        base = dof_per_node * (nlab - 1)
        if dof_per_node == 3:
            # RF1/2/3
            fx = float(force_vec[base + 0]) + eps_load
            fy = float(force_vec[base + 1]) + eps_load
            fz = float(force_vec[base + 2]) + eps_load
            blk.append('%s, 1, %.16e' % (node_key, fx))
            blk.append('%s, 2, %.16e' % (node_key, fy))
            blk.append('%s, 3, %.16e' % (node_key, fz))
        elif dof_per_node == 6:
            # RF1/2/3
            fx = float(force_vec[base + 0]) + eps_load
            fy = float(force_vec[base + 1]) + eps_load
            fz = float(force_vec[base + 2]) + eps_load
            blk.append('%s, 1, %.16e' % (node_key, fx))
            blk.append('%s, 2, %.16e' % (node_key, fy))
            blk.append('%s, 3, %.16e' % (node_key, fz))
            # RM1/2/3
            mx = float(force_vec[base + 3]) + eps_load
            my = float(force_vec[base + 4]) + eps_load
            mz = float(force_vec[base + 5]) + eps_load
            blk.append('%s, 4, %.16e' % (node_key, mx))
            blk.append('%s, 5, %.16e' % (node_key, my))
            blk.append('%s, 6, %.16e' % (node_key, mz))
        else:
            raise RuntimeError("dof_per_node has not been assigned correctly.")

    return blk


def _extract_reaction_vector_from_odb(odb_path, field_name, instance_name, all_nodes_num, dof_per_node=3,
                                      force_field='RF', moment_field='RM'):
    """
    Return a vector with shape=(all_nodes_num*dof_per_node,1), whose order strictly corresponds to nodeLabel=1...all_nodes_num.
    - Solid: RF(3).
    - Beam/shell: RF(3) + RM(3) -> 6 components.
      Missing RM is treated as an invalid 6-DOF result and raises RuntimeError.
    Automatically exclude nodes (such as RP) that do not belong to instance_name.
    """
    odb_obj, opened_here = _acquire_odb_for_read(odb_path)
    try:
        fr = odb_obj.steps['Step-1'].frames[-1]
        inst = odb_obj.rootAssembly.instances[instance_name]
        require_rotational_field(fr.fieldOutputs, dof_per_node, moment_field)
        # RF
        fo_f = fr.fieldOutputs[force_field].getSubset(region=inst)
        arr = np.zeros((all_nodes_num, dof_per_node), dtype=float)
        for v in fo_f.values:
            lab = v.nodeLabel
            if 1 <= lab <= all_nodes_num:
                arr[lab - 1, 0:3] = v.data[0:3]
        # RM
        if dof_per_node == 6:
            fo_m = fr.fieldOutputs[moment_field].getSubset(region=inst)
            for v in fo_m.values:
                lab = v.nodeLabel
                if 1 <= lab <= all_nodes_num:
                    # RM is a 3-vector (about 1,2,3 axes)
                    arr[lab - 1, 3:6] = v.data[0:3]
        return arr.flatten()
    finally:
        _close_odb_if_owned(odb_obj, opened_here)


def _extract_displacement_vector_from_odb(odb_path, instance_name, all_nodes_num, dof_per_node=3,
                                          u_field='U', ur_field='UR'):
    """
    field_name is recommended to use 'U' (the one you used before) or 'UT'
    The returned shape is (all_nodes_num*6, 1) with strict alignment, and nodeLabel ranges from 1 to all_nodes_num
    """
    odb_obj, opened_here = _acquire_odb_for_read(odb_path)
    try:
        fr = odb_obj.steps['Step-1'].frames[-1]
        inst = odb_obj.rootAssembly.instances[instance_name]
        require_rotational_field(fr.fieldOutputs, dof_per_node, ur_field)
        arr = np.zeros((all_nodes_num, dof_per_node), dtype=float)
        # U
        if u_field in fr.fieldOutputs.keys():
            fo_u = fr.fieldOutputs[u_field].getSubset(region=inst)
            for u in fo_u.values:
                lab = u.nodeLabel
                if 1 <= lab <= all_nodes_num:
                    arr[lab - 1, 0:3] = u.data[0:3]
        # UR
        if dof_per_node == 6:
            fo_ur = fr.fieldOutputs[ur_field].getSubset(region=inst)
            for r in fo_ur.values:
                lab = r.nodeLabel
                if 1 <= lab <= all_nodes_num:
                    arr[lab - 1, 3:6] = r.data[0:3]
        return arr.flatten()
    finally:
        _close_odb_if_owned(odb_obj, opened_here)


def _extract_reaction_vector6dof_from_odb2stat4(odb_path, instance_name, all_nodes_num, dof_per_node=3, field_name='RF', moment_field='RM'):
    """
    Returns
    -------
        f1xminus: The reaction forces of nodes in the backbc node set (in the order within the set)
        f1yminus: The reaction forces of nodes in the botbc node set (in the order within the set)
    """
    odb, opened_here = _acquire_odb_for_read(odb_path)
    try:
        fr = odb.steps['Step-1'].frames[-1]
        require_rotational_field(fr.fieldOutputs, dof_per_node, moment_field)
        fo = fr.fieldOutputs[field_name]
        back_set = odb.rootAssembly.nodeSets['BACKBC']  # 'backbc'.upper()
        bot_set  = odb.rootAssembly.nodeSets['BOTBC']   # 'botbc'.upper()
        fo_back = fo.getSubset(region=back_set)
        fo_bot  = fo.getSubset(region=bot_set)
        arr_back = np.zeros((all_nodes_num, dof_per_node), dtype=float)
        for v in fo_back.values:
            lab = v.nodeLabel
            if 1 <= lab <= all_nodes_num:
                arr_back[lab - 1, 0] = v.data[0]
                arr_back[lab - 1, 1] = v.data[1]
                arr_back[lab - 1, 2] = v.data[2]
        arr_bot = np.zeros((all_nodes_num, dof_per_node), dtype=float)
        for v in fo_bot.values:
            lab = v.nodeLabel
            if 1 <= lab <= all_nodes_num:
                arr_bot[lab - 1, 0] = v.data[0]
                arr_bot[lab - 1, 1] = v.data[1]
                arr_bot[lab - 1, 2] = v.data[2]
        # RM
        if dof_per_node == 6:
            fo_m_back = fr.fieldOutputs[moment_field].getSubset(region=back_set)
            fo_m_bot = fr.fieldOutputs[moment_field].getSubset(region=bot_set)
            for v in fo_m_back.values:
                lab = v.nodeLabel
                if 1 <= lab <= all_nodes_num:
                    # RM is a 3-vector (about 1,2,3 axes)
                    arr_back[lab - 1, 3:6] = v.data[0:3]
            for v in fo_m_bot.values:
                lab = v.nodeLabel
                if 1 <= lab <= all_nodes_num:
                    # RM is a 3-vector (about 1,2,3 axes)
                    arr_bot[lab - 1, 3:6] = v.data[0:3]
        return arr_back.flatten(), arr_bot.flatten()
    finally:
        _close_odb_if_owned(odb, opened_here)


def _extract_displacement_vector6dof_from_odb2stat2(odb_path, all_nodes_num, dof_per_node=3, field_name='U', ur_field = 'UR'):
    """
    Returns
    -------
        u1xminus: Displacement in the backbc node set (in the order within the set)
        u1yminus: displacement in the botbc node set (in the order within the set)
    """
    odb, opened_here = _acquire_odb_for_read(odb_path)
    try:
        fr = odb.steps['Step-1'].frames[-1]
        require_rotational_field(fr.fieldOutputs, dof_per_node, ur_field)
        fo = fr.fieldOutputs[field_name]
        back_set = odb.rootAssembly.nodeSets['BACKBC']
        bot_set  = odb.rootAssembly.nodeSets['BOTBC']
        fo_back = fo.getSubset(region=back_set)
        fo_bot = fo.getSubset(region=bot_set)
        fo_ur_back = None
        fo_ur_bot = None
        if dof_per_node == 6:
            fo_ur_back = fr.fieldOutputs[ur_field].getSubset(region=back_set)
            fo_ur_bot = fr.fieldOutputs[ur_field].getSubset(region=bot_set)
        arr_back = flatten_values_by_node_label(
            fo_back.values,
            dof_per_node=dof_per_node,
            rotation_values=fo_ur_back.values if fo_ur_back is not None else None
        )
        arr_bot = flatten_values_by_node_label(
            fo_bot.values,
            dof_per_node=dof_per_node,
            rotation_values=fo_ur_bot.values if fo_ur_bot is not None else None
        )
        return arr_back, arr_bot
    finally:
        _close_odb_if_owned(odb, opened_here)


def _niah_chain_one_case_inp(case_name_prefix, disp_vec, work_place, all_nodes_num, instanceName, numCpus,
                             reaction_field_name, BASE_INP_PBC, BASE_INP_NOPBC, LOG_PATH, _t0, dof_node = 0,
                             resume_solver=False):
    """
    (1) NOPBC: Displacement -> FX1
    (2) PBC: Force -> X2
    (3) NOPBC: Displacement X2 -> FX2
    BASE_INP_NOPBC / BASE_INP_PBC are only written once in the main program.
    """
    # (1) NOPBC model: inject boundary displacement X1.
    run_inp = case_name_prefix + '_FX1'
    run_inp_path = choose_available_input_path(work_place, run_inp)
    blk = _make_boundary_block_from_disp(disp_vec, all_nodes_num, instanceName, dof_per_node = dof_node)
    _inject_block_into_step(BASE_INP_NOPBC, run_inp_path, 'Step-1', blk)
    log("-----Start submitting job : %s ..." % run_inp, LOG_PATH, _t0)
    odb_path1 = _run_abaqus_input(
        work_place, run_inp, run_inp_path, numCpus, resume_solver=resume_solver
    )
    modif1 = _extract_reaction_vector_from_odb(odb_path1, reaction_field_name,
                                                instanceName.upper(), all_nodes_num, dof_per_node=dof_node)
    # (2) PBC model: inject concentrated nodal load FX1.
    run_inp2 = case_name_prefix + '_X2'
    run_inp2_path = choose_available_input_path(work_place, run_inp2)
    blk2 = _make_cload_block_from_force(-modif1, all_nodes_num, instanceName, dof_per_node = dof_node, eps_load=0.0)
    _inject_block_into_step(BASE_INP_PBC, run_inp2_path, 'Step-1', blk2)
    log("-----Start submitting job : %s ..." % run_inp2, LOG_PATH, _t0)
    odb_path2 = _run_abaqus_input(
        work_place, run_inp2, run_inp2_path, numCpus, resume_solver=resume_solver
    )
    modiu2 = _extract_displacement_vector_from_odb(odb_path2,
                                                   instanceName.upper(), all_nodes_num, dof_per_node=dof_node)
    # (3) NOPBC model: inject boundary displacement X2.
    run_inp3 = case_name_prefix + '_FX2'
    run_inp3_path = choose_available_input_path(work_place, run_inp3)
    blk3 = _make_boundary_block_from_disp(modiu2, all_nodes_num, instanceName, dof_per_node = dof_node)
    _inject_block_into_step(BASE_INP_NOPBC, run_inp3_path, 'Step-1', blk3)
    log("-----Start submitting job : %s ..." % run_inp3, LOG_PATH, _t0)
    odb_path3 = _run_abaqus_input(
        work_place, run_inp3, run_inp3_path, numCpus, resume_solver=resume_solver
    )
    modif2 = _extract_reaction_vector_from_odb(odb_path3, reaction_field_name,
                                                instanceName.upper(), all_nodes_num, dof_per_node=dof_node)
    return modif1, modiu2, modif2, odb_path2


def _niah_chain_one_case_inp_additional(case_name_prefix, disp_vec, work_place, all_nodes_num, instanceName, numCpus,
                             reaction_field_name, BASE_INP_PBC, BASE_INP_NOPBC, LOG_PATH, _t0, dof_node = 0,
                             resume_solver=False, x2_odb_path=None):
    # (4) NOPBC model: inject boundary displacement X1 + X2.
    run_inp4 = case_name_prefix + '_FX3'
    run_inp4_path = choose_available_input_path(work_place, run_inp4)
    if x2_odb_path is None:
        raise RuntimeError(
            "x2_odb_path is required for the additional chain of %s." %
            case_name_prefix
        )
    modiu2 = _extract_displacement_vector_from_odb(x2_odb_path,
                                                   instanceName.upper(), all_nodes_num, dof_per_node = dof_node)
    uixminus, uiyminus = _extract_displacement_vector6dof_from_odb2stat2(x2_odb_path,
                                                                        all_nodes_num, dof_per_node = dof_node)
    blk4 = _make_boundary_block_from_disp(disp_vec + modiu2, all_nodes_num, instanceName, dof_per_node = dof_node)
    _inject_block_into_step(BASE_INP_NOPBC, run_inp4_path, 'Step-1', blk4)
    log("-----Start submitting job : %s ..." % run_inp4, LOG_PATH, _t0)
    odb_path4 = _run_abaqus_input(
        work_place, run_inp4, run_inp4_path, numCpus, resume_solver=resume_solver
    )
    fixminus, fiyminus = _extract_reaction_vector6dof_from_odb2stat4(
        odb_path4,
        instanceName.upper(),
        all_nodes_num,
        dof_per_node=dof_node,
        field_name=reaction_field_name
    )
    return fixminus, fiyminus, uixminus, uiyminus


# =========================
# Shear identification from energies
# =========================
def _niah_share_one_case_inp(case_name_prefix, disp_vec, mode, work_place, all_nodes_num, instanceName, numCpus,
                             reaction_field_name, BASE_INP_NOPBC, BASE_JOB_REALNOPBC, BASE_INP_REALNOPBC, meshsens,
                             LOG_PATH, _t0, rigid_node, rigid_node_tol, us1minus=None, fs1minus=None, dof_node = None,
                             resume_solver=False):
    """
    (1) NOPBC: Displacement: shear1u1 -> shear1f1
    (2) Special non-homogeneous PBC: -force shear1f1  -> shear1u2
    (3) NOPBC: Displacement shear1u2 -> shear1f2
    BASE_INP_NOPBC / BASE_INP_PBC are only written once in the main program.
    """
    clean_shear_base = None
    if resume_solver:
        clean_shear_base = find_active_clean_shear_base(
            work_place,
            BASE_JOB_REALNOPBC
        )
    if clean_shear_base is None:
        clean_shear_base = BASE_INP_REALNOPBC
        if resume_solver:
            print(
                "NIAH resume: no preserved clean base for %s; "
                "legacy imported-model cleanup remains enabled." %
                BASE_JOB_REALNOPBC
            )

    dependency_fingerprint = build_shear_dependency_fingerprint(
        metadata={
            'case_prefix': case_name_prefix,
            'mode': mode,
            'instance_name': instanceName.upper(),
            'all_nodes_num': int(all_nodes_num),
            'dof_node': int(dof_node),
            'meshsens': float(meshsens),
            'rigid_node': tuple(rigid_node),
            'rigid_node_tol': float(rigid_node_tol),
            'reaction_field_name': reaction_field_name,
        },
        file_paths={
            'base_nopbc': BASE_INP_NOPBC,
            'clean_shear_base': clean_shear_base,
            'pbc_builder': os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                'PBC_constraint_builder.py'
            ),
            'boundary_ordering': os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                'boundary_ordering.py'
            ),
            'periodic_mesh': os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                'periodic_mesh.py'
            ),
        },
        arrays={
            'disp_vec': disp_vec,
            'us1minus': us1minus,
            'fs1minus': fs1minus,
        }
    )

    reusable_chain = None
    reusable_stages = {'SH1': False, 'SH2': False, 'SH3': False}
    if resume_solver:
        reusable_chain = find_shear_chain_checkpoint(
            work_place,
            case_name_prefix,
            dependency_fingerprint
        )
        upstream_valid = True
        if reusable_chain is not None:
            for stage_name in ('SH1', 'SH2', 'SH3'):
                stage = reusable_chain['stages'][stage_name]
                stage_valid = bool(stage['artifact_valid'] and upstream_valid)
                if stage_valid:
                    stage_valid = _odb_has_completed_step(
                        stage['odb_path'],
                        stage['actual_job_name'],
                        raise_error=False
                    )
                reusable_stages[stage_name] = stage_valid
                upstream_valid = stage_valid

    if all(reusable_stages.values()):
        print(
            "NIAH resume: reuse complete shear chain %s; "
            "skip CAE import, PBC reconstruction and INP generation." %
            case_name_prefix
        )
        stage1 = reusable_chain['stages']['SH1']
        stage2 = reusable_chain['stages']['SH2']
        stage3 = reusable_chain['stages']['SH3']
        shear1f1 = _extract_reaction_vector_from_odb(
            stage1['odb_path'],
            reaction_field_name,
            instanceName.upper(),
            all_nodes_num,
            dof_per_node=dof_node
        )
        shear1u2 = _extract_displacement_vector_from_odb(
            stage2['odb_path'],
            instanceName.upper(),
            all_nodes_num,
            dof_per_node=dof_node
        )
        shear1f2 = _extract_reaction_vector_from_odb(
            stage3['odb_path'],
            reaction_field_name,
            instanceName.upper(),
            all_nodes_num,
            dof_per_node=dof_node
        )
        return shear1f1, shear1u2, shear1f2

    # (1) NOPBC model: inject the first shear displacement field.
    run_inp = case_name_prefix + '_SH1'
    if reusable_stages['SH1']:
        odb_path1 = reusable_chain['stages']['SH1']['odb_path']
        print(
            "NIAH resume: reuse shear-chain stage %s_SH1." %
            case_name_prefix
        )
    else:
        run_inp_path = choose_available_input_path(work_place, run_inp)
        blk = _make_boundary_block_from_disp(disp_vec, all_nodes_num, instanceName, dof_per_node=dof_node)
        _inject_block_into_step(BASE_INP_NOPBC, run_inp_path, 'Step-1', blk)
        log("-----Start submitting job : %s ..." % run_inp, LOG_PATH, _t0)
        odb_path1 = _run_abaqus_input(
            work_place, run_inp, run_inp_path, numCpus, resume_solver=resume_solver
        )
    shear1f1 = _extract_reaction_vector_from_odb(odb_path1, reaction_field_name,
                                                instanceName.upper(), all_nodes_num, dof_per_node=dof_node)
    if reusable_stages['SH2']:
        odb_path5 = reusable_chain['stages']['SH2']['odb_path']
        print(
            "NIAH resume: reuse shear-chain stage %s_SH2; "
            "skip CAE import and PBC reconstruction." %
            case_name_prefix
        )
        shear1u2 = _extract_displacement_vector_from_odb(
            odb_path5,
            instanceName.upper(),
            all_nodes_num,
            dof_per_node=dof_node
        )
    else:
        # (2) NHPBC model: inject the corrected shear nodal-force field.
        model_name_sh1 = None
        if mode == 'SHXZ':
            model_name_sh1 = 'BASE_realNOPBCX'
        elif mode == 'SHYZ':
            model_name_sh1 = 'BASE_realNOPBCY'
        else:
            raise ValueError("mode must be either SHXZ or SHYZ.")

        Mdb()
        mdb.ModelFromInputFile(
            name=model_name_sh1,
            inputFileName=abaqus_api_path(clean_shear_base)
        )
        model = mdb.models[model_name_sh1]
        removed_state = reset_imported_shear_base_model(model)
        print(
            "NIAH shear: reset imported base state "
            "(sets=%d, constraints=%d, boundary_conditions=%d, "
            "reference_points=%d)." %
            (
                removed_state['sets'],
                removed_state['constraints'],
                removed_state['boundary_conditions'],
                removed_state['reference_point_features']
            )
        )
        a = model.rootAssembly
        # Apply PBC once on BASE_realNOPBC (just once!)
        if dof_node == 3:
            condition_ch = 5
        elif dof_node == 6:
            condition_ch = 6
        else:
            raise ValueError("dof_node must be either 3 or 6.")

        _, _, _, _, error = main_apply_pbc_constraint(
            model_name=model_name_sh1,
            instance_name=instanceName.upper(),
            meshsens=meshsens,
            CPU=numCpus,
            condition_ch=condition_ch,
            mode=mode,
            us1minus=us1minus,
            fs1minus=fs1minus
        )
        if error:
            raise RuntimeError(
                "Special shear PBC creation failed for %s." %
                case_name_prefix
            )
        # Add a rigid body constraint to model_name_sh1 (Initial is recommended)
        a = mdb.models[model_name_sh1].rootAssembly
        _nodes = a.instances[instanceName.upper()].nodes
        region_rp, node_bd = _build_rigid_anchor_region(
            _nodes, rigid_node, rigid_node_tol
        )
        if 'BC-c7' in mdb.models[model_name_sh1].boundaryConditions:
            del mdb.models[model_name_sh1].boundaryConditions['BC-c7']

        # The nonhomogeneous cell also needs only a translational anchor.
        # Rotational DOFs remain governed by the 6DOF periodic equations.
        mdb.models[model_name_sh1].DisplacementBC(
            name='BC-c7', createStepName='Initial',
            region=region_rp, u1=0.0, u2=0.0, u3=0.0,
            ur1=UNSET, ur2=UNSET, ur3=UNSET
        )

        # Keep the preprocessing base INP immutable.  Earlier versions
        # overwrote BASE_INP_REALNOPBC with the PBC-applied model, so every
        # resume run imported and added another layer of boundary conditions.
        shear_pbc_job_name = BASE_JOB_REALNOPBC + '_SHEAR_PBC'
        shear_pbc_inp = _write_job_input_from_model(
            shear_pbc_job_name,
            model_name_sh1,
            work_place,
            numCpus
        )
        run_inp5 = case_name_prefix + '_SH2'
        run_inp5_path = choose_available_input_path(work_place, run_inp5)
        load_SH = shear1f1 + fs1minus
        blk5 = _make_cload_block_from_force(-load_SH, all_nodes_num, instanceName.upper(), dof_per_node = dof_node, eps_load=0.0)
        _inject_block_into_step(shear_pbc_inp, run_inp5_path, 'Step-1', blk5)
        log("-----Start submitting job : %s ..." % run_inp5, LOG_PATH, _t0)
        odb_path5 = _run_abaqus_input(
            work_place, run_inp5, run_inp5_path, numCpus, resume_solver=resume_solver
        )
        shear1u2 = _extract_displacement_vector_from_odb(
            odb_path5,
            instanceName.upper(),
            all_nodes_num,
            dof_per_node=dof_node
        )
    # (3) NOPBC model: inject the shear correction displacement field.
    run_inp6 = case_name_prefix + '_SH3'
    if reusable_stages['SH3']:
        odb_path6 = reusable_chain['stages']['SH3']['odb_path']
        print(
            "NIAH resume: reuse shear-chain stage %s_SH3." %
            case_name_prefix
        )
    else:
        run_inp6_path = choose_available_input_path(work_place, run_inp6)
        blk6 = _make_boundary_block_from_disp(shear1u2, all_nodes_num, instanceName, dof_per_node=dof_node)
        _inject_block_into_step(BASE_INP_NOPBC, run_inp6_path, 'Step-1', blk6)
        log("-----Start submitting job : %s ..." % run_inp6, LOG_PATH, _t0)
        odb_path6 = _run_abaqus_input(
            work_place, run_inp6, run_inp6_path, numCpus, resume_solver=resume_solver
        )
    shear1f2 = _extract_reaction_vector_from_odb(odb_path6, reaction_field_name,
                                                instanceName.upper(), all_nodes_num, dof_per_node=dof_node)
    if resume_solver:
        record_shear_chain_checkpoint(
            work_place,
            case_name_prefix,
            dependency_fingerprint,
            (odb_path1, odb_path5, odb_path6)
        )
        print(
            "NIAH resume: recorded complete shear chain %s." %
            case_name_prefix
        )
    return shear1f1, shear1u2, shear1f2


def _compute_E_Dbar_from_EH6(EH6):
    A = EH6[0:3, 0:3].copy()
    B = EH6[0:3, 3:6].copy()
    D = EH6[3:6, 3:6].copy()
    Ainv = np.linalg.inv(A)
    E = - np.dot(Ainv, B)
    Dbar = D - np.dot(B.T, np.dot(Ainv, B))
    return A, B, D, E, Dbar


def _build_lincurv_macro_disp(coords, mode, L=None, H=None, dof_node=None, E_couple=None):
    """
    Build the linear-curvature macro generalized displacement field.

    Parameters
    ----------
    coords : ndarray of shape (n, 3)
        Nodal coordinates.
    mode : str
        Supported modes:
            - 'LCXZ': linear-curvature field associated with x-z bending/shear
            - 'LCYZ': linear-curvature field associated with y-z bending/shear
    L : float, optional
        Characteristic length used in LCXZ mode.
    H : float, optional
        Characteristic length used in LCYZ mode.
    dof_node : int
        Degrees of freedom per node. Must be 3 or 6.
    E_couple : ndarray of shape (3, 3), optional
        Membrane-curvature coupling matrix, typically:
            E_couple = -inv(A) @ B
        If provided, in-plane mid-surface displacements are added.

    Returns
    -------
    v : ndarray of shape (n * dof_node,)
        Generalized displacement vector.
    """
    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("coords must have shape (n, 3).")

    if dof_node not in (3, 6):
        raise ValueError("dof_node must be either 3 or 6.")

    m = mode.upper()
    if m not in ("LCXZ", "LCYZ"):
        raise ValueError("Unknown linear-curvature mode: %s" % mode)

    if m == "LCXZ":
        if L is None or np.isclose(L, 0.0):
            raise ValueError("L must be provided and nonzero for LCXZ mode.")
    elif m == "LCYZ":
        if H is None or np.isclose(H, 0.0):
            raise ValueError("H must be provided and nonzero for LCYZ mode.")

    if E_couple is not None:
        E_couple = np.asarray(E_couple, dtype=float)
        if E_couple.shape != (3, 3):
            raise ValueError("E_couple must have shape (3, 3).")

    # center the RVE
    xr = coords[:, 0] - 0.5 * (coords[:, 0].max() + coords[:, 0].min())
    yr = coords[:, 1] - 0.5 * (coords[:, 1].max() + coords[:, 1].min())
    zr = coords[:, 2] - 0.5 * (coords[:, 2].max() + coords[:, 2].min())
    n = coords.shape[0]
    v = np.zeros(n * dof_node, dtype=float)
    if dof_node == 6:
        u0 = np.zeros(n, dtype=float)
        v0 = np.zeros(n, dtype=float)
        if m == "LCXZ":
            # w0 = -x^3 / (6L), dw/dx = -x^2 / (2L)
            # adopt theta_y = -dw/dx = x^2 / (2L)
            w0 = -(xr ** 3) / (6.0 * L)
            thy = (xr ** 2) / (2.0 * L)
            if E_couple is not None:
                e11 = float(E_couple[0, 0])
                e21 = float(E_couple[1, 0])
                e31 = float(E_couple[2, 0])
                u0 = (0.5 * e11 * xr**2 - 0.5 * e21 * yr**2) / L
                v0 = (e21 * xr * yr + 0.5 * e31 * xr**2) / L

            U1 = u0 + zr * thy
            U2 = v0
            U3 = w0
            UR1 = np.zeros(n, dtype=float)
            UR2 = thy
            UR3 = np.zeros(n, dtype=float)
        else:  # LCYZ
            # w0 = -y^3 / (6H), dw/dy = -y^2 / (2H)
            # adopt theta_x = dw/dy = -y^2 / (2H)
            w0 = -(yr ** 3) / (6.0 * H)
            thx = -(yr ** 2) / (2.0 * H)
            if E_couple is not None:
                e12 = float(E_couple[0, 1])
                e22 = float(E_couple[1, 1])
                e32 = float(E_couple[2, 1])
                u0 = (e12 * xr * yr + 0.5 * e32 * yr**2) / H
                v0 = (0.5 * e22 * yr**2 - 0.5 * e12 * xr**2) / H

            U1 = u0
            U2 = v0 - zr * thx
            U3 = w0
            UR1 = thx
            UR2 = np.zeros(n, dtype=float)
            UR3 = np.zeros(n, dtype=float)

        v[0::6] = U1
        v[1::6] = U2
        v[2::6] = U3
        v[3::6] = UR1
        v[4::6] = UR2
        v[5::6] = UR3
    else:  # dof_node == 3
        u = np.zeros(n, dtype=float)
        vv = np.zeros(n, dtype=float)
        w = np.zeros(n, dtype=float)
        if m == "LCXZ":
            w = -(xr ** 3) / (6.0 * L)
            u_b = zr * (xr ** 2) / (2.0 * L)
            u0 = np.zeros(n, dtype=float)
            v0 = np.zeros(n, dtype=float)
            if E_couple is not None:
                e11 = float(E_couple[0, 0])
                e21 = float(E_couple[1, 0])
                e31 = float(E_couple[2, 0])
                u0 = (0.5 * e11 * xr**2 - 0.5 * e21 * yr**2) / L
                v0 = (e21 * xr * yr + 0.5 * e31 * xr**2) / L

            u = u0 + u_b
            vv = v0
        else:  # LCYZ
            w = -(yr ** 3) / (6.0 * H)
            v_b = zr * (yr ** 2) / (2.0 * H)
            u0 = np.zeros(n, dtype=float)
            v0 = np.zeros(n, dtype=float)
            if E_couple is not None:
                e12 = float(E_couple[0, 1])
                e22 = float(E_couple[1, 1])
                e32 = float(E_couple[2, 1])
                u0 = (e12 * xr * yr + 0.5 * e32 * yr**2) / H
                v0 = (0.5 * e22 * yr**2 - 0.5 * e12 * xr**2) / H

            u = u0
            vv = v0 + v_b

        v[0::3] = u
        v[1::3] = vv
        v[2::3] = w

    return v


def _dot_area(f, u, cell_S):
    return float(np.dot(f, u)) / cell_S


def _solve_temp(A_sub, rig):
    return np.linalg.solve(A_sub, rig)


def _compute_su_sf(shear_u_tot, shear_f_tot,
                   u_mode_a_tot, f_mode_a_tot,
                   u_mode_b_tot, f_mode_b_tot,
                   temp):
    """
      su = shear_tot + temp1*mode_a_tot + temp2*mode_b_tot
      sf = shear_tot + temp1*mode_a_tot + temp2*mode_b_tot
    """
    t1, t2 = float(temp[0, 0]), float(temp[1, 0])
    su = shear_u_tot + t1 * u_mode_a_tot + t2 * u_mode_b_tot
    sf = shear_f_tot + t1 * f_mode_a_tot + t2 * f_mode_b_tot
    return su, sf


def _compute_K_from_energy(su1, sf1, su2, sf2, Dbar, L, H):
    """
      rig = [ l/b*su1'*sf1 - Dbar11/12*l^2;
              b/l*su2'*sf2 - Dbar22/12*b^2 ]
      temp = inv([[Dbar11^2, Dbar13^2],[Dbar23^2, Dbar22^2]]) * rig
      K = 1./temp
    """
    rig = np.zeros((2, 1), dtype=float)
    rig[0, 0] = (L / H) * float(np.dot(su1, sf1)) - (Dbar[0, 0] / 12.0) * (L * L)
    rig[1, 0] = (H / L) * float(np.dot(su2, sf2)) - (Dbar[1, 1] / 12.0) * (H * H)
    M = np.array([
        [Dbar[0, 0] ** 2, Dbar[0, 2] ** 2],
        [Dbar[1, 2] ** 2, Dbar[1, 1] ** 2]
    ], dtype=float)
    temp = np.linalg.solve(M, rig)   # 2x1
    K = 1.0 / temp                   # 2x1
    return K, rig, temp
