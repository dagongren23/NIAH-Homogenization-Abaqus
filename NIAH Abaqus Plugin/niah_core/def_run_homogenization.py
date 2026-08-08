# -*-coding:UTF-8-*-
"""
NIAH-ABAQUS: 2. Homogenization solver
Author: Zhihui Liu
Date:   2026/04/07
Email:  dutme_lzh@163.com
"""
from abaqus import *

def run_homogenization(runtime_input):
    from driverUtils import executeOnCaeStartup
    from datetime import datetime
    import numpy as np
    import os
    executeOnCaeStartup()
    Mdb()

    # Input
    # (partname, stru_type, element_type, condition_ch, dof_per_node, modelName, instanceName, numCpus, meshsens,
    #  material_name, material_E, material_v, shell_thickness, beam_radius, beam_n1, periodicity_ch,
    #  rigid_node, rigid_node_tol, work_place, outtxt, suffix, LOG_PATH, _t0,
    #  model_name_NOPBC, model_name_PBC, model_name_NOPBCX, model_name_NOPBCY,
    #  BASE_JOB_REALNOPBCX, BASE_JOB_REALNOPBCY, BASE_JOB_NOPBC, BASE_JOB_PBC) = runtime_input
    partname = runtime_input['partname']
    stru_type = runtime_input['stru_type']
    element_type = runtime_input['element_type']
    condition_ch = runtime_input['condition_ch']
    dof_per_node = runtime_input['dof_per_node']
    modelName = runtime_input['model_name']
    instanceName = runtime_input['instance_name']
    numCpus = runtime_input['numCpus']
    meshsens = runtime_input['meshsens']
    material_name = runtime_input['material_name']
    material_E = runtime_input['material_E']
    material_v = runtime_input['material_v']
    shell_thickness = runtime_input['shell_thickness']
    beam_radius = runtime_input['beam_radius']
    beam_n1 = runtime_input['beam_n1']
    periodicity_ch = runtime_input['periodicity_ch']
    rigid_node = runtime_input['rigid_node']
    rigid_node_tol = runtime_input['rigid_node_tol']
    work_place = runtime_input['workbench_path']
    data_out_path = runtime_input['data_out_path']
    outtxt = runtime_input['outtxt']
    suffix = runtime_input['suffix']
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
    resume_solver = bool(runtime_input.get('resume_solver', False))
    os.chdir(work_place)
    reaction_field_name = 'RF'
    BASE_INP_REALNOPBCX = os.path.join(work_place, BASE_JOB_REALNOPBCX + '.inp')
    BASE_INP_REALNOPBCY = os.path.join(work_place, BASE_JOB_REALNOPBCY + '.inp')
    BASE_INP_NOPBC = os.path.join(work_place, BASE_JOB_NOPBC + '.inp')
    BASE_INP_PBC = os.path.join(work_place, BASE_JOB_PBC + '.inp')

    os.chdir(work_place)
    from Utility_function import (log, _get_nodes_coordinates, _niah_chain_one_case_inp, _build_3D_X1,
                                  _build_v0_vstar_vectors, _niah_chain_one_case_inp_additional,
                                  _compute_E_Dbar_from_EH6, _niah_share_one_case_inp, _build_lincurv_macro_disp,
                                  _dot_area, _solve_temp, _compute_su_sf, _compute_K_from_energy)
    from homogenization_math import assemble_effective_stiffness_from_chain
    from job_resume import abaqus_api_path

    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_PATH, "a") as f:
        f.write("-" * 50 + "\n")
        f.write("Continue the NIAH job - {}\n".format(current_time))
        f.flush()

    log("NIAH-ABAQUS: Homogenization solver", LOG_PATH, _t0)

    # import inp
    inp_name = '{}.inp'.format(BASE_JOB_NOPBC)
    mdb.ModelFromInputFile(
        name=BASE_JOB_NOPBC,
        inputFileName=abaqus_api_path(inp_name)
    )
    del mdb.models[modelName]
    mdb.models.changeKey(fromName=BASE_JOB_NOPBC, toName=modelName)

    # assemble
    a = mdb.models[modelName].rootAssembly
    a.regenerate()
    parts_name = mdb.models[modelName].parts.keys()

    mdb.models[modelName].rootAssembly.features.changeKey(fromName=parts_name[0] + '-1', toName=instanceName)
    a.regenerate()
    mdb.models[modelName].parts.changeKey(fromName=parts_name[0], toName=partname)
    p = mdb.models[modelName].parts[partname]
    # for setName in list(a.sets.keys()):
    #     del a.sets[setName]

    for consName in list(mdb.models[modelName].constraints.keys()):
        del mdb.models[modelName].constraints[consName]

    for featName in list(a.features.keys()):
        if featName.startswith('RP-'):
            del a.features[featName]

    p.renumberNode(startLabel=1, increment=1)
    p.renumberElement(startLabel=1, increment=1)

    a = mdb.models[modelName].rootAssembly
    _nodes, nodes_coords = _get_nodes_coordinates(a, instanceName)
    all_nodes_num = len(_nodes)
    len_botbc = len(a.sets['botbc'.upper()].nodes)
    len_backbc = len(a.sets['backbc'.upper()].nodes)
    log("Nodes loaded: %d" % all_nodes_num, LOG_PATH, _t0)
    log("botbc nodes loaded: %d" % len_botbc, LOG_PATH, _t0)
    log("backbc nodes loaded: %d" % len_backbc, LOG_PATH, _t0)

    x = nodes_coords[:, 0].copy()
    y = nodes_coords[:, 1].copy()
    z = nodes_coords[:, 2].copy()
    L = abs(max(x) - min(x))
    H = abs(max(y) - min(y))
    W = abs(max(z) - min(z))
    if L is None or W is None or H is None:
        raise RuntimeError("L/H/W has not been assigned correctly.")

    log("Unit envelope size: L:{} H:{} W:{}".format(round(L), round(H), round(W)), LOG_PATH, _t0)

    X1 = np.zeros((6, all_nodes_num * dof_per_node))
    X2 = np.zeros((6, all_nodes_num * dof_per_node))
    FX1 = np.zeros((6, all_nodes_num * dof_per_node))
    FX2 = np.zeros((6, all_nodes_num * dof_per_node))
    EH = np.zeros((6, 6))

    if stru_type == '3D':
        case_names = ["XX", "YY", "ZZ", "YZ", "XZ", "XY"]
        X1 = _build_3D_X1(nodes_coords, dof_per_node=dof_per_node)
        for I in range(6):
            prefix = "NIAH_SOLID_CASE_%02d_%s" % (I + 1, case_names[I])
            log("Start submitting job : %s ..." % prefix, LOG_PATH, _t0)
            v_vec = X1[I, :]
            fx1_row, x2_row, fx2_row, _x2_odb_path = _niah_chain_one_case_inp(
                prefix, v_vec, work_place, all_nodes_num, instanceName, numCpus, reaction_field_name,
                BASE_INP_PBC, BASE_INP_NOPBC, LOG_PATH, _t0, dof_node=dof_per_node,
                resume_solver=resume_solver
            )
            FX1[I, :] = fx1_row
            X2[I, :] = x2_row
            FX2[I, :] = fx2_row
            log("Job %s completed" % prefix, LOG_PATH, _t0)

        if np.all(X1 + X2 == 0):
            log("X1 + X2 is all zeros. This might indicate an issue in the calculation of X1 or X2.", LOG_PATH, _t0)
            raise RuntimeError("calculation of X1 or X2 failed.")

        cell_V = L * H * W
        # Recover every component of the effective stiffness matrix.
        # X2 is the signed correction generated by applying -FX1.
        # Include every translational and rotational DOF in the energy product.
        EH = assemble_effective_stiffness_from_chain(X1, X2, FX1, FX2, cell_V)

        with open(outtxt, 'w') as f:
            # EH - 6x6
            f.write("=" * 50 + "\n")
            f.write("EH (6x6 stiffness matrix):\n")
            np.savetxt(f, EH, fmt='%.12f', delimiter='  ')  # '%20.12e'
            f.write("\n")
    elif stru_type == 'shell':
        kl_list = ['11', '22', '12']
        kl_to_row_mem = {'11': 0, '22': 1, '12': 2}
        kl_to_row_ben = {'11': 3, '22': 4, '12': 5}
        uxminus = np.zeros((5, len_backbc * dof_per_node))
        uyminus = np.zeros((5, len_botbc * dof_per_node))
        fxminus = np.zeros((5, all_nodes_num * dof_per_node))
        fyminus = np.zeros((5, all_nodes_num * dof_per_node))
        for kl in kl_list:
            v0_vec, vstar_vec = _build_v0_vstar_vectors(nodes_coords, kl, dof_per_node=dof_per_node)
            X1[kl_to_row_mem[kl], :] = v0_vec
            X1[kl_to_row_ben[kl], :] = vstar_vec
            prefix_mem = 'NIAH_PLATE_%s_MEM' % kl
            log("Start submitting job : %s ..." % prefix_mem, LOG_PATH, _t0)
            fx1_row, x2_row, fx2_row, mem_x2_odb_path = _niah_chain_one_case_inp(prefix_mem, v0_vec, work_place, all_nodes_num,
                                                                instanceName,
                                                                numCpus, reaction_field_name, BASE_INP_PBC,
                                                                BASE_INP_NOPBC,
                                                                LOG_PATH, _t0, dof_node=dof_per_node,
                                                                resume_solver=resume_solver)
            f1xminus, f1yminus, u1xminus, u1yminus = _niah_chain_one_case_inp_additional(prefix_mem, v0_vec, work_place,
                                                                                         all_nodes_num, instanceName,
                                                                                         numCpus,
                                                                                         reaction_field_name,
                                                                                         BASE_INP_PBC,
                                                                                         BASE_INP_NOPBC, LOG_PATH, _t0,
                                                                                         dof_node=dof_per_node,
                                                                                         resume_solver=resume_solver,
                                                                                         x2_odb_path=mem_x2_odb_path)
            FX1[kl_to_row_mem[kl], :] = fx1_row
            X2[kl_to_row_mem[kl], :] = x2_row
            FX2[kl_to_row_mem[kl], :] = fx2_row
            prefix_ben = 'NIAH_PLATE_%s_BEN' % kl
            log("Start submitting job : %s ..." % prefix_ben, LOG_PATH, _t0)
            fx1_row, x2_row, fx2_row, ben_x2_odb_path = _niah_chain_one_case_inp(prefix_ben, vstar_vec, work_place, all_nodes_num,
                                                                instanceName,
                                                                numCpus, reaction_field_name, BASE_INP_PBC,
                                                                BASE_INP_NOPBC,
                                                                LOG_PATH, _t0, dof_node=dof_per_node,
                                                                resume_solver=resume_solver)
            if kl_to_row_ben[kl] < 5:
                f2xminus, f2yminus, u2xminus, u2yminus = _niah_chain_one_case_inp_additional(prefix_ben, vstar_vec,
                                                                                             work_place,
                                                                                             all_nodes_num,
                                                                                             instanceName,
                                                                                             numCpus,
                                                                                             reaction_field_name,
                                                                                             BASE_INP_PBC,
                                                                                             BASE_INP_NOPBC, LOG_PATH,
                                                                                             _t0, dof_node=dof_per_node,
                                                                                             resume_solver=resume_solver,
                                                                                             x2_odb_path=ben_x2_odb_path)
                uxminus[kl_to_row_ben[kl], :] = u2xminus
                uxminus[kl_to_row_mem[kl], :] = u1xminus
                uyminus[kl_to_row_ben[kl], :] = u2yminus
                uyminus[kl_to_row_mem[kl], :] = u1yminus
                fxminus[kl_to_row_ben[kl], :] = f2xminus
                fxminus[kl_to_row_mem[kl], :] = f1xminus
                fyminus[kl_to_row_ben[kl], :] = f2yminus
                fyminus[kl_to_row_mem[kl], :] = f1yminus

            FX1[kl_to_row_ben[kl], :] = fx1_row
            X2[kl_to_row_ben[kl], :] = x2_row
            FX2[kl_to_row_ben[kl], :] = fx2_row
            log("Job %s completed" % kl, LOG_PATH, _t0)

        cell_S = L * H
        # -------------------- Assemble EH (6x6) --------------------
        for I in range(6):
            for J in range(6):
                EH[I, J] = np.sum((X1[I, :] + X2[I, :]) * (FX1[J, :] + FX2[J, :])) / cell_S

        EH = (EH + EH.T) / 2
        A, B, D, E_couple, Dbar = _compute_E_Dbar_from_EH6(EH)
        us1minus = E_couple[0, 0] * uxminus[0, :] + E_couple[1, 0] * uxminus[1, :] + E_couple[2, 0] * uxminus[2, :] + \
                   uxminus[3, :]
        fs1minus = E_couple[0, 0] * fxminus[0, :] + E_couple[1, 0] * fxminus[1, :] + E_couple[2, 0] * fxminus[2, :] + \
                   fxminus[3, :]

        us2minus = E_couple[0, 1] * uyminus[0, :] + E_couple[1, 1] * uyminus[1, :] + E_couple[2, 1] * uyminus[2, :] + \
                   uyminus[4, :]
        fs2minus = E_couple[0, 1] * fyminus[0, :] + E_couple[1, 1] * fyminus[1, :] + E_couple[2, 1] * fyminus[2, :] + \
                   fyminus[4, :]

        prefix_LCX = 'NIAH_PLATE_LCXZ' + 'sv'
        disp_lcx = _build_lincurv_macro_disp(nodes_coords, 'LCXZ', L, H, dof_node=dof_per_node, E_couple=E_couple)
        shear1f1, shear1u2, shear1f2 = _niah_share_one_case_inp(prefix_LCX, disp_lcx, 'SHXZ', work_place, all_nodes_num,
                                                                instanceName, numCpus, reaction_field_name,
                                                                BASE_INP_NOPBC, BASE_JOB_REALNOPBCX,
                                                                BASE_INP_REALNOPBCX,
                                                                meshsens, LOG_PATH, _t0, rigid_node, rigid_node_tol,
                                                                us1minus=us1minus, fs1minus=fs1minus,
                                                                dof_node=dof_per_node,
                                                                resume_solver=resume_solver)
        shear1u1 = disp_lcx
        prefix_LCY = 'NIAH_PLATE_LCYZ' + 'sv'
        disp_lcy = _build_lincurv_macro_disp(nodes_coords, 'LCYZ', L, H, dof_node=dof_per_node, E_couple=E_couple)
        shear2f1, shear2u2, shear2f2 = _niah_share_one_case_inp(prefix_LCY, disp_lcy, 'SHYZ', work_place, all_nodes_num,
                                                                instanceName, numCpus, reaction_field_name,
                                                                BASE_INP_NOPBC, BASE_JOB_REALNOPBCY,
                                                                BASE_INP_REALNOPBCY,
                                                                meshsens, LOG_PATH, _t0, rigid_node, rigid_node_tol,
                                                                us1minus=us2minus, fs1minus=fs2minus,
                                                                dof_node=dof_per_node,
                                                                resume_solver=resume_solver)
        shear2u1 = disp_lcy
        U_tot = X1 + X2  # shape: (6, ndof)
        F_tot = FX1 + FX2  # shape: (6, ndof)
        U_mem1 = U_tot[0, :]  # Mode 1: unit x-normal membrane strain.
        U_mem2 = U_tot[1, :]  # Mode 2: unit y-normal membrane strain.
        U_mem3 = U_tot[2, :]  # Mode 3: unit in-plane engineering shear strain.
        F_mem1 = F_tot[0, :]
        F_mem2 = F_tot[1, :]
        F_mem3 = F_tot[2, :]
        shear1_u_tot = shear1u1 + shear1u2
        shear1_f_tot = shear1f1 + shear1f2
        rig1 = np.zeros((2, 1), dtype=float)
        rig1[0, 0] = -_dot_area(shear1_f_tot, U_mem1, cell_S)
        rig1[1, 0] = -_dot_area(shear1_f_tot, U_mem3, cell_S)
        A_sub_xz = np.array([[A[0, 0], A[0, 2]],
                             [A[0, 2], A[2, 2]]], dtype=float)
        temp1 = _solve_temp(A_sub_xz, rig1)
        su1, sf1 = _compute_su_sf(
            shear_u_tot=shear1_u_tot,
            shear_f_tot=shear1_f_tot,
            u_mode_a_tot=U_mem1, f_mode_a_tot=F_mem1,
            u_mode_b_tot=U_mem3, f_mode_b_tot=F_mem3,
            temp=temp1
        )
        shear2_u_tot = shear2u1 + shear2u2
        shear2_f_tot = shear2f1 + shear2f2
        rig2 = np.zeros((2, 1), dtype=float)
        rig2[0, 0] = -_dot_area(shear2_f_tot, U_mem2, cell_S)
        rig2[1, 0] = -_dot_area(shear2_f_tot, U_mem3, cell_S)
        A_sub_yz = np.array([[A[1, 1], A[1, 2]],
                             [A[1, 2], A[2, 2]]], dtype=float)
        temp2 = _solve_temp(A_sub_yz, rig2)
        su2, sf2 = _compute_su_sf(
            shear_u_tot=shear2_u_tot,
            shear_f_tot=shear2_f_tot,
            u_mode_a_tot=U_mem2, f_mode_a_tot=F_mem2,
            u_mode_b_tot=U_mem3, f_mode_b_tot=F_mem3,
            temp=temp2
        )
        K, rig_energy, temp_energy = _compute_K_from_energy(su1, sf1, su2, sf2, Dbar, L, H)
        with open(outtxt, 'w') as f:
            # EH - 6x6
            f.write("=" * 50 + "\n")
            f.write("EH (6x6 stiffness matrix):\n")
            np.savetxt(f, EH, fmt='%20.12e', delimiter='  ')  # '%.12f'
            f.write("\n")

            # A - 3x3
            f.write("=" * 50 + "\n")
            f.write("A (3x3 extensional stiffness):\n")
            np.savetxt(f, A, fmt='%20.12e', delimiter='  ')  # '%.12f'
            f.write("\n")

            # B - 3x3
            f.write("=" * 50 + "\n")
            f.write("B (3x3 coupling stiffness):\n")
            np.savetxt(f, B, fmt='%20.12e', delimiter='  ')  # '%.12f'
            f.write("\n")

            # D - 3x3
            f.write("=" * 50 + "\n")
            f.write("D (3x3 bending stiffness):\n")
            np.savetxt(f, D, fmt='%20.12e', delimiter='  ')  # '%.12f'
            f.write("\n")

            # K
            Kvec = np.ravel(K)  # 1-D
            f.write("=" * 50 + "\n")
            f.write("K (shear stiffness):\n")
            f.write("Kxz = %20.12e\n" % float(Kvec[0]))  # '%.12f'
            f.write("Kyz = %20.12e\n" % float(Kvec[1]))  # '%.12f'
    else:
        raise RuntimeError("stru_type has not been assigned correctly.")

    log("NIAH-ABAQUS has been done!", LOG_PATH, _t0)
    pass
