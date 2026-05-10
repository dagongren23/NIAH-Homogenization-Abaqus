# -*- coding: utf-8 -*-
from rsg.rsgGui import *
from abaqusConstants import INTEGER, FLOAT

dialogBox = RsgDialog(
    title='NIAH Homogenization',
    kernelModule='niah_kernel',
    kernelFunction='run_niah_plugin',
    includeApplyBtn=False,
    includeSeparator=True,
    okBtnText='OK',
    applyBtnText='Apply',
    execDir=thisDir
)

# =========================
# Main container
# =========================
RsgHorizontalFrame(name='HFrame_main', p='DialogBox', layout='0')

# =========================================================
# LEFT COLUMN
# =========================================================
RsgVerticalFrame(name='VFrame_left', p='HFrame_main', layout='0')

# ----- Group 1: Model / analysis -----
RsgGroupBox(name='GB_basic', p='VFrame_left', text='Model and analysis settings', layout='0')

RsgGroupBox(name='GB_basic_filepath', p='GB_basic', text='File path', layout='0')
# --- custom data output directory ---
RsgTextField(p='GB_basic_filepath', fieldType='String', ncols=30,
             labelText='Data Output Dir (Optional)',
             keyword='data_dir', default='')

RsgLabel(p='GB_basic_filepath', text='Leave blank to use default workbench.', useBoldFont=False)
RsgLabel(p='GB_basic_filepath', text='Will be used for .txt, .log, and .inp files.', useBoldFont=False)

RsgGroupBox(name='GB_basic_input', p='GB_basic', text='Model input', layout='0')
RsgTextField(p='GB_basic_input', fieldType='String', ncols=24,
             labelText='Part name',
             keyword='partname', default='honeycomb_plane_shell_homo')

RsgLabel(p='GB_basic_input', text='Part name examples:', useBoldFont=True)
RsgLabel(p='GB_basic_input', text='3D part name: cube_mesh, beam_octet, cubic_center_8ele', useBoldFont=False)
RsgLabel(p='GB_basic_input', text='Shell part name: pingban4, truss_sandwich_20n, honeycomb_plane_shell_homo', useBoldFont=False)

RsgGroupBox(name='GB_basic_type', p='GB_basic', text='Analysis type', layout='0')
RsgTextField(p='GB_basic_type', fieldType='String', ncols=12,
             labelText='Structure type',
             keyword='stru_type', default='shell')

RsgLabel(p='GB_basic_type', text="Structure type allowed values: '3D' or 'shell'", useBoldFont=False)

RsgTextField(p='GB_basic_type', fieldType='String', ncols=16,
             labelText='Element type',
             keyword='element_type', default='S4R')

RsgLabel(p='GB_basic_type', text='Element type examples: S4R, B31, C3D8R, C3D4', useBoldFont=False)
RsgLabel(p='GB_basic_type', text='The type must match the actual Abaqus mesh.', useBoldFont=False)

RsgGroupBox(name='GB_basic_other', p='GB_basic', text='Other settings', layout='0')

RsgTextField(p='GB_basic_other', fieldType='String', ncols=16,
             labelText='Model name',
             keyword='model_name', default='Model-1')

RsgTextField(p='GB_basic_other', fieldType='Integer', ncols=8,
             labelText='Number of CPUs',
             keyword='numCpus', default='8')

RsgTextField(p='GB_basic_other', fieldType='Float', ncols=10,
             labelText='Mesh sensitivity',
             keyword='meshsens', default='1.0E-3')

RsgLabel(p='GB_basic_other',
         text='Mesh sensitivity used for opposite-boundary node matching.',
         useBoldFont=False)

# =========================================================
# RIGHT COLUMN
# =========================================================
RsgVerticalFrame(name='VFrame_right', p='HFrame_main', layout='0')

# ----- Group 2: Material / section -----
RsgGroupBox(name='GB_mat', p='VFrame_right', text='Material and section parameters', layout='0')

RsgTextField(p='GB_mat', fieldType='Float', ncols=10,
             labelText='Elastic modulus E (MPa)',
             keyword='material_E', default='1.0')

RsgTextField(p='GB_mat', fieldType='Float', ncols=10,
             labelText='Poisson ratio ν',
             keyword='material_v', default='0.3')

RsgTextField(p='GB_mat', fieldType='Float', ncols=10,
             labelText='Shell thickness (mm)',
             keyword='shell_thickness', default='0.01')

RsgTextField(p='GB_mat', fieldType='Float', ncols=10,
             labelText='Beam inner radius (mm)',
             keyword='beam_radius', default='0.01')

RsgLabel(p='GB_mat',
         text='For shell RVEs, shell thickness should match the section definition.',
         useBoldFont=False)
RsgLabel(p='GB_mat',
         text='For beam RVEs, beam radius should match the geometric model.',
         useBoldFont=False)

# ----- Group 3: Beam orientation -----
RsgGroupBox(name='GB_n1', p='VFrame_right', text='Beam local n1 direction', layout='0')

RsgTextField(p='GB_n1', fieldType='Float', ncols=10,
             labelText='n1-x', keyword='beam_n1x', default='0.1')
RsgTextField(p='GB_n1', fieldType='Float', ncols=10,
             labelText='n1-y', keyword='beam_n1y', default='0.0')
RsgTextField(p='GB_n1', fieldType='Float', ncols=10,
             labelText='n1-z', keyword='beam_n1z', default='-1.0')

RsgLabel(p='GB_n1',
         text='Used mainly for beam/shell local orientation definition.',
         useBoldFont=False)

# ----- Group 4: Periodicity / rigid node -----
RsgGroupBox(name='GB_period', p='VFrame_right', text='Periodicity and rigid-node settings', layout='0')

RsgTextField(p='GB_period', fieldType='Integer', ncols=8,
             labelText='Non-periodic direction',
             keyword='periodicity_ch', default='3')

RsgLabel(p='GB_period', text='1 = x non-periodic', useBoldFont=False)
RsgLabel(p='GB_period', text='2 = y non-periodic', useBoldFont=False)
RsgLabel(p='GB_period', text='3 = z non-periodic', useBoldFont=False)

RsgTextField(p='GB_period', fieldType='Float', ncols=10,
             labelText='Rigid node x (mm)', keyword='rigid_nodex', default='2.0')
RsgTextField(p='GB_period', fieldType='Float', ncols=10,
             labelText='Rigid node y (mm)', keyword='rigid_nodey', default='0.0')
RsgTextField(p='GB_period', fieldType='Float', ncols=10,
             labelText='Rigid node z (mm)', keyword='rigid_nodez', default='0.5')

RsgTextField(p='GB_period', fieldType='Float', ncols=10,
             labelText='Rigid-node tolerance (mm)',
             keyword='rigid_node_tol', default='0.05')

RsgLabel(p='GB_period',
         text='Use a correct internal constraint point, usually near the RVE center.',
         useBoldFont=False)

# =========================================================
# BOTTOM AREA
# =========================================================
RsgGroupBox(name='GB_workflow', p='DialogBox', text='Workflow options', layout='0')

RsgCheckButton(p='GB_workflow', text='Run Pre-processing',
               keyword='run_pre', default=True)

# --- Preprocessing mode selection ---
RsgTextField(p='GB_workflow', fieldType='String', ncols=12,
             labelText='Pre-process Mode',
             keyword='pre_mode', default='Part')
RsgLabel(p='GB_workflow',
         text="Input 'Part' (auto material) or 'Model' (pre-defined material).",
         useBoldFont=False)

RsgCheckButton(p='GB_workflow', text='Run Homogenization solver',
               keyword='run_solver', default=True)
RsgCheckButton(p='GB_workflow', text='Run Stiffness visualization',
               keyword='run_vis', default=True)

RsgLabel(p='GB_workflow',
         text='Run only the selected stages of the NIAH workflow.',
         useBoldFont=False)

RsgGroupBox(name='GB_notes', p='DialogBox', text='Notes', layout='0')

RsgLabel(p='GB_notes',
         text='Keep the entire niah_core package under the same plug-in directory.',
         useBoldFont=False)
RsgLabel(p='GB_notes',
         text='The plug-in workbench and text outputs will be created automatically.',
         useBoldFont=False)
RsgLabel(p='GB_notes',
         text='Element type, structure type, and model geometry must be consistent.',
         useBoldFont=False)

dialogBox.show()