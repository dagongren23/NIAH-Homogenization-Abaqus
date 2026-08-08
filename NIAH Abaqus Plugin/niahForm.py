# -*- coding: UTF-8 -*-
"""
Optional GUI Toolkit form version.
This file is not required when you use the RSG dialog route in niahDB.py,
but it is included because you asked for a full starter framework.
If you prefer pure AFXForm/AFXDataDialog later, you can extend from this file.
"""

from abaqusGui import AFXForm, AFXGuiCommand
from abaqusGui import AFXStringKeyword, AFXIntKeyword, AFXFloatKeyword, AFXBoolKeyword


class NIAHForm(AFXForm):
    def __init__(self, owner):
        AFXForm.__init__(self, owner)
        self.cmd = AFXGuiCommand(
            mode=self,
            method='run_niah_plugin',
            objectName='niah_kernel',
            registerQuery=False,
        )

        self.partnameKw = AFXStringKeyword(self.cmd, 'partname', True, 'honeycomb_plane_shell_homo')
        self.struTypeKw = AFXStringKeyword(self.cmd, 'stru_type', True, 'shell')
        self.elementTypeKw = AFXStringKeyword(self.cmd, 'element_type', True, 'S4R')
        self.modelNameKw = AFXStringKeyword(self.cmd, 'model_name', True, 'Model-1')
        self.numCpusKw = AFXIntKeyword(self.cmd, 'numCpus', True, 8)
        self.meshsensKw = AFXFloatKeyword(self.cmd, 'meshsens', True, 1.0e-3)
        self.materialEKw = AFXFloatKeyword(self.cmd, 'material_E', True, 1.0)
        self.materialVKw = AFXFloatKeyword(self.cmd, 'material_v', True, 0.3)
        self.shellThicknessKw = AFXFloatKeyword(self.cmd, 'shell_thickness', True, 0.01)
        self.beamRadiusKw = AFXFloatKeyword(self.cmd, 'beam_radius', True, 0.01)
        self.beamN1xKw = AFXFloatKeyword(self.cmd, 'beam_n1x', True, 0.1)
        self.beamN1yKw = AFXFloatKeyword(self.cmd, 'beam_n1y', True, 0.0)
        self.beamN1zKw = AFXFloatKeyword(self.cmd, 'beam_n1z', True, -1.0)
        self.periodicityKw = AFXIntKeyword(self.cmd, 'periodicity_ch', True, 3)
        self.rigidNodeXKw = AFXFloatKeyword(self.cmd, 'rigid_nodex', True, 2.0)
        self.rigidNodeYKw = AFXFloatKeyword(self.cmd, 'rigid_nodey', True, 0.0)
        self.rigidNodeZKw = AFXFloatKeyword(self.cmd, 'rigid_nodez', True, 0.5)
        self.rigidTolKw = AFXFloatKeyword(self.cmd, 'rigid_node_tol', True, 0.05)
        self.runPreKw = AFXBoolKeyword(self.cmd, 'run_pre', AFXBoolKeyword.TRUE_FALSE, True)
        self.runSolverKw = AFXBoolKeyword(self.cmd, 'run_solver', AFXBoolKeyword.TRUE_FALSE, True)
        self.resumeSolverKw = AFXBoolKeyword(self.cmd, 'resume_solver', AFXBoolKeyword.TRUE_FALSE, True)
        self.runVisKw = AFXBoolKeyword(self.cmd, 'run_vis', AFXBoolKeyword.TRUE_FALSE, True)
