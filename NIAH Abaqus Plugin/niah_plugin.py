# -*- coding: UTF-8 -*-
from abaqusGui import getAFXApp, Activator, AFXMode, afxCreatePNGIcon
from abaqusConstants import ALL
import os

thisPath = os.path.abspath(__file__)
thisDir = os.path.dirname(thisPath)

toolset = getAFXApp().getAFXMainWindow().getPluginToolset()
toolset.registerGuiMenuButton(
    buttonText='NIAH Homogenization',
    object=Activator(os.path.join(thisDir, 'niahDB.py')),
    kernelInitString='import niah_kernel',
    messageId=AFXMode.ID_ACTIVATE,
    icon=afxCreatePNGIcon(os.path.join(thisDir,'icons', 'pluginsmo_clean.png')),
    applicableModules=ALL,
    version='0.1',
    author='Zhihui Liu',
    description='NIAH Abaqus plugin starter',
    helpUrl='N/A'
)
