# Third-party notices

## Abaqus

The NIAH plug-in calls the Abaqus/CAE scripting API. Abaqus is proprietary software owned by Dassault Systèmes and is not included in this repository. Users must obtain their own Abaqus installation and license.

## EasyPBC literature attribution

The project acknowledges the following paper as related implementation literature:

Omairey, Dunning, and Sriramula, “Development of an Abaqus plugin tool for periodic RVE homogenisation,” *Engineering with Computers* 35 (2019) 567–577. https://doi.org/10.1007/s00366-018-0616-4

No EasyPBC source code, bytecode, icon, or distributed plug-in asset is included in this repository. The active NIAH periodic-boundary implementation was independently rewritten from PBC mathematics and project requirements. The retained attribution identifies related prior software and literature; it does not incorporate or relicense EasyPBC source.

## Standalone visualization dependencies

The optional visualization scripts use Python, NumPy, Matplotlib, Numba, and Tkinter. Those packages and the Python runtime are not vendored here and remain subject to their respective licenses.
