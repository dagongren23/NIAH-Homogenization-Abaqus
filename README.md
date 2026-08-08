# NIAH Homogenization for Abaqus

A unified, open-source Abaqus–Python plug-in that integrates established three-dimensional periodic homogenization and Reissner–Mindlin plate homogenization formulations in one automated workflow.

The repository provides the complete readable homogenization-kernel source, manuscript input models, reference progress logs, effective-stiffness TXT outputs, processed validation workbooks, and a case-specific execution guide. Abaqus itself is proprietary software and is not included.

## Capabilities

- automated boundary classification, deterministic opposite-node matching, and equation-based periodic constraints;
- 3D effective elastic stiffness recovery for solid, beam, shell, and supported mixed beam/shell models;
- plate extensional, coupling, bending, and diagonal transverse-shear stiffness recovery;
- Part-mode handling for uniform-property meshes and Model-mode handling for input files containing mixed elements or multiple section definitions;
- restart-aware load-case execution and recoverable output cleanup;
- standalone stiffness-result reading and directional-property visualization;
- non-Abaqus unit/static tests for the numerical helpers, topology logic, resume metadata, and cleanup policy.

## Repository layout

```text
NIAH Abaqus Plugin/                 Complete readable plug-in source
examples/input_files/               Manuscript and convergence-study INP files
reference_outputs/logs/             Recorded preprocessing/solver logs
reference_outputs/txt_results/      Effective stiffness reference outputs
reference_outputs/statistics/       Comparison and validation workbooks
reference_outputs/structural_data/  Structure-level CSV data
reproducibility/                    Run guide, protocol, and checksums
docs/                               Provenance audit and third-party notice
NIAH Abaqus Plugin/tests/           Ordinary Python tests that do not start Abaqus
```

## Tested environment

- Abaqus/CAE 2020;
- Abaqus built-in Python 2.7 for the plug-in kernel;
- Windows 10 Professional, 64-bit;
- 13th Gen Intel Core i7-13700KF, 32 GB RAM;
- eight solver CPUs for the supplied reference runs.

The standalone visualization utility uses Python 3 with NumPy, Matplotlib, Numba, and Tkinter.

## Installation

1. Clone or download the repository.
2. Copy the complete `NIAH Abaqus Plugin` directory into the Abaqus/CAE plug-ins directory, for example:

   ```text
   C:\SIMULIA\CAE\plugins\2020\
   ```

3. Restart Abaqus/CAE.
4. Open `Plug-ins > NIAH Homogenization`.

Do not copy only selected `.py` files: the plug-in depends on the complete `niah_core` directory, including `periodic_mesh.py`.

## Reproducing a manuscript case

1. Open `reproducibility/NIAH_case_run_guide.xlsx` and locate the required case.
2. Use the listed INP file from `examples/input_files` without renumbering its nodes.
3. Select the listed analysis type, preprocessing mode, primary element type, eight CPUs, and a mesh tolerance of `1.0E-3`.
4. Choose an existing node as the translational anchor and enter its exact coordinates.
5. Run preprocessing and homogenization.
6. Compare the generated TXT result with the corresponding file in `reference_outputs/txt_results` and inspect the recorded progress log.

See [REPRODUCIBILITY.md](reproducibility/REPRODUCIBILITY.md) for model groups, integrity checks, and release-validation status.

## Ordinary Python checks

From the repository root, run:

```text
cd "NIAH Abaqus Plugin"
python -m unittest discover -s tests -p "test_*.py"
```

These checks do not import or launch Abaqus. Dynamic Abaqus regression requires a licensed Abaqus installation and should be completed by the release owner before creating a versioned release tag.

## Provenance and attribution

The periodic boundary-condition implementation is an independent implementation of standard PBC kinematics and the NIAH workflow. Boundary classification and coordinate matching are isolated in an Abaqus-independent module. EasyPBC source code is not included.

Related literature acknowledged in the source:

- Omairey, Dunning, and Sriramula, “Development of an Abaqus plugin tool for periodic RVE homogenisation,” *Engineering with Computers* 35 (2019) 567–577. https://doi.org/10.1007/s00366-018-0616-4

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and [OPEN_SOURCE_CODE_AUDIT.md](docs/OPEN_SOURCE_CODE_AUDIT.md).

## Citation

Until the article record is final, cite this repository using the metadata in [`CITATION.cff`](CITATION.cff). The manuscript title is:

> A unified computational framework for asymptotic homogenization of periodic microstructures: 3D and Reissner–Mindlin plate numerical implementations

## License

The repository is released under the [MIT License](LICENSE). Abaqus is a product of Dassault Systèmes and is not distributed with this repository.
