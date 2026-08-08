# Reproducibility protocol

## Scope

This package contains the readable implementation and the input/output evidence required to inspect and rerun the manuscript homogenization examples:

- 27 Abaqus INP models;
- 28 reference effective-stiffness TXT outputs;
- 27 progress logs containing recorded preprocessing and solver timings;
- processed comparison and structure-level workbooks;
- structure-level CSV and XLSX data;
- a case-specific Excel run guide.

## Supported Abaqus environments

- Abaqus/CAE 2020 with its built-in Python 2.7;
- Abaqus/CAE 2024 with its built-in Python 3.10;

## Reference timing environment

- Abaqus/CAE 2020;
- Windows 10 Professional, 64-bit;
- 13th Gen Intel Core i7-13700KF;
- 32 GB DDR4-3200 memory;
- WD Blue SN580 1 TB SSD;
- eight solver CPUs.

Wall-clock time is machine- and load-dependent. The supplied logs are the authoritative timing records for the reference computer.

## Case groups

- `Cube_E*`: 3D solid-cube mesh-convergence study.
- `Cubic_center_solid`: C3D4 solid-lattice benchmark.
- `Cubic_center_beam` and `Octet_beam`: beam-lattice cases with INP-defined sections.
- `Plate_T*`: solid-plate mesh-convergence study.
- `Hexagonal_honeycomb_*`: S4R, S4, and high-density C3D8R formulation comparison.
- `Truss_sandwich_solid`: C3D20R plate benchmark.
- `Lattice_Hybrid_model`: mixed B31/S3 demonstration, with 3D and plate output records.

## Execution

1. Install the complete plug-in directory as described in the root README.
2. Open `NIAH_case_run_guide.xlsx` and use the row corresponding to the target result.
3. Preserve the supplied mesh and node numbering.
4. Use Part mode for uniform-property cases and Model mode for cases that retain INP-defined mixed elements or multiple sections.
5. Set the listed analysis/element options. The supplied reference runs used eight CPUs; for new runs, another even CPU count may be used provided that it does not exceed the computer's available CPU count.
6. Select one existing node as the translational anchor and enter its exact coordinates.
7. Run preprocessing and the homogenization solver.
8. Compare the generated TXT matrix with the corresponding reference TXT.

## Validation status

The plug-in has been validated with representative cases in Abaqus/CAE 2020 and Abaqus/CAE 2024. The manuscript cases can be rerun using the supplied input files and compared with the corresponding TXT results and progress logs. Generated Abaqus ODB files are not required to inspect the released results and are not included in the lightweight repository package.
