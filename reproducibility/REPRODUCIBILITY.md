# Reproducibility protocol

## Scope

This package contains the readable implementation and the input/output evidence required to inspect and rerun the manuscript homogenization examples:

- 27 Abaqus INP models;
- 28 reference effective-stiffness TXT outputs;
- 27 progress logs containing recorded preprocessing and solver timings;
- processed comparison and structure-level workbooks;
- structure-level CSV data;
- a case-specific Excel run guide.

## Reference environment

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
5. Set eight CPUs and the listed analysis/element options.
6. Select one existing node as the translational anchor and enter its exact coordinates.
7. Run preprocessing and the homogenization solver.
8. Compare the generated TXT matrix with the corresponding reference TXT.

## Integrity

`MANIFEST.sha256` records SHA-256 hashes for the distributed source, input, log, result, workbook, and CSV files. Recompute hashes after download to detect accidental changes. Generated Abaqus ODB files are not required to inspect the released results and are not included in the lightweight repository package.

## Validation status

The release source passed 82 ordinary Python unit/static tests and an implementation-similarity audit. The audit did not launch Abaqus. Before creating a final versioned release tag, the release owner should execute at least the six primary benchmark cases in Abaqus/CAE 2020 and compare their generated TXT matrices with the supplied references.
