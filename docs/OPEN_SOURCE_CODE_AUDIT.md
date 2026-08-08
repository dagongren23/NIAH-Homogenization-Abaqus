# NIAH open-source code audit

Date: 2026-08-08

Target license: MIT

Runtime scope: Abaqus/CAE 2020 (Python 2.7-compatible kernel) and a standalone Python 3 visualization utility

## 1. PBC implementation provenance review

The supplied legacy `PBC_constraint_builder.py` was compared line by line and structurally with the locally installed EasyPBC 1.4 implementation. The installed EasyPBC distribution includes the GNU General Public License. Therefore, implementation-level copying would be incompatible with an unqualified repository-wide MIT release unless the affected code were separately licensed and distributed under the applicable GPL terms.

The legacy NIAH builder showed implementation-level similarity beyond the common periodic-boundary mathematics:

- 1,903 NIAH source lines versus 2,220 EasyPBC source lines;
- 135 NIAH executable records in exact normalized runs of at least three lines (8.8%);
- 31.6% identifier-normalized whole-file token similarity;
- repeated correspondence in face/edge classification, pairing loops, Abaqus set creation, and equation-generation order.

The legacy file was preserved for internal provenance review and excluded from the release. It was replaced by an independent implementation based on standard PBC kinematics, the published EasyPBC paper, and the NIAH software requirements. The rewrite separates Abaqus-independent mesh classification and deterministic coordinate matching into `periodic_mesh.py`; `PBC_constraint_builder.py` now adapts those results to Abaqus sets and equation constraints through data-driven relation tables.

Post-rewrite comparison results:

- 853 NIAH source lines;
- no exact normalized executable run of three or more lines;
- 4.1% identifier-normalized whole-file token similarity;
- no shared function names or non-trivial comments.

The remaining common identifiers are expected PBC/Abaqus vocabulary and backward-compatible NIAH set names. The post-rewrite technical verdict is low implementation similarity. This is an engineering provenance assessment, not legal advice.

Related paper used for attribution: Omairey, Dunning, and Sriramula, “Development of an Abaqus plugin tool for periodic RVE homogenisation,” *Engineering with Computers* 35 (2019) 567–577, https://doi.org/10.1007/s00366-018-0616-4.

## 2. Open-source normalization

The active release source was reviewed for readability and repository hygiene. The following changes were applied:

- converted active source comments, docstrings, labels, and error messages to English/ASCII;
- removed broad bare `except:` clauses in active modules;
- repaired stale or malformed annotations without changing numerical formulas;
- retained Abaqus wildcard imports where required by the Abaqus 2020 scripting API;
- removed unreachable statements from the plug-in dispatcher;
- added the new `periodic_mesh.py` dependency to the shear-resume fingerprint;
- replaced irreversible generated-file deletion with recoverable archiving under `delete/run_outputs` outside `niah_workbench`;
- kept cleanup in the successful plug-in path on every run; when resume is enabled, INP/STA/ODB checkpoints remain in place;
- excluded bytecode, IDE metadata, internal backups, and EasyPBC source from the release tree.

## 3. Verification performed

- 21 active source files compiled with the ordinary Python parser.
- Active Abaqus plug-in/kernel source contains no non-ASCII characters.
- No active source contains `os.remove`, recursive deletion, or a bare `except:` clause.
- 82 non-Abaqus unit/static tests passed, including periodic mesh classification, optional boundary topology, equation-rank checks, resume behavior, path handling, and recoverable cleanup.
- The post-rewrite EasyPBC similarity report found no exact normalized run of three or more executable lines.

Abaqus/CAE and the Abaqus solver were **not** launched for this audit. Dynamic Abaqus regression remains a release-owner acceptance step because it requires a licensed Abaqus installation and was not authorized in this task.

## 4. Release conclusion

On the reviewed technical evidence, the independently rewritten active source is suitable for publication under the repository MIT License with the literature attribution retained in the PBC module and third-party notice. Do not publish any file from `backup/`, `delete/`, `.idea/`, `__pycache__/`, or the locally installed EasyPBC directory.
