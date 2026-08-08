# PBC source-similarity audit

## Scope and method

The audit compares the supplied NIAH PBC builder with the locally installed EasyPBC 1.4 source. It evaluates exact normalized line runs, identifier-normalized token structure, function organization, comments/messages, Abaqus set naming, and boundary-classification/control-flow patterns. Blank lines and indentation-only changes are ignored for the exact-run metric.

## Quantitative indicators

- NIAH source lines: 853
- EasyPBC source lines: 2220
- NIAH executable/non-comment records: 741
- NIAH records contained in exact normalized runs of at least three lines: 0 (0.0%)
- Identifier-normalized whole-file token similarity: 4.1%
- Shared function names: 0
- Shared non-trivial comment lines: 0

## Longest exact normalized runs

| NIAH lines | EasyPBC lines | Executable lines | First NIAH statement |
| --- | --- | ---: | --- |
| None | None | 0 | No run met the threshold |

## Structural indicators

Shared function names: none

Frequently shared distinctive identifiers include: `btedge`, `bbedge`, `bredge`, `backs`, `rbedge`, `fronts`, `bledge`, `lbedge`, `fbedge`, `fredge`, `fledge`, `ltedge`, `ftedge`, `rtedge`, `lefts`, `rights`, `backbc`, `frontbc`, `topbc`, `botbc`, `label`, `point`, `UNSET`, `rightbc`, `leftbc`, `error`, `constraints`, `region`, `models`, `meshsens`, `print`, `rbedgexyz`, `frontsxyz`, `leftsxyz`, `fbedgexyz`, `bledgexyz`, `ltedgexyz`, `btedgexyz`, `rightsxyz`, `backsxyz`, `fredgexyz`, `ftedgexyz`, `lbedgexyz`, `rtedgexyz`, `bbedgexyz`, `terms`, `value`, `bredgexyz`, `fledgexyz`, `topsxyz`, `botsxyz`, `rightbcxyz`, `UNIFORM`, `botbcxyz`, `frontbcxyz`, `backbcxyz`, `append`, `topbcxyz`, `leftbcxyz`, `errorset`, `periodic`, `boundary`, `shear`, `instances`, `ReferencePoint`, `multiprocessing`, `SetFromNodeLabels`, `creation`, `referencePoints`, `shell`, `Boundary`, `nodeLabels`, `features`, `without`, `module`, `Computers`, `round`, `rootAssembly`, `Python`, `fixed`

Representative shared comments/messages in the NIAH file:

- None detected by exact comment matching.

## Four requested checks

1. **Continuous multi-line identity or variable-only substitution:** No. No exact normalized executable run of at least three lines was detected, and identifier-normalized token similarity is low.
2. **Function partitioning, loop order, and if/elif structure:** No material implementation-level correspondence was detected. Both programs necessarily perform the domain-level stages of boundary classification and PBC equation creation, but the reviewed NIAH organization is independently data-driven.
3. **Abaqus Set creation and node classification:** No near-verbatim correspondence was detected. Legacy set names remain only as a compatibility interface for downstream NIAH routines; classification and matching are implemented in a separate Abaqus-independent module.
4. **Comments, messages, set names, and temporary identifiers:** No shared non-trivial comments or messages were detected. Remaining shared identifiers are expected PBC/Abaqus domain terms and compatibility set names.

## Verdict

Low implementation similarity after the independent rewrite. The remaining overlap is consistent with shared PBC mathematics, Abaqus API vocabulary, and backward-compatible NIAH set names.

This is a technical similarity assessment, not legal advice. The release decision should also consider the provenance and license terms of every third-party component.
