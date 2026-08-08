# -*- coding: UTF-8 -*-
"""Pure helpers for making imported shear base models reusable."""
from __future__ import print_function


def _clear_repository(repository):
    names = list(repository.keys())
    for name in names:
        del repository[name]
    return len(names)


def reset_imported_shear_base_model(model):
    """Remove solver-generated state before rebuilding nonhomogeneous PBC.

    The X/Y shear base INP is a geometry/mesh template.  Sets, equations,
    boundary conditions and reference points written by an earlier solver run
    must not be carried into the next run, otherwise constraints accumulate.
    """
    assembly = model.rootAssembly
    removed = {
        'sets': _clear_repository(assembly.sets),
        'constraints': _clear_repository(model.constraints),
        'boundary_conditions': _clear_repository(
            model.boundaryConditions
        ),
        'reference_point_features': 0,
    }

    rp_names = [
        name for name in list(assembly.features.keys())
        if str(name).startswith('RP-')
    ]
    for name in rp_names:
        del assembly.features[name]
    removed['reference_point_features'] = len(rp_names)
    return removed
