# -*- coding: UTF-8 -*-
"""
Small numerical helpers for NIAH homogenization.

This module intentionally avoids Abaqus imports so the algebra can be tested
with a normal Python interpreter.
"""
from __future__ import print_function

import numpy as np


def assemble_effective_stiffness_from_chain(x1, x2, f1, f2, measure):
    """Assemble stiffness from the NIAH chain solution.

    ``x1`` and ``f1`` are the macro displacement field and its reaction.
    ``x2`` and ``f2`` are the signed periodic correction returned by the
    second and third Abaqus jobs. Because the second job applies ``-f1``,
    the stored correction is already negative. The final admissible state
    is therefore ``x1 + x2`` for both 3DOF and 6DOF discretizations.
    """
    if measure == 0:
        raise ValueError("measure must be nonzero.")

    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)
    f1 = np.asarray(f1, dtype=float)
    f2 = np.asarray(f2, dtype=float)

    if x1.shape != x2.shape:
        raise ValueError("x1 and x2 must have the same shape.")
    if f1.shape != f2.shape:
        raise ValueError("f1 and f2 must have the same shape.")
    if x1.shape != f1.shape:
        raise ValueError("displacement and force arrays must have the same shape.")

    return np.dot(x1 + x2, (f1 + f2).T) / float(measure)
