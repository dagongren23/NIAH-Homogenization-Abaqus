# -*- coding: UTF-8 -*-
"""
Ordering helpers for NIAH periodic boundary data.

These helpers intentionally avoid Abaqus imports so their ordering behavior can
be tested in a normal Python interpreter.
"""
from __future__ import print_function

import numpy as np


def _edge_group(name, prefix_a, labels_a, prefix_b, labels_b):
    labels_a = list(labels_a)
    labels_b = list(labels_b)
    if len(labels_a) != len(labels_b):
        raise ValueError(
            "%s length mismatch: %s (%d) != %s (%d)." %
            (name, prefix_a, len(labels_a), prefix_b, len(labels_b))
        )
    return name, prefix_a, labels_a, prefix_b, labels_b


def plate_homogeneous_free_surface_edge_pairs(fledge, bledge, bredge, fredge,
                                              ltedge, rtedge, rbedge, lbedge):
    """Return plate PBC pairs on z-free-surface boundary edges.

    Mindlin plate periodicity is imposed in the in-plane x/y directions only.
    Face-interior and vertical-edge constraints do not cover the boundary edges
    that lie on the free z-surfaces. These four reduced pair groups close those
    edges without tying zmax to zmin.
    """
    return [
        _edge_group("FLEDGE-BLEDGE", "fledge", fledge, "bledge", bledge),
        _edge_group("FREDGE-BREDGE", "fredge", fredge, "bredge", bredge),
        _edge_group("LTEDGE-LBEDGE", "ltedge", ltedge, "lbedge", lbedge),
        _edge_group("RTEDGE-RBEDGE", "rtedge", rtedge, "rbedge", rbedge),
    ]


def plate_shear_pure_edge_pairs(mode,
                                ftedge, btedge, fbedge, bbedge,
                                fledge, bledge, bredge, fredge,
                                ltedge, rtedge, rbedge, lbedge):
    """Return reduced pure-periodic edge pairs for shear PBC.

    The non-homogeneous direction already constrains one full boundary pair.
    The returned groups add only the complementary pure-periodic edge links
    needed to close the in-plane topology, avoiding a redundant four-edge loop.
    """
    mode = mode.upper()
    if mode == "SHXZ":
        return [
            # frontbc is the eliminated side of the non-homogeneous x pair.
            # Pivot on the retained backbc side to avoid reusing a DOF.
            _edge_group("BTEDGE-BBEDGE", "btedge", btedge, "bbedge", bbedge),
            _edge_group("LTEDGE-LBEDGE", "ltedge", ltedge, "lbedge", lbedge),
            _edge_group("RTEDGE-RBEDGE", "rtedge", rtedge, "rbedge", rbedge),
        ]
    if mode == "SHYZ":
        return [
            # topbc is the eliminated side of the non-homogeneous y pair.
            # Pivot on the retained botbc side to avoid reusing a DOF.
            _edge_group("FBEDGE-BBEDGE", "fbedge", fbedge, "bbedge", bbedge),
            _edge_group("FLEDGE-BLEDGE", "fledge", fledge, "bledge", bledge),
            _edge_group("FREDGE-BREDGE", "fredge", fredge, "bredge", bredge),
        ]
    raise ValueError("Unsupported mode: %s. Expected 'SHXZ' or 'SHYZ'." % mode)


def plate_shear_corner_pairs(mode):
    """Return a spanning-tree subset of complementary corner links.

    The full non-homogeneous face pair already links both sides in the primary
    periodic direction.  Only one complementary link per free z-surface is
    needed; adding the opposite-side link would close a redundant four-corner
    loop.
    """
    mode = mode.upper()
    if mode == "SHXZ":
        return [("c2", "c6"), ("c3", "c7")]
    if mode == "SHYZ":
        return [("c5", "c6"), ("c8", "c7")]
    raise ValueError("Unsupported mode: %s. Expected 'SHXZ' or 'SHYZ'." % mode)


def _optional_corner_label(corner_labels, corner_name):
    """Return one physical node label for an optional corner set.

    Empty corner sets are valid for open-cell/lattice geometries. More than one
    node in a geometric corner is ambiguous and must not be silently accepted.
    ``corner_labels=None`` preserves the legacy symbolic behavior for external
    callers that do not provide topology metadata.
    """
    if corner_labels is None:
        return corner_name

    labels = list(corner_labels.get(corner_name, []))
    if len(labels) > 1:
        raise ValueError(
            "Corner set %s contains %d nodes; expected at most one. "
            "Reduce the boundary tolerance or inspect the mesh." %
            (corner_name, len(labels))
        )
    if not labels:
        return None
    return labels[0]


def unique_periodic_pair_relations(groups):
    """Expand paired groups and remove duplicate physical-node relations.

    Shell meshes have ``zmin == zmax``. Consequently, zmax/zmin edge aliases
    can refer to the same physical nodes. Abaqus set names are different, but
    equations on those aliases constrain the same DOFs and are redundant.
    Deduplication therefore uses node labels, not set names.
    """
    relations = []
    seen_physical_pairs = set()
    for group_name, prefix_a, labels_a, prefix_b, labels_b in groups:
        labels_a = list(labels_a)
        labels_b = list(labels_b)
        if len(labels_a) != len(labels_b):
            raise ValueError(
                "%s length mismatch: %s (%d) != %s (%d)." %
                (group_name, prefix_a, len(labels_a), prefix_b, len(labels_b))
            )
        for label_a, label_b in zip(labels_a, labels_b):
            if label_a == label_b:
                continue
            physical_key = frozenset((label_a, label_b))
            if physical_key in seen_physical_pairs:
                continue
            seen_physical_pairs.add(physical_key)
            relations.append(
                (group_name, prefix_a, label_a, prefix_b, label_b)
            )
    return relations


def plate_homogeneous_corner_pairs(corner_labels=None):
    """Return a full-row-rank spanning forest for existing plate corners.

    Each free z-surface is handled independently for a solid plate. When a
    shell has coincident zmin/zmax corner aliases, physical-label
    deduplication keeps only one copy of each relation. Missing corners are
    removed before forming the chain, so no equation references a nonexistent
    Abaqus set.
    """
    plane_chains = (
        ("ZMAX", ("c1", "c2", "c6", "c5")),
        ("ZMIN", ("c4", "c3", "c7", "c8")),
    )
    relations = []
    seen_physical_pairs = set()

    for plane_name, corner_names in plane_chains:
        present = []
        seen_plane_labels = set()
        for corner_name in corner_names:
            label = _optional_corner_label(corner_labels, corner_name)
            if label is None or label in seen_plane_labels:
                continue
            present.append((corner_name, label))
            seen_plane_labels.add(label)

        for (name_a, label_a), (name_b, label_b) in zip(
                present[:-1], present[1:]):
            physical_key = frozenset((label_a, label_b))
            if physical_key in seen_physical_pairs:
                continue
            seen_physical_pairs.add(physical_key)
            relations.append(
                (plane_name, name_a, name_b, label_a, label_b)
            )

    return relations


def existing_corner_chain_pairs(corner_names, corner_labels=None):
    """Return a spanning tree over the corner sets that physically exist.

    The helper is intended for the legacy 3-D PBC path, where all eight
    rectangular-cell corners belong to one periodic equivalence class.
    Missing geometric corners are valid for open-cell structures and are
    removed before the tree is formed.  Coincident aliases are deduplicated by
    node label, and ambiguous multi-node corner sets are rejected.
    """
    present = []
    seen_labels = set()
    for corner_name in corner_names:
        label = _optional_corner_label(corner_labels, corner_name)
        if label is None or label in seen_labels:
            continue
        present.append((corner_name, label))
        seen_labels.add(label)

    return [
        (name_a, name_b, label_a, label_b)
        for (name_a, label_a), (name_b, label_b) in zip(
            present[:-1], present[1:]
        )
    ]


def plate_existing_shear_corner_pairs(mode, corner_labels=None):
    """Filter shear corner links by actual topology and physical node labels.

    A pair with both corners absent is a valid empty intersection and is
    skipped. A one-sided pair is a real periodic-topology mismatch: silently
    skipping it would leave an existing boundary node without its counterpart.
    """
    relations = []
    seen_physical_pairs = set()
    for set_a, set_b in plate_shear_corner_pairs(mode):
        label_a = _optional_corner_label(corner_labels, set_a)
        label_b = _optional_corner_label(corner_labels, set_b)

        if label_a is None and label_b is None:
            continue
        if (label_a is None) != (label_b is None):
            missing_name = set_a if label_a is None else set_b
            existing_name = set_b if label_a is None else set_a
            raise ValueError(
                "Periodic corner pair %s/%s is incomplete: %s exists but %s "
                "does not. Check the unit-cell geometry, mesh periodicity, "
                "and boundary tolerance." %
                (set_a, set_b, existing_name, missing_name)
            )
        if label_a == label_b:
            continue

        physical_key = frozenset((label_a, label_b))
        if physical_key in seen_physical_pairs:
            continue
        seen_physical_pairs.add(physical_key)
        relations.append((set_a, set_b, label_a, label_b))

    return relations


def ensure_job_output_ready(job_name, job_status, odb_exists, lock_exists):
    """Reject aborted or still-locked Abaqus outputs before opening an ODB."""
    status_text = str(job_status).upper()
    if status_text in ("ABORTED", "TERMINATED"):
        raise RuntimeError(
            "Job %s ended with status %s. Check %s.msg/.dat/.sta for the "
            "original Abaqus error." % (job_name, status_text, job_name)
        )
    if lock_exists:
        raise RuntimeError(
            "Job %s left an ODB lock file. The analysis did not finish "
            "cleanly or the ODB is still being written; check "
            "%s.msg/.dat/.sta." % (job_name, job_name)
        )
    if not odb_exists:
        raise RuntimeError(
            "Job %s finished but ODB was not found. Check %s.msg/.dat/.sta." %
            (job_name, job_name)
        )


def require_rotational_field(field_outputs, dof_per_node, field_name):
    """Require UR/RM output whenever a six-DOF extraction is requested."""
    if dof_per_node == 6 and field_name not in field_outputs.keys():
        raise RuntimeError(
            "6-DOF extraction requires field output '%s', but it is missing "
            "from the ODB frame. Check the Abaqus field output request." %
            field_name
        )


def sort_pairs_by_negative_label(positive_labels, negative_labels):
    """Return paired labels sorted by the negative-side label.

    The non-homogeneous shear PBC applies a prescribed jump value using the
    pair index. Sorting by the negative boundary label keeps that index aligned
    with displacement vectors flattened in node-label order.
    """
    if len(positive_labels) != len(negative_labels):
        raise ValueError(
            "positive_labels and negative_labels must have the same length."
        )

    pairs = sorted(
        zip(list(positive_labels), list(negative_labels)),
        key=lambda item: int(item[1])
    )
    return [item[0] for item in pairs], [item[1] for item in pairs]


def _values_to_component_map(values, component_count):
    out = {}
    for value in values:
        out[int(value.nodeLabel)] = tuple(float(x) for x in value.data[:component_count])
    return out


def flatten_values_by_node_label(values, labels=None, dof_per_node=3, rotation_values=None):
    """Flatten field values in node-label order.

    Parameters
    ----------
    values
        Abaqus field values with ``nodeLabel`` and translational ``data``.
    labels
        Optional explicit node-label order. If omitted, labels are sorted from
        ``values``.
    dof_per_node
        Either 3 or 6.
    rotation_values
        Optional Abaqus field values for UR. Used only when ``dof_per_node`` is
        6 or larger.
    """
    if dof_per_node not in (3, 6):
        raise ValueError("dof_per_node must be either 3 or 6.")

    value_list = list(values)
    if labels is None:
        ordered_labels = sorted(int(value.nodeLabel) for value in value_list)
    else:
        ordered_labels = [int(label) for label in labels]

    trans_by_label = _values_to_component_map(value_list, 3)
    rot_by_label = {}
    if dof_per_node >= 6 and rotation_values is not None:
        rot_by_label = _values_to_component_map(list(rotation_values), 3)

    arr = np.zeros((len(ordered_labels), dof_per_node), dtype=float)
    for row_index, label in enumerate(ordered_labels):
        if label in trans_by_label:
            arr[row_index, 0:3] = trans_by_label[label][0:3]
        if dof_per_node >= 6 and label in rot_by_label:
            arr[row_index, 3:6] = rot_by_label[label][0:3]

    return arr.flatten()
