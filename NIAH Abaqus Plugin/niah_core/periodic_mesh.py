# -*- coding: utf-8 -*-
"""Abaqus-independent boundary classification for rectangular periodic cells.

The routines in this module operate on node labels and coordinates only.  They
are deliberately separated from Abaqus model mutation so that correspondence
logic can be unit-tested in a normal Python interpreter.

This is an independent NIAH implementation of standard periodic-mesh
classification and coordinate matching.  It does not contain EasyPBC source.
"""
from __future__ import print_function

import itertools
import math


class PeriodicMeshError(ValueError):
    """Raised when the mesh cannot support one-to-one periodic pairing."""


LEGACY_REGION_NAMES = (
    'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8',
    'ftedge', 'fbedge', 'btedge', 'bbedge',
    'fledge', 'fredge', 'bledge', 'bredge',
    'ltedge', 'lbedge', 'rtedge', 'rbedge',
    'fronts', 'backs', 'lefts', 'rights', 'tops', 'bots',
    'frontbc', 'backbc', 'leftbc', 'rightbc', 'topbc', 'botbc',
)


_FACE_PAIR_SPECS = (
    ('fronts', 'backs', (1, 2)),
    ('tops', 'bots', (0, 2)),
    ('lefts', 'rights', (0, 1)),
    ('frontbc', 'backbc', (1, 2)),
    ('topbc', 'botbc', (0, 2)),
    ('leftbc', 'rightbc', (0, 1)),
)


_EDGE_FAMILY_SPECS = (
    (('ftedge', 'btedge', 'bbedge', 'fbedge'), (2,)),
    (('fledge', 'bledge', 'bredge', 'fredge'), (1,)),
    (('ltedge', 'rtedge', 'rbedge', 'lbedge'), (0,)),
)


def _node_records(nodes):
    """Return ``(label, (x, y, z))`` records from Abaqus or plain nodes."""
    records = []
    for item in nodes:
        if hasattr(item, 'label') and hasattr(item, 'coordinates'):
            label = int(item.label)
            coordinates = item.coordinates
        else:
            label, coordinates = item
            label = int(label)
        if len(coordinates) < 3:
            raise PeriodicMeshError(
                'Node %s has fewer than three coordinate components.' % label
            )
        records.append((
            label,
            (float(coordinates[0]), float(coordinates[1]),
             float(coordinates[2])),
        ))
    if not records:
        raise PeriodicMeshError('The selected instance contains no mesh nodes.')
    if len(set(label for label, _ in records)) != len(records):
        raise PeriodicMeshError('Node labels are not unique in the selected instance.')
    return records


def coordinate_bounds(nodes):
    """Return axis-aligned bounds and cell dimensions for a node sequence."""
    records = _node_records(nodes)
    axes = list(zip(*[coordinates for _, coordinates in records]))
    minima = tuple(min(axis) for axis in axes)
    maxima = tuple(max(axis) for axis in axes)
    dimensions = tuple(maximum - minimum
                       for minimum, maximum in zip(minima, maxima))
    return records, minima, maxima, dimensions


def _near(value, target, tolerance):
    return abs(value - target) <= tolerance


def _outside_endpoint(value, minimum, maximum, tolerance):
    return (not _near(value, minimum, tolerance) and
            not _near(value, maximum, tolerance))


def classify_boundary_nodes(nodes, tolerance):
    """Classify rectangular-cell boundary nodes into legacy NIAH regions.

    A region value is a dictionary mapping node label to the original
    coordinate triple.  Face-interior sets exclude edges and corners, whereas
    ``*bc`` face sets include the complete boundary face.  Zero-thickness
    shell cells are supported: a physical node may then belong to both z-side
    aliases, which is resolved later by physical-label deduplication.
    """
    tolerance = float(tolerance)
    if tolerance <= 0.0:
        raise PeriodicMeshError('Boundary tolerance must be greater than zero.')

    records, minima, maxima, dimensions = coordinate_bounds(nodes)
    x_min, y_min, z_min = minima
    x_max, y_max, z_max = maxima
    regions = dict((name, {}) for name in LEGACY_REGION_NAMES)

    corner_specs = (
        ('c1', x_max, y_max, z_max),
        ('c2', x_min, y_max, z_max),
        ('c3', x_min, y_max, z_min),
        ('c4', x_max, y_max, z_min),
        ('c5', x_max, y_min, z_max),
        ('c6', x_min, y_min, z_max),
        ('c7', x_min, y_min, z_min),
        ('c8', x_max, y_min, z_min),
    )

    for label, coordinates in records:
        x_value, y_value, z_value = coordinates
        x_low = _near(x_value, x_min, tolerance)
        x_high = _near(x_value, x_max, tolerance)
        y_low = _near(y_value, y_min, tolerance)
        y_high = _near(y_value, y_max, tolerance)
        z_low = _near(z_value, z_min, tolerance)
        z_high = _near(z_value, z_max, tolerance)

        complete_faces = (
            ('frontbc', x_high), ('backbc', x_low),
            ('topbc', y_high), ('botbc', y_low),
            ('leftbc', z_high), ('rightbc', z_low),
        )
        for name, present in complete_faces:
            if present:
                regions[name][label] = coordinates

        for name, x_target, y_target, z_target in corner_specs:
            if (_near(x_value, x_target, tolerance) and
                    _near(y_value, y_target, tolerance) and
                    _near(z_value, z_target, tolerance)):
                regions[name][label] = coordinates

        z_inside = _outside_endpoint(z_value, z_min, z_max, tolerance)
        y_inside = _outside_endpoint(y_value, y_min, y_max, tolerance)
        x_inside = _outside_endpoint(x_value, x_min, x_max, tolerance)

        edge_flags = (
            ('ftedge', x_high and y_high and z_inside),
            ('fbedge', x_high and y_low and z_inside),
            ('btedge', x_low and y_high and z_inside),
            ('bbedge', x_low and y_low and z_inside),
            ('fledge', x_high and z_high and y_inside),
            ('fredge', x_high and z_low and y_inside),
            ('bledge', x_low and z_high and y_inside),
            ('bredge', x_low and z_low and y_inside),
            ('ltedge', z_high and y_high and x_inside),
            ('lbedge', z_high and y_low and x_inside),
            ('rtedge', z_low and y_high and x_inside),
            ('rbedge', z_low and y_low and x_inside),
        )
        for name, present in edge_flags:
            if present:
                regions[name][label] = coordinates

        face_flags = (
            ('fronts', x_high and y_inside and z_inside),
            ('backs', x_low and y_inside and z_inside),
            ('tops', y_high and x_inside and z_inside),
            ('bots', y_low and x_inside and z_inside),
            ('lefts', z_high and x_inside and y_inside),
            ('rights', z_low and x_inside and y_inside),
        )
        for name, present in face_flags:
            if present:
                regions[name][label] = coordinates

    return {
        'regions': regions,
        'records': records,
        'minima': minima,
        'maxima': maxima,
        'dimensions': dimensions,
    }


def _projection(coordinates, coordinate_indices):
    return tuple(coordinates[index] for index in coordinate_indices)


def _bucket_key(projected_coordinates, tolerance):
    return tuple(int(math.floor(value / tolerance))
                 for value in projected_coordinates)


def _neighbor_keys(key):
    offsets = [(-1, 0, 1)] * len(key)
    for delta in itertools.product(*offsets):
        yield tuple(value + step for value, step in zip(key, delta))


def pair_regions(source, target, coordinate_indices, tolerance,
                 source_name='source', target_name='target'):
    """Return deterministic one-to-one label lists matched by projection."""
    if len(source) != len(target):
        raise PeriodicMeshError(
            '%s/%s node-count mismatch: %d != %d.' %
            (source_name, target_name, len(source), len(target))
        )
    if not source:
        return [], []

    tolerance = float(tolerance)
    buckets = {}
    for label, coordinates in target.items():
        projected = _projection(coordinates, coordinate_indices)
        key = _bucket_key(projected, tolerance)
        buckets.setdefault(key, []).append((int(label), projected))

    ordered_source = sorted(
        ((int(label), _projection(coordinates, coordinate_indices))
         for label, coordinates in source.items()),
        key=lambda item: item[1] + (item[0],),
    )
    used_target_labels = set()
    source_labels = []
    target_labels = []

    for source_label, source_projection in ordered_source:
        key = _bucket_key(source_projection, tolerance)
        candidates = []
        for neighbor in _neighbor_keys(key):
            for target_label, target_projection in buckets.get(neighbor, []):
                if target_label in used_target_labels:
                    continue
                differences = [abs(a - b) for a, b in
                               zip(source_projection, target_projection)]
                if all(difference <= tolerance for difference in differences):
                    squared_distance = sum(difference * difference
                                           for difference in differences)
                    candidates.append((squared_distance, target_label))
        if not candidates:
            raise PeriodicMeshError(
                'No %s counterpart was found for %s node %s within tolerance %g.' %
                (target_name, source_name, source_label, tolerance)
            )
        candidates.sort()
        chosen_label = candidates[0][1]
        used_target_labels.add(chosen_label)
        source_labels.append(source_label)
        target_labels.append(chosen_label)

    if len(used_target_labels) != len(target):
        unused = sorted(set(int(label) for label in target) - used_target_labels)
        raise PeriodicMeshError(
            '%s contains unmatched nodes: %s.' %
            (target_name, ', '.join(str(label) for label in unused[:20]))
        )
    return source_labels, target_labels


def validate_correspondence_counts(regions):
    """Return a list of region families with inconsistent node counts."""
    mismatches = []
    for positive_name, negative_name, _ in _FACE_PAIR_SPECS:
        if len(regions[positive_name]) != len(regions[negative_name]):
            mismatches.append((positive_name, negative_name))
    for family_names, _ in _EDGE_FAMILY_SPECS:
        counts = [len(regions[name]) for name in family_names]
        if counts and min(counts) != max(counts):
            mismatches.append(tuple(family_names))
    return mismatches


def build_ordered_boundary_sets(nodes, tolerance):
    """Classify nodes and order every periodic counterpart list."""
    classification = classify_boundary_nodes(nodes, tolerance)
    regions = classification['regions']
    mismatches = validate_correspondence_counts(regions)
    if mismatches:
        descriptions = []
        for family in mismatches:
            descriptions.append(', '.join(
                '%s=%d' % (name, len(regions[name])) for name in family
            ))
        raise PeriodicMeshError(
            'Periodic boundary node counts are inconsistent: %s.' %
            '; '.join(descriptions)
        )

    ordered = {}
    for positive_name, negative_name, coordinate_indices in _FACE_PAIR_SPECS:
        positive, negative = pair_regions(
            regions[positive_name], regions[negative_name],
            coordinate_indices, tolerance,
            source_name=positive_name, target_name=negative_name,
        )
        ordered[positive_name] = positive
        ordered[negative_name] = negative

    for family_names, coordinate_indices in _EDGE_FAMILY_SPECS:
        reference_name = family_names[0]
        reference_region = regions[reference_name]
        if not reference_region:
            for name in family_names:
                ordered[name] = []
            continue
        reference_order = None
        for target_name in family_names[1:]:
            positive, negative = pair_regions(
                reference_region, regions[target_name],
                coordinate_indices, tolerance,
                source_name=reference_name, target_name=target_name,
            )
            if reference_order is None:
                reference_order = positive
            elif positive != reference_order:
                raise PeriodicMeshError(
                    'Internal ordering mismatch in edge family %s.' %
                    ', '.join(family_names)
                )
            ordered[target_name] = negative
        ordered[reference_name] = reference_order or []

    for corner_name in ('c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8'):
        ordered[corner_name] = sorted(int(label)
                                      for label in regions[corner_name])

    classification['ordered_sets'] = ordered
    return classification
