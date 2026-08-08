# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
"""Create NIAH periodic boundary sets and Abaqus equation constraints.

This module is an independent implementation based on standard periodic
boundary-condition kinematics and the NIAH workflow. Boundary classification
and node matching are implemented without Abaqus dependencies in
``periodic_mesh.py``.

Related literature: Omairey, Dunning, and Sriramula, Engineering with
Computers 35 (2019), 567-577. EasyPBC source is not included here.
"""
from __future__ import print_function

import logging
import multiprocessing
import time

try:
    from abaqus import mdb
    from abaqusConstants import OFF, UNIFORM, UNSET
except ImportError:  # Permit import and static tests outside Abaqus.
    mdb = None
    OFF = 'OFF'
    UNIFORM = 'UNIFORM'
    UNSET = 'UNSET'

from boundary_ordering import (
    existing_corner_chain_pairs,
    plate_existing_shear_corner_pairs,
    plate_homogeneous_corner_pairs,
    plate_homogeneous_free_surface_edge_pairs,
    plate_shear_pure_edge_pairs,
    sort_pairs_by_negative_label,
    unique_periodic_pair_relations,
)
from periodic_mesh import (
    PeriodicMeshError,
    build_ordered_boundary_sets,
    pair_regions,
)


THREE_DOF = (1, 2, 3)
SIX_DOF = (1, 2, 3, 4, 5, 6)
CORNER_CHAIN = ('c1', 'c5', 'c4', 'c8', 'c2', 'c6', 'c3', 'c7')
LOG_FORMAT = '%(levelname)s:%(name)s:%(message)s'


def get_logger(name=__name__):
    """Return a logger suitable for the Abaqus/CAE console."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


_LOG = get_logger(__name__)


def _return_error(message, return_value):
    _LOG.error(message)
    try:
        print(message)
    except Exception:
        pass
    return return_value


def _require_abaqus():
    if mdb is None:
        raise RuntimeError(
            'PBC constraint creation must run inside the Abaqus Python environment.'
        )


def _clear_repository(repository):
    for name in list(repository.keys()):
        del repository[name]


def _set_name(prefix, label):
    return '%s%s' % (prefix, label)


def _create_reference_set(assembly, name, point):
    feature = assembly.ReferencePoint(point=point)
    reference_point = assembly.referencePoints[feature.id]
    assembly.Set(referencePoints=(reference_point,), name=name)
    return name


def _prepare_global_reference_sets(assembly, minima, maxima):
    """Create six deterministic global reference-point sets."""
    for set_name in ('RP1', 'RP2', 'RP3', 'RP4', 'RP5', 'RP6'):
        if set_name in assembly.sets.keys():
            del assembly.sets[set_name]
    for feature_name in list(assembly.features.keys()):
        if str(feature_name).startswith('RP'):
            del assembly.features[feature_name]

    x_min, y_min, z_min = minima
    x_max, y_max, z_max = maxima
    x_mid = 0.5 * (x_min + x_max)
    y_mid = 0.5 * (y_min + y_max)
    z_mid = 0.5 * (z_min + z_max)
    x_span = abs(x_max - x_min)
    y_span = abs(y_max - y_min)
    z_span = abs(z_max - z_min)
    points = (
        ('RP1', (x_mid, y_max + 0.2 * y_span, z_mid)),
        ('RP2', (x_mid, y_mid, z_max + 0.2 * z_span)),
        ('RP3', (x_max + 0.2 * x_span, y_mid, z_mid)),
        ('RP4', (x_max + 0.4 * x_span, y_mid, z_mid)),
        ('RP5', (x_max + 0.6 * x_span, y_mid, z_mid)),
        ('RP6', (x_max + 0.8 * x_span, y_mid, z_mid)),
    )
    for name, point in points:
        _create_reference_set(assembly, name, point)


def _create_region_sets(assembly, instance_name, ordered_sets):
    empty_set_names = []
    for set_name in sorted(ordered_sets):
        node_labels = ordered_sets[set_name]
        if node_labels:
            assembly.SetFromNodeLabels(
                name=set_name,
                nodeLabels=((instance_name, tuple(node_labels)),),
            )
        else:
            empty_set_names.append(set_name)
    if empty_set_names:
        print(
            'NIAH PBC: optional empty boundary sets were skipped: %s' %
            ', '.join(empty_set_names)
        )


def _create_singleton_sets(assembly, instance_name, prefix, labels):
    for label in labels:
        assembly.SetFromNodeLabels(
            name=_set_name(prefix, label),
            nodeLabels=((instance_name, (label,)),),
        )


def create_single_nodes_set(
        assembly, instance_name,
        fronts, backs, tops, bots, lefts, rights,
        ftedge, btedge, fbedge, bbedge,
        fledge, bledge, bredge, fredge,
        ltedge, rtedge, rbedge, lbedge,
        frontbc, backbc, topbc, botbc, leftbc, rightbc):
    """Create one-node Abaqus sets referenced by equation terms."""
    groups = (
        ('fronts', fronts), ('backs', backs),
        ('tops', tops), ('bots', bots),
        ('lefts', lefts), ('rights', rights),
        ('ftedge', ftedge), ('btedge', btedge),
        ('fbedge', fbedge), ('bbedge', bbedge),
        ('fledge', fledge), ('bledge', bledge),
        ('bredge', bredge), ('fredge', fredge),
        ('ltedge', ltedge), ('rtedge', rtedge),
        ('rbedge', rbedge), ('lbedge', lbedge),
        ('frontbc', frontbc), ('backbc', backbc),
        ('topbc', topbc), ('botbc', botbc),
        ('leftbc', leftbc), ('rightbc', rightbc),
    )
    for prefix, labels in groups:
        _create_singleton_sets(assembly, instance_name, prefix, labels)


def _pair_relations(group_name, prefix_a, labels_a, prefix_b, labels_b):
    if len(labels_a) != len(labels_b):
        raise ValueError(
            '%s length mismatch: %s=%d, %s=%d.' %
            (group_name, prefix_a, len(labels_a), prefix_b, len(labels_b))
        )
    return [
        (group_name, prefix_a, label_a, prefix_b, label_b)
        for label_a, label_b in zip(labels_a, labels_b)
    ]


def _add_equation(model, name, set_a, set_b, dof, reference_set=None):
    terms = [(1.0, set_a, dof), (-1.0, set_b, dof)]
    if reference_set is not None:
        terms.append((-1.0, reference_set, dof))
    model.Equation(name=name, terms=tuple(terms))


def _tie_relations(model, relations, dofs, name_prefix):
    for relation_index, relation in enumerate(relations, 1):
        group_name, prefix_a, label_a, prefix_b, label_b = relation
        for dof in dofs:
            _add_equation(
                model,
                '%s-%s-%d-%d' %
                (name_prefix, group_name, relation_index, dof),
                _set_name(prefix_a, label_a),
                _set_name(prefix_b, label_b),
                dof,
            )


def _tie_corner_sets(model, relations, dofs, name_prefix):
    for relation_index, relation in enumerate(relations, 1):
        set_a, set_b = relation[0], relation[1]
        for dof in dofs:
            _add_equation(
                model,
                '%s-%d-%d' % (name_prefix, relation_index, dof),
                set_a, set_b, dof,
            )


def _three_dimensional_relations(
        fronts, backs, tops, bots, lefts, rights,
        ftedge, btedge, fbedge, bbedge,
        fledge, bledge, bredge, fredge,
        ltedge, rtedge, rbedge, lbedge):
    relations = []
    for spec in (
            ('FACE-X', 'fronts', fronts, 'backs', backs),
            ('FACE-Y', 'tops', tops, 'bots', bots),
            ('FACE-Z', 'lefts', lefts, 'rights', rights)):
        relations.extend(_pair_relations(*spec))

    for front_left, back_left, back_right, front_right in zip(
            fledge, bledge, bredge, fredge):
        relations.extend((
            ('EDGE-Y-1', 'fledge', front_left, 'bledge', back_left),
            ('EDGE-Y-2', 'bledge', back_left, 'bredge', back_right),
            ('EDGE-Y-3', 'bredge', back_right, 'fredge', front_right),
        ))
    for left_top, right_top, right_bottom, left_bottom in zip(
            ltedge, rtedge, rbedge, lbedge):
        relations.extend((
            ('EDGE-X-1', 'ltedge', left_top, 'rtedge', right_top),
            ('EDGE-X-2', 'rtedge', right_top, 'rbedge', right_bottom),
            ('EDGE-X-3', 'lbedge', left_bottom, 'rbedge', right_bottom),
        ))
    for front_top, back_top, back_bottom, front_bottom in zip(
            ftedge, btedge, bbedge, fbedge):
        relations.extend((
            ('EDGE-Z-1', 'ftedge', front_top, 'btedge', back_top),
            ('EDGE-Z-2', 'btedge', back_top, 'bbedge', back_bottom),
            ('EDGE-Z-3', 'fbedge', front_bottom, 'bbedge', back_bottom),
        ))
    return relations


def create_3D_periodic_constraints_equation(
        model_name, fronts, backs, tops, bots, lefts, rights,
        ftedge, btedge, fbedge, bbedge,
        fledge, bledge, bredge, fredge,
        ltedge, rtedge, rbedge, lbedge, corner_labels=None):
    """Create translational 3D homogeneous periodic equations."""
    model = mdb.models[model_name]
    _clear_repository(model.constraints)
    relations = _three_dimensional_relations(
        fronts, backs, tops, bots, lefts, rights,
        ftedge, btedge, fbedge, bbedge,
        fledge, bledge, bredge, fredge,
        ltedge, rtedge, rbedge, lbedge,
    )
    _tie_relations(model, relations, THREE_DOF, 'NIAH-3D')
    corners = existing_corner_chain_pairs(CORNER_CHAIN, corner_labels)
    _tie_corner_sets(model, corners, THREE_DOF, 'NIAH-3D-CORNER')


def create_3D_periodic_constraints_equation_6dof(
        model_name, fronts, backs, tops, bots, lefts, rights,
        ftedge, btedge, fbedge, bbedge,
        fledge, bledge, bredge, fredge,
        ltedge, rtedge, rbedge, lbedge, corner_labels=None):
    """Create translational and rotational 3D homogeneous equations."""
    model = mdb.models[model_name]
    _clear_repository(model.constraints)
    relations = _three_dimensional_relations(
        fronts, backs, tops, bots, lefts, rights,
        ftedge, btedge, fbedge, bbedge,
        fledge, bledge, bredge, fredge,
        ltedge, rtedge, rbedge, lbedge,
    )
    for d in (1, 2, 3, 4, 5, 6):
        _tie_relations(model, relations, (d,), 'NIAH-3D6')
    corners = existing_corner_chain_pairs(CORNER_CHAIN, corner_labels)
    _tie_corner_sets(model, corners, SIX_DOF, 'NIAH-3D6-CORNER')


def _plate_homogeneous_relations(
        fronts, backs, tops, bots,
        ftedge, btedge, fbedge, bbedge,
        fledge, bledge, bredge, fredge,
        ltedge, rtedge, rbedge, lbedge):
    relations = []
    relations.extend(_pair_relations(
        'PLATE-X', 'fronts', fronts, 'backs', backs
    ))
    relations.extend(_pair_relations(
        'PLATE-Y', 'tops', tops, 'bots', bots
    ))
    for front_top, back_top, back_bottom, front_bottom in zip(
            ftedge, btedge, bbedge, fbedge):
        relations.extend((
            ('PLATE-EDGE-1', 'ftedge', front_top, 'btedge', back_top),
            ('PLATE-EDGE-2', 'btedge', back_top, 'bbedge', back_bottom),
            ('PLATE-EDGE-3', 'fbedge', front_bottom, 'bbedge', back_bottom),
        ))
    free_surface_groups = plate_homogeneous_free_surface_edge_pairs(
        fledge, bledge, bredge, fredge,
        ltedge, rtedge, rbedge, lbedge,
    )
    relations.extend(unique_periodic_pair_relations(free_surface_groups))
    return relations


def _create_plate_homogeneous_constraints(
        model_name, dofs, fronts, backs, tops, bots,
        ftedge, btedge, fbedge, bbedge,
        fledge, bledge, bredge, fredge,
        ltedge, rtedge, rbedge, lbedge, corner_labels):
    model = mdb.models[model_name]
    _clear_repository(model.constraints)
    relations = _plate_homogeneous_relations(
        fronts, backs, tops, bots,
        ftedge, btedge, fbedge, bbedge,
        fledge, bledge, bredge, fredge,
        ltedge, rtedge, rbedge, lbedge,
    )
    _tie_relations(model, relations, dofs, 'NIAH-PLATE')
    corners = plate_homogeneous_corner_pairs(corner_labels)
    _tie_corner_sets(model, corners, dofs, 'NIAH-PLATE-CORNER')


def create_mindlin_periodic_constrains_equation_3dof(
        model_name, fronts, backs, tops, bots,
        ftedge, btedge, fbedge, bbedge,
        fledge=None, bledge=None, bredge=None, fredge=None,
        ltedge=None, rtedge=None, rbedge=None, lbedge=None,
        corner_labels=None):
    """Create the three-DOF plate homogeneous PBC system."""
    _create_plate_homogeneous_constraints(
        model_name, THREE_DOF, fronts, backs, tops, bots,
        ftedge, btedge, fbedge, bbedge,
        fledge or [], bledge or [], bredge or [], fredge or [],
        ltedge or [], rtedge or [], rbedge or [], lbedge or [],
        corner_labels,
    )


def create_mindlin_periodic_constrains_equation_6dof(
        model_name, fronts, backs, tops, bots,
        ftedge, btedge, fbedge, bbedge,
        fledge=None, bledge=None, bredge=None, fredge=None,
        ltedge=None, rtedge=None, rbedge=None, lbedge=None,
        corner_labels=None):
    """Create the six-DOF beam/shell plate homogeneous PBC system."""
    _create_plate_homogeneous_constraints(
        model_name, SIX_DOF, fronts, backs, tops, bots,
        ftedge, btedge, fbedge, bbedge,
        fledge or [], bledge or [], bredge or [], fredge or [],
        ltedge or [], rtedge or [], rbedge or [], lbedge or [],
        corner_labels,
    )


def _node_coordinate_map(nodes):
    return dict(
        (int(node.label), tuple(float(value) for value in node.coordinates[:3]))
        for node in nodes
    )


def _displacement_bc_arguments(name, step_name, region, dof, value):
    arguments = {
        'name': name, 'createStepName': step_name, 'region': region,
        'u1': UNSET, 'u2': UNSET, 'u3': UNSET,
        'ur1': UNSET, 'ur2': UNSET, 'ur3': UNSET,
        'amplitude': UNSET, 'fixed': OFF,
        'distributionType': UNIFORM, 'fieldName': '', 'localCsys': None,
    }
    component_names = ('u1', 'u2', 'u3', 'ur1', 'ur2', 'ur3')
    arguments[component_names[dof - 1]] = value
    return arguments


def _create_plate_shear_constraints(
        assembly, model_name, nodes,
        fronts, backs, tops, bots,
        frontbc, backbc, topbc, botbc,
        mode, prescribed_values, cell_length_x, cell_length_y, step_name,
        ftedge, btedge, fbedge, bbedge,
        fledge, bledge, bredge, fredge,
        ltedge, rtedge, rbedge, lbedge,
        corner_labels, dofs):
    mode = str(mode).upper()
    if mode not in ('SHXZ', 'SHYZ'):
        raise ValueError("Unsupported shear mode '%s'." % mode)
    if prescribed_values is None:
        raise ValueError('Prescribed shear displacement values are required.')

    if mode == 'SHXZ':
        positive_prefix, negative_prefix = 'frontbc', 'backbc'
        positive_labels, negative_labels = frontbc, backbc
        pure_prefix_a, pure_prefix_b = 'tops', 'bots'
        pure_labels_a, pure_labels_b = tops, bots
        span, offset_axis = cell_length_x, 0
    else:
        positive_prefix, negative_prefix = 'topbc', 'botbc'
        positive_labels, negative_labels = topbc, botbc
        pure_prefix_a, pure_prefix_b = 'fronts', 'backs'
        pure_labels_a, pure_labels_b = fronts, backs
        span, offset_axis = cell_length_y, 1

    if span is None or float(span) <= 0.0:
        raise ValueError('A positive in-plane cell dimension is required for %s.' % mode)
    positive_labels, negative_labels = sort_pairs_by_negative_label(
        positive_labels, negative_labels
    )
    required_value_count = len(dofs) * len(positive_labels)
    if len(prescribed_values) < required_value_count:
        raise ValueError(
            'Insufficient prescribed values: received %d, expected at least %d.' %
            (len(prescribed_values), required_value_count)
        )

    model = mdb.models[model_name]
    _clear_repository(model.constraints)
    coordinates_by_label = _node_coordinate_map(nodes)
    for pair_index, (positive_label, negative_label) in enumerate(
            zip(positive_labels, negative_labels)):
        if negative_label not in coordinates_by_label:
            raise ValueError('Node label %s is not present in the instance.' % negative_label)
        base_coordinates = list(coordinates_by_label[negative_label])
        for value_index, dof in enumerate(dofs):
            point = list(base_coordinates)
            point[offset_axis] -= float(dof) * float(span)
            reference_name = 'NIAH-RP-%s-%s-D%d' % (
                mode, positive_label, dof
            )
            _create_reference_set(assembly, reference_name, tuple(point))
            _add_equation(
                model,
                'NIAH-%s-JUMP-%d-%d' % (mode, pair_index + 1, dof),
                _set_name(positive_prefix, positive_label),
                _set_name(negative_prefix, negative_label),
                dof, reference_set=reference_name,
            )
            value = prescribed_values[pair_index * len(dofs) + value_index]
            model.DisplacementBC(**_displacement_bc_arguments(
                'BC-' + reference_name, step_name,
                assembly.sets[reference_name], dof, value,
            ))

    pure_relations = _pair_relations(
        mode + '-FACE', pure_prefix_a, pure_labels_a,
        pure_prefix_b, pure_labels_b,
    )
    _tie_relations(model, pure_relations, dofs, 'NIAH-' + mode)
    edge_groups = plate_shear_pure_edge_pairs(
        mode,
        ftedge, btedge, fbedge, bbedge,
        fledge, bledge, bredge, fredge,
        ltedge, rtedge, rbedge, lbedge,
    )
    edge_relations = unique_periodic_pair_relations(edge_groups)
    _tie_relations(model, edge_relations, dofs, 'NIAH-' + mode + '-EDGE')
    corners = plate_existing_shear_corner_pairs(mode, corner_labels)
    _tie_corner_sets(model, corners, dofs, 'NIAH-' + mode + '-CORNER')


def create_mindlin_nonhomogeneous_periodic_constraints_equation(
        assembly, model_name, nodes,
        fronts, backs, tops, bots, frontbc, backbc, topbc, botbc,
        mode, us1minus, fs1minus, L=None, H=None, step_name='Step-1',
        ftedge=None, btedge=None, fbedge=None, bbedge=None,
        fledge=None, bledge=None, bredge=None, fredge=None,
        ltedge=None, rtedge=None, rbedge=None, lbedge=None,
        corner_labels=None):
    """Create three-DOF non-homogeneous plate shear constraints."""
    _ = fs1minus
    _create_plate_shear_constraints(
        assembly, model_name, nodes, fronts, backs, tops, bots,
        frontbc, backbc, topbc, botbc, mode, us1minus, L, H, step_name,
        ftedge or [], btedge or [], fbedge or [], bbedge or [],
        fledge or [], bledge or [], bredge or [], fredge or [],
        ltedge or [], rtedge or [], rbedge or [], lbedge or [],
        corner_labels, THREE_DOF,
    )


def create_mindlin_nonhomogeneous_periodic_constraints_equation_6dof(
        assembly, model_name, nodes,
        fronts, backs, tops, bots, frontbc, backbc, topbc, botbc,
        mode, us1minus, fs1minus, L=None, H=None, step_name='Step-1',
        ftedge=None, btedge=None, fbedge=None, bbedge=None,
        fledge=None, bledge=None, bredge=None, fredge=None,
        ltedge=None, rtedge=None, rbedge=None, lbedge=None,
        corner_labels=None):
    """Create six-DOF non-homogeneous beam/shell shear constraints."""
    _ = fs1minus
    _create_plate_shear_constraints(
        assembly, model_name, nodes, fronts, backs, tops, bots,
        frontbc, backbc, topbc, botbc, mode, us1minus, L, H, step_name,
        ftedge or [], btedge or [], fbedge or [], bbedge or [],
        fledge or [], bledge or [], bredge or [], fredge or [],
        ltedge or [], rtedge or [], rbedge or [], lbedge or [],
        corner_labels, SIX_DOF,
    )


def _ordered_constraint_arguments(ordered_sets):
    return [ordered_sets[name] for name in (
        'fronts', 'backs', 'tops', 'bots', 'lefts', 'rights',
        'ftedge', 'btedge', 'fbedge', 'bbedge',
        'fledge', 'bledge', 'bredge', 'fredge',
        'ltedge', 'rtedge', 'rbedge', 'lbedge',
    )]


def main_apply_pbc_constraint(model_name, instance_name, meshsens, CPU,
                              condition_ch, mode=None, us1minus=None,
                              fs1minus=None):
    """Classify the RVE boundary and apply the selected NIAH PBC system."""
    global mass, L, H, W, error
    _require_abaqus()
    if model_name not in mdb.models.keys():
        return _return_error(
            "Model '%s' does not exist." % model_name,
            (None, None, None, None, True),
        )
    if float(meshsens) <= 0.0:
        return _return_error(
            'Boundary tolerance must be greater than zero.',
            (None, None, None, None, True),
        )
    if float(CPU) <= 0.0:
        return _return_error(
            'The requested CPU count must be greater than zero.',
            (None, None, None, None, True),
        )

    model = mdb.models[model_name]
    assembly = model.rootAssembly
    if instance_name not in assembly.instances.keys():
        return _return_error(
            "Instance '%s' does not exist in model '%s'." %
            (instance_name, model_name),
            (None, None, None, None, True),
        )
    requested_cpus = int(round(float(CPU)))
    available_cpus = multiprocessing.cpu_count()
    if requested_cpus > available_cpus:
        print(
            'NIAH PBC: requested %d CPUs; this workstation reports %d.' %
            (requested_cpus, available_cpus)
        )

    start_time = time.time()
    instance_nodes = assembly.instances[instance_name].nodes
    try:
        boundary_data = build_ordered_boundary_sets(instance_nodes, meshsens)
    except PeriodicMeshError as exc:
        return _return_error(
            'NIAH PBC boundary classification failed: %s' % exc,
            (None, None, None, None, True),
        )

    L, H, W = boundary_data['dimensions']
    error = False
    ordered_sets = boundary_data['ordered_sets']
    _prepare_global_reference_sets(
        assembly, boundary_data['minima'], boundary_data['maxima']
    )
    _create_region_sets(assembly, instance_name, ordered_sets)
    create_single_nodes_set(
        assembly, instance_name,
        ordered_sets['fronts'], ordered_sets['backs'],
        ordered_sets['tops'], ordered_sets['bots'],
        ordered_sets['lefts'], ordered_sets['rights'],
        ordered_sets['ftedge'], ordered_sets['btedge'],
        ordered_sets['fbedge'], ordered_sets['bbedge'],
        ordered_sets['fledge'], ordered_sets['bledge'],
        ordered_sets['bredge'], ordered_sets['fredge'],
        ordered_sets['ltedge'], ordered_sets['rtedge'],
        ordered_sets['rbedge'], ordered_sets['lbedge'],
        ordered_sets['frontbc'], ordered_sets['backbc'],
        ordered_sets['topbc'], ordered_sets['botbc'],
        ordered_sets['leftbc'], ordered_sets['rightbc'],
    )

    if condition_ch in (1, 2):
        arguments = [model_name] + _ordered_constraint_arguments(ordered_sets)
        arguments.append(ordered_sets)
        if condition_ch == 1:
            create_3D_periodic_constraints_equation(*arguments)
        else:
            create_3D_periodic_constraints_equation_6dof(*arguments)
    elif condition_ch in (3, 4):
        function = (create_mindlin_periodic_constrains_equation_3dof
                    if condition_ch == 3 else
                    create_mindlin_periodic_constrains_equation_6dof)
        function(
            model_name,
            ordered_sets['fronts'], ordered_sets['backs'],
            ordered_sets['tops'], ordered_sets['bots'],
            ordered_sets['ftedge'], ordered_sets['btedge'],
            ordered_sets['fbedge'], ordered_sets['bbedge'],
            ordered_sets['fledge'], ordered_sets['bledge'],
            ordered_sets['bredge'], ordered_sets['fredge'],
            ordered_sets['ltedge'], ordered_sets['rtedge'],
            ordered_sets['rbedge'], ordered_sets['lbedge'],
            corner_labels=ordered_sets,
        )
    elif condition_ch in (5, 6):
        function = (create_mindlin_nonhomogeneous_periodic_constraints_equation
                    if condition_ch == 5 else
                    create_mindlin_nonhomogeneous_periodic_constraints_equation_6dof)
        function(
            assembly, model_name, instance_nodes,
            ordered_sets['fronts'], ordered_sets['backs'],
            ordered_sets['tops'], ordered_sets['bots'],
            ordered_sets['frontbc'], ordered_sets['backbc'],
            ordered_sets['topbc'], ordered_sets['botbc'],
            mode, us1minus, fs1minus, L=L, H=H,
            ftedge=ordered_sets['ftedge'], btedge=ordered_sets['btedge'],
            fbedge=ordered_sets['fbedge'], bbedge=ordered_sets['bbedge'],
            fledge=ordered_sets['fledge'], bledge=ordered_sets['bledge'],
            bredge=ordered_sets['bredge'], fredge=ordered_sets['fredge'],
            ltedge=ordered_sets['ltedge'], rtedge=ordered_sets['rtedge'],
            rbedge=ordered_sets['rbedge'], lbedge=ordered_sets['lbedge'],
            corner_labels=ordered_sets,
        )
    else:
        create_only_sets()

    mass = assembly.getMassProperties().get('mass')
    print('NIAH PBC creation completed in %.3f s.' % (time.time() - start_time))
    return mass, L, H, W, error


def create_only_sets():
    print('NIAH PBC: boundary sets created without constraint equations.')


# Compatibility helpers for scripts written against earlier NIAH releases.
# New code should call periodic_mesh.classify_boundary_nodes and pair_regions.


def create_reference_point(assembly, x, y, z):
    """Create one assembly reference point."""
    assembly.ReferencePoint(point=(x, y, z))


def create_bcxyz_point(node, coord_index, boundary_value, mesh_tol, out_dict):
    """Collect a node that lies on one specified coordinate plane."""
    if abs(node.coordinates[coord_index] - boundary_value) <= mesh_tol:
        out_dict[int(node.label)] = tuple(node.coordinates[:3])


def create_corner_point(node, x_value, y_value, z_value, mesh_tol,
                        corner_list):
    """Collect a node at one specified cell corner."""
    targets = (x_value, y_value, z_value)
    if all(abs(node.coordinates[index] - targets[index]) <= mesh_tol
           for index in range(3)):
        corner_list.append(int(node.label))


def _create_axis_edge_point(node, fixed_axes, free_axis, endpoints,
                            mesh_tol, out_dict):
    if not all(abs(node.coordinates[index] - value) <= mesh_tol
               for index, value in fixed_axes):
        return
    endpoint_a, endpoint_b = endpoints
    free_value = node.coordinates[free_axis]
    if (abs(free_value - endpoint_a) > mesh_tol and
            abs(free_value - endpoint_b) > mesh_tol):
        out_dict[int(node.label)] = tuple(node.coordinates[:3])


def create_edge_point_z(node, x_value, y_value, z_hi, z_lo,
                        mesh_tol, out_dict):
    _create_axis_edge_point(
        node, ((0, x_value), (1, y_value)), 2, (z_lo, z_hi),
        mesh_tol, out_dict,
    )


def create_edge_point_y(node, x_value, z_value, y_hi, y_lo,
                        mesh_tol, out_dict):
    _create_axis_edge_point(
        node, ((0, x_value), (2, z_value)), 1, (y_lo, y_hi),
        mesh_tol, out_dict,
    )


def create_edge_point_x(node, z_value, y_value, x_hi, x_lo,
                        mesh_tol, out_dict):
    _create_axis_edge_point(
        node, ((2, z_value), (1, y_value)), 0, (x_lo, x_hi),
        mesh_tol, out_dict,
    )


def _create_face_interior_point(node, fixed_axis, fixed_value,
                                excluded_axes, mesh_tol, out_dict):
    if abs(node.coordinates[fixed_axis] - fixed_value) > mesh_tol:
        return
    for axis, endpoint_a, endpoint_b in excluded_axes:
        coordinate = node.coordinates[axis]
        if (abs(coordinate - endpoint_a) <= mesh_tol or
                abs(coordinate - endpoint_b) <= mesh_tol):
            return
    out_dict[int(node.label)] = tuple(node.coordinates[:3])


def create_face_point_fb(node, x_value, y_hi, y_lo, z_hi, z_lo,
                         mesh_tol, out_dict):
    _create_face_interior_point(
        node, 0, x_value, ((1, y_lo, y_hi), (2, z_lo, z_hi)),
        mesh_tol, out_dict,
    )


def create_face_point_lr(node, z_value, y_hi, y_lo, x_hi, x_lo,
                         mesh_tol, out_dict):
    _create_face_interior_point(
        node, 2, z_value, ((1, y_lo, y_hi), (0, x_lo, x_hi)),
        mesh_tol, out_dict,
    )


def create_face_point_tb(node, y_value, x_hi, x_lo, z_hi, z_lo,
                         mesh_tol, out_dict):
    _create_face_interior_point(
        node, 1, y_value, ((0, x_lo, x_hi), (2, z_lo, z_hi)),
        mesh_tol, out_dict,
    )


def check_point_correspondence(
        error, frontsxyz, backsxyz, topsxyz, botsxyz, leftsxyz,
        rightsxyz, ftedgexyz, btedgexyz, fbedgexyz, bbedgexyz,
        fledgexyz, bledgexyz, bredgexyz, fredgexyz, ltedgexyz,
        rtedgexyz, rbedgexyz, lbedgexyz, frontbcxyz, backbcxyz,
        topbcxyz, botbcxyz, leftbcxyz, rightbcxyz):
    """Validate legacy region-count inputs with the new matching model."""
    names = (
        'fronts', 'backs', 'tops', 'bots', 'lefts', 'rights',
        'ftedge', 'btedge', 'fbedge', 'bbedge',
        'fledge', 'bledge', 'bredge', 'fredge',
        'ltedge', 'rtedge', 'rbedge', 'lbedge',
        'frontbc', 'backbc', 'topbc', 'botbc', 'leftbc', 'rightbc',
    )
    values = (
        frontsxyz, backsxyz, topsxyz, botsxyz, leftsxyz, rightsxyz,
        ftedgexyz, btedgexyz, fbedgexyz, bbedgexyz,
        fledgexyz, bledgexyz, bredgexyz, fredgexyz,
        ltedgexyz, rtedgexyz, rbedgexyz, lbedgexyz,
        frontbcxyz, backbcxyz, topbcxyz, botbcxyz,
        leftbcxyz, rightbcxyz,
    )
    regions = dict(zip(names, values))
    pair_families = (
        ('fronts', 'backs'), ('tops', 'bots'), ('lefts', 'rights'),
        ('frontbc', 'backbc'), ('topbc', 'botbc'),
        ('leftbc', 'rightbc'),
        ('ftedge', 'btedge', 'bbedge', 'fbedge'),
        ('fledge', 'bledge', 'bredge', 'fredge'),
        ('ltedge', 'rtedge', 'rbedge', 'lbedge'),
    )
    for family in pair_families:
        counts = [len(regions[name]) for name in family]
        if min(counts) != max(counts):
            return True
    return bool(error)


def _replace_list(target, values):
    del target[:]
    target.extend(values)


def check_correspondence_transform_lable(
        error, errorset, meshsens, frontsxyz, backsxyz, topsxyz,
        botsxyz, leftsxyz, rightsxyz, ftedgexyz, btedgexyz,
        fbedgexyz, bbedgexyz, fledgexyz, bledgexyz, bredgexyz,
        fredgexyz, ltedgexyz, rtedgexyz, rbedgexyz, lbedgexyz,
        frontbcxyz, backbcxyz, topbcxyz, botbcxyz, leftbcxyz,
        rightbcxyz, fronts, backs, tops, bots, lefts, rights,
        ftedge, btedge, fbedge, bbedge, fledge, bledge, bredge,
        fredge, ltedge, rtedge, rbedge, lbedge, frontbc, backbc,
        topbc, botbc, leftbc, rightbc):
    """Populate legacy output lists using deterministic coordinate matching."""
    face_specs = (
        (frontsxyz, backsxyz, (1, 2), fronts, backs, 'fronts', 'backs'),
        (topsxyz, botsxyz, (0, 2), tops, bots, 'tops', 'bots'),
        (leftsxyz, rightsxyz, (0, 1), lefts, rights, 'lefts', 'rights'),
        (frontbcxyz, backbcxyz, (1, 2), frontbc, backbc,
         'frontbc', 'backbc'),
        (topbcxyz, botbcxyz, (0, 2), topbc, botbc, 'topbc', 'botbc'),
        (leftbcxyz, rightbcxyz, (0, 1), leftbc, rightbc,
         'leftbc', 'rightbc'),
    )
    edge_specs = (
        ((ftedgexyz, btedgexyz, bbedgexyz, fbedgexyz),
         (ftedge, btedge, bbedge, fbedge),
         ('ftedge', 'btedge', 'bbedge', 'fbedge'), (2,)),
        ((fledgexyz, bledgexyz, bredgexyz, fredgexyz),
         (fledge, bledge, bredge, fredge),
         ('fledge', 'bledge', 'bredge', 'fredge'), (1,)),
        ((ltedgexyz, rtedgexyz, rbedgexyz, lbedgexyz),
         (ltedge, rtedge, rbedge, lbedge),
         ('ltedge', 'rtedge', 'rbedge', 'lbedge'), (0,)),
    )
    try:
        for spec in face_specs:
            source, target, indices, output_a, output_b, name_a, name_b = spec
            labels_a, labels_b = pair_regions(
                source, target, indices, meshsens, name_a, name_b
            )
            _replace_list(output_a, labels_a)
            _replace_list(output_b, labels_b)
        for region_group, output_group, names, indices in edge_specs:
            reference_labels = []
            for index in range(1, len(region_group)):
                labels_a, labels_b = pair_regions(
                    region_group[0], region_group[index], indices, meshsens,
                    names[0], names[index],
                )
                reference_labels = labels_a
                _replace_list(output_group[index], labels_b)
            _replace_list(output_group[0], reference_labels)
    except PeriodicMeshError as exc:
        print('NIAH PBC correspondence failed: %s' % exc)
        labels = set(errorset)
        for region in (
                frontsxyz, backsxyz, topsxyz, botsxyz, leftsxyz, rightsxyz,
                ftedgexyz, btedgexyz, fbedgexyz, bbedgexyz,
                fledgexyz, bledgexyz, bredgexyz, fredgexyz,
                ltedgexyz, rtedgexyz, rbedgexyz, lbedgexyz):
            labels.update(int(label) for label in region)
        _replace_list(errorset, sorted(labels))
        return True
    return bool(error)
