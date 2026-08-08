"""Export nodal displacement fields from Abaqus ODB files to CSV.

This script is intentionally compatible with the Python 2.7 interpreter shipped
with Abaqus 2020.  It performs no interpolation and creates no figures.  Run it
with ``abaqus python``; analyse the exported CSV files with
``compare_static_displacement_fields.py`` under a normal scientific Python
environment.

Examples
--------
Export the four mixed-cell validation ODB files::

    abaqus python extract_odb_displacement.py --odb-dir E:\\abaqus2020\\temp

Export one ODB explicitly::

    abaqus python extract_odb_displacement.py --odb model.odb --output model.csv
"""

from __future__ import print_function

import argparse
import csv
import json
import math
import os
import sys


BATCH_CASES = (
    ("shell_homo", "Job-shell-static.odb"),
    ("shell_fine", "Lattice_Hybrid_shell_static.odb"),
    ("solid_homo", "Job-3D-static.odb"),
    ("solid_fine", "Lattice_Hybrid_model_static.odb"),
)


def _repository_keys(repository):
    """Return Abaqus repository keys as a plain list in repository order."""
    return list(repository.keys())


def _select_step(odb, requested_name):
    keys = _repository_keys(odb.steps)
    if not keys:
        raise RuntimeError("ODB contains no analysis steps")
    if requested_name:
        if requested_name not in odb.steps:
            raise RuntimeError(
                "Step %r was not found; available steps: %s"
                % (requested_name, ", ".join(keys))
            )
        return requested_name, odb.steps[requested_name]
    name = keys[-1]
    return name, odb.steps[name]


def _select_frame(step, frame_index):
    frame_count = len(step.frames)
    if frame_count == 0:
        raise RuntimeError("Selected step contains no frames")
    index = frame_index
    if index < 0:
        index = frame_count + index
    if index < 0 or index >= frame_count:
        raise RuntimeError(
            "Frame index %d is outside the available range 0..%d"
            % (frame_index, frame_count - 1)
        )
    return index, step.frames[index]


def _select_instance(odb, requested_name):
    instances = odb.rootAssembly.instances
    keys = _repository_keys(instances)
    if not keys:
        raise RuntimeError("ODB root assembly contains no instances")
    if requested_name:
        requested_upper = requested_name.upper()
        matches = [name for name in keys if name.upper() == requested_upper]
        if not matches:
            raise RuntimeError(
                "Instance %r was not found; available instances: %s"
                % (requested_name, ", ".join(keys))
            )
        name = matches[0]
        return name, instances[name]

    # Rigid loading plates and reference-point instances may also be present.
    # The structural instance is selected deterministically by node count.
    name = max(keys, key=lambda key: len(instances[key].nodes))
    return name, instances[name]


def _pad3(values):
    data = list(values)
    while len(data) < 3:
        data.append(0.0)
    return float(data[0]), float(data[1]), float(data[2])


def _field_value_data(value):
    """Read either single- or double-precision Abaqus field data."""
    try:
        return value.data
    except Exception as single_precision_error:
        try:
            return value.dataDouble
        except Exception:
            raise single_precision_error


def _node_record(node):
    x, y, z = _pad3(node.coordinates)
    return int(node.label), x, y, z


def _spatially_sample_nodes(nodes, max_nodes):
    """Return deterministic, envelope-preserving node records.

    The implementation uses only Python's standard library so it works in
    Abaqus 2020/Python 2.7.  It scans the ODB node repository twice but retains
    only one node per occupied spatial bin, making million-node ODBs practical.
    """
    total_nodes = len(nodes)
    if max_nodes is None or max_nodes <= 0 or total_nodes <= max_nodes:
        return [_node_record(node) for node in nodes]
    if max_nodes < 8:
        raise ValueError("max_nodes must be at least 8 when sampling is enabled")

    lower = [float("inf")] * 3
    upper = [float("-inf")] * 3
    lower_records = [None] * 3
    upper_records = [None] * 3
    for node in nodes:
        record = _node_record(node)
        for direction in range(3):
            coordinate = record[direction + 1]
            if coordinate < lower[direction]:
                lower[direction] = coordinate
                lower_records[direction] = record
            if coordinate > upper[direction]:
                upper[direction] = coordinate
                upper_records[direction] = record

    span = [upper[index] - lower[index] for index in range(3)]
    tolerance = max(max(span), 1.0) * 1.0e-10
    active = [index for index in range(3) if span[index] > tolerance]
    if not active:
        return [lower_records[0]]

    extreme_records = {}
    for record in lower_records + upper_records:
        if record is not None:
            extreme_records[record[0]] = record
    bin_capacity = max(1, max_nodes - len(extreme_records))
    bins_per_axis = max(
        1, int(math.floor(bin_capacity ** (1.0 / float(len(active)))))
    )

    selected_bins = {}
    for node in nodes:
        record = _node_record(node)
        normalized = []
        key = []
        for direction in active:
            value = (record[direction + 1] - lower[direction]) / span[direction]
            normalized.append(value)
            bin_value = min(int(value * bins_per_axis), bins_per_axis - 1)
            key.append(bin_value)
        centre_distance2 = 0.0
        for value, bin_value in zip(normalized, key):
            centre = (bin_value + 0.5) / float(bins_per_axis)
            centre_distance2 += (value - centre) ** 2
        key = tuple(key)
        previous = selected_bins.get(key)
        if previous is None or centre_distance2 < previous[0]:
            selected_bins[key] = (centre_distance2, record)

    selected = dict(extreme_records)
    for _, record in selected_bins.values():
        selected[record[0]] = record
    return [selected[label] for label in sorted(selected)]


def _open_csv_for_writer(path):
    if sys.version_info[0] < 3:
        return open(path, "wb")
    return open(path, "w", newline="")


def export_one(odb_path, output_path, step_name=None, frame_index=-1,
               instance_name=None, max_nodes=None):
    """Export one structural instance and return traceability metadata."""
    try:
        from odbAccess import openOdb
        from abaqusConstants import NODAL
    except ImportError:
        raise RuntimeError(
            "Abaqus odbAccess is unavailable. Run this script with "
            "'abaqus python', not the system Python interpreter."
        )

    odb_path = os.path.abspath(odb_path)
    output_path = os.path.abspath(output_path)
    if not os.path.isfile(odb_path):
        raise IOError("ODB file does not exist: %s" % odb_path)
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    odb = None
    try:
        odb = openOdb(path=str(odb_path), readOnly=True)
        selected_step_name, step = _select_step(odb, step_name)
        selected_frame_index, frame = _select_frame(step, frame_index)
        selected_instance_name, instance = _select_instance(odb, instance_name)

        if "U" not in frame.fieldOutputs:
            raise RuntimeError(
                "Displacement field U is absent from step %s, frame %d"
                % (selected_step_name, selected_frame_index)
            )

        selected_nodes = _spatially_sample_nodes(instance.nodes, max_nodes)
        selected_labels = set(record[0] for record in selected_nodes)

        displacement = frame.fieldOutputs["U"].getSubset(
            region=instance, position=NODAL
        )
        displacement_by_label = {}
        for value in displacement.values:
            label = int(value.nodeLabel)
            if label in selected_labels:
                displacement_by_label[label] = _pad3(_field_value_data(value))

        rows = []
        missing_labels = []
        for label, x, y, z in selected_nodes:
            if label not in displacement_by_label:
                missing_labels.append(label)
                continue
            u1, u2, u3 = displacement_by_label[label]
            umag = math.sqrt(u1 * u1 + u2 * u2 + u3 * u3)
            rows.append(
                (selected_instance_name, label, x, y, z, u1, u2, u3, umag)
            )

        if not rows:
            raise RuntimeError(
                "No nodal displacements were found for instance %s"
                % selected_instance_name
            )

        with _open_csv_for_writer(output_path) as stream:
            writer = csv.writer(stream)
            writer.writerow(
                ("instance", "node_label", "x", "y", "z",
                 "u1", "u2", "u3", "umag")
            )
            writer.writerows(rows)

        coordinates = [row[2:5] for row in rows]
        metadata = {
            "odb_path": odb_path,
            "csv_path": output_path,
            "step": selected_step_name,
            "frame_index": selected_frame_index,
            "frame_value": float(frame.frameValue),
            "instance": selected_instance_name,
            "instance_node_count": int(len(instance.nodes)),
            "requested_max_nodes": (
                int(max_nodes) if max_nodes is not None and max_nodes > 0 else None
            ),
            "spatially_selected_node_count": int(len(selected_nodes)),
            "exported_node_count": int(len(rows)),
            "missing_displacement_count": int(len(missing_labels)),
            "coordinate_min": [min(row[i] for row in coordinates) for i in range(3)],
            "coordinate_max": [max(row[i] for row in coordinates) for i in range(3)],
        }
        metadata_path = os.path.splitext(output_path)[0] + "_metadata.json"
        with open(metadata_path, "w") as stream:
            json.dump(metadata, stream, indent=2, sort_keys=True)
        print(
            "Exported %d nodes from %s / %s / frame %d -> %s"
            % (len(rows), os.path.basename(odb_path), selected_step_name,
               selected_frame_index, output_path)
        )
        if missing_labels:
            print(
                "Warning: %d instance nodes had no nodal U value"
                % len(missing_labels)
            )
        return metadata
    finally:
        if odb is not None:
            odb.close()


def _build_parser():
    parser = argparse.ArgumentParser(
        description="Export Abaqus ODB nodal displacement fields to CSV."
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--odb", help="Path to one ODB file")
    mode.add_argument(
        "--odb-dir",
        help="Directory containing the four predefined mixed-cell ODB files",
    )
    parser.add_argument(
        "--output", help="Output CSV path; valid only together with --odb"
    )
    parser.add_argument(
        "--output-dir",
        help="Batch output directory (default: <odb-dir>/static_field_csv)",
    )
    parser.add_argument("--step", help="Step name (default: last step)")
    parser.add_argument(
        "--frame", type=int, default=-1, help="Frame index (default: -1, last)"
    )
    parser.add_argument(
        "--instance",
        help="Instance name (default: instance with the most nodes)",
    )
    parser.add_argument(
        "--max-nodes", type=int,
        help=(
            "Spatial node-export target. The exact model envelope is preserved; "
            "omit to export all structural-instance nodes."
        ),
    )
    return parser


def main(argv=None):
    args = _build_parser().parse_args(argv)
    if not args.odb and not args.odb_dir:
        raise SystemExit("Specify either --odb or --odb-dir")

    if args.odb:
        if not args.output:
            raise SystemExit("--output is required together with --odb")
        export_one(
            args.odb,
            args.output,
            step_name=args.step,
            frame_index=args.frame,
            instance_name=args.instance,
            max_nodes=args.max_nodes,
        )
        return 0

    odb_dir = os.path.abspath(args.odb_dir)
    output_dir = (
        os.path.abspath(args.output_dir)
        if args.output_dir
        else os.path.join(odb_dir, "static_field_csv")
    )
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    manifest = []
    for case_name, odb_filename in BATCH_CASES:
        manifest.append(
            export_one(
                os.path.join(odb_dir, odb_filename),
                os.path.join(output_dir, case_name + ".csv"),
                step_name=args.step,
                frame_index=args.frame,
                instance_name=args.instance,
                max_nodes=args.max_nodes,
            )
        )
    manifest_path = os.path.join(output_dir, "extraction_manifest.json")
    with open(manifest_path, "w") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
    print("Wrote extraction manifest: %s" % manifest_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
