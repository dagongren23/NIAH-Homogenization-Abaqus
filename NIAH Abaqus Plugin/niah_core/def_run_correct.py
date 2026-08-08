"""Archive generated Abaqus job files without irreversibly deleting them."""

import os
import shutil
import time


_GENERATED_EXTENSIONS = (
    '.dat', '.ipm', '.jnl', '.log', '.com', '.msg', '.prt', '.sim', '.sta',
    '.stt', '.res', '.abq', '.pac', '.sel', '.mdl', '.sat', '.lck', '.dmp',
    '.rpy', '.rec', '.7', '.inp'
)


def _unique_destination(path):
    """Return a non-existing destination path without overwriting an archive."""
    if not os.path.exists(path):
        return path
    index = 1
    while os.path.exists('%s.%d' % (path, index)):
        index += 1
    return '%s.%d' % (path, index)


def _archive_file(file_path, source_root, archive_root):
    relative_path = os.path.relpath(file_path, source_root)
    destination = _unique_destination(os.path.join(archive_root, relative_path))
    destination_dir = os.path.dirname(destination)
    if not os.path.isdir(destination_dir):
        os.makedirs(destination_dir)
    shutil.move(file_path, destination)
    return destination


def run_correct(choose, runtime_input):
    """Move generated files into a recoverable ``delete`` archive.

    ``choose == 1`` searches the workbench recursively, whereas ``choose == 2``
    processes only files located directly in the workbench. ODB files are
    intentionally retained in place, matching the historical cleanup scope.
    """
    directory = os.path.abspath(runtime_input['workbench_path'])
    if not os.path.isdir(directory):
        raise ValueError('Workbench directory does not exist: %s' % directory)

    plugin_root = os.path.abspath(runtime_input.get(
        'plugin_root',
        os.path.join(os.path.dirname(__file__), os.pardir)
    ))
    archive_root = os.path.join(
        plugin_root,
        'delete',
        'run_outputs',
        'niah_generated_%s' % time.strftime('%Y%m%d_%H%M%S')
    )

    extensions = _GENERATED_EXTENSIONS
    if runtime_input.get('resume_solver'):
        extensions = tuple(
            extension for extension in extensions
            if extension not in ('.inp', '.sta')
        )

    if choose == 1:
        candidates = []
        delete_root = os.path.normcase(os.path.join(directory, 'delete'))
        for root, dirs, files in os.walk(directory):
            dirs[:] = [
                name for name in dirs
                if os.path.normcase(os.path.join(root, name)) != delete_root
            ]
            for file_name in files:
                if file_name.lower().endswith(extensions):
                    candidates.append(os.path.join(root, file_name))
    elif choose == 2:
        candidates = [
            os.path.join(directory, file_name)
            for file_name in os.listdir(directory)
            if os.path.isfile(os.path.join(directory, file_name))
            and file_name.lower().endswith(extensions)
        ]
    else:
        raise ValueError('choose must be 1 (recursive) or 2 (top-level only).')

    archived = []
    for file_path in sorted(candidates):
        archived.append(_archive_file(file_path, directory, archive_root))

    if archived:
        print('Archived %d generated files under %s.' % (len(archived), archive_root))
    return tuple(archived)
