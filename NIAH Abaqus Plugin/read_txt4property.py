import numpy as np
import re


def read_stiffness_txt(file_path, stru_type):
    """
    Read the txt file of the shell/3D homogenization results and return the various stiffness values.
    It is allowed for some data to be absent; when absent, None should be returned accordingly.

    Parameters
    ----------
    file_path : str
        txt file path
    stru_type : str
        '3D' or 'shell'

    Returns
    -------
    dict
        {
            'EH': ndarray or None,
            'A': ndarray or None,
            'B': ndarray or None,
            'D': ndarray or None,
            'Kxz': float or None,
            'Kyz': float or None
        }
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    lines = text.splitlines()

    # initialization
    EH = None
    A = None
    B = None
    D = None
    Kxz = None
    Kyz = None

    def extract_matrix(lines, header, nrows, ncols, required=False):
        start_idx = None
        for i, line in enumerate(lines):
            if header in line:
                start_idx = i + 1
                break

        if start_idx is None:
            if required:
                raise ValueError(f"Matrix title not found: {header}")
            return None

        data = []
        for line in lines[start_idx:]:
            s = line.strip()

            # Skip blank lines and separators
            if not s or s.startswith('===='):
                if len(data) >= nrows:
                    break
                continue

            # If a new title is encountered, stop
            if ':' in s and len(data) > 0:
                break

            nums = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', s)
            if nums:
                row = [float(x) for x in nums]
                if len(row) != ncols:
                    if required:
                        raise ValueError(
                            "A row in {} does not contain {} columns: {}".format(
                                header, ncols, line
                            )
                        )
                    return None
                data.append(row)

            if len(data) == nrows:
                break

        if len(data) != nrows:
            if required:
                raise ValueError(
                    "Failed to read {}: expected {} rows, found {}.".format(
                        header, nrows, len(data)
                    )
                )
            return None

        return np.array(data, dtype=float)

    def extract_scalar(text, key, required=False):
        pattern = rf'{re.escape(key)}\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)'
        match = re.search(pattern, text)
        if not match:
            if required:
                raise ValueError(f"Find no scalar: {key}")
            return None
        return float(match.group(1))

    def extract_matrix_by_headers(lines, headers, nrows, ncols, required=False):
        for header in headers:
            arr = extract_matrix(lines, header, nrows, ncols, required=False)
            if arr is not None:
                return arr
        if required:
            raise ValueError(f"Matrix title not found: {headers}")
        return None

    stru_type = stru_type.strip().lower()
    if stru_type == '3d':
        EH = extract_matrix_by_headers(
            lines,
            ['EH (6x6 plate stiffness matrix):', 'EH (6x6 stiffness matrix):', 'EH:'],
            6, 6, required=True
        )

    elif stru_type == 'shell':
        EH = extract_matrix_by_headers(
            lines,
            ['EH (6x6 plate stiffness matrix):', 'EH (6x6 stiffness matrix):', 'EH:'],
            6, 6, required=True
        )
        A = extract_matrix(lines, 'A (3x3 extensional stiffness):', 3, 3, required=True)
        B = extract_matrix(lines, 'B (3x3 coupling stiffness):', 3, 3, required=True)
        D = extract_matrix(lines, 'D (3x3 bending stiffness):', 3, 3, required=True)
        Kxz = extract_scalar(text, 'Kxz', required=False)
        Kyz = extract_scalar(text, 'Kyz', required=False)

    else:
        raise ValueError("stru_type must be either '3D' or 'shell'.")

    return {
        'EH': EH,
        'A': A,
        'B': B,
        'D': D,
        'Kxz': Kxz,
        'Kyz': Kyz
    }
