"""
Pose-error metrics used to report calibration accuracy.
"""

from typing import Dict, Sequence, Tuple

import numpy as np

from data_types.nominal_types import Frame


def frame_error(F_a: Frame, F_b: Frame) -> Tuple[float, float]:
    """
    TODO: implement.

    The mismatch between two frames is defined as
    dF = F_a^-1 F_b ~= Frame(I + sk(alpha), epsilon); report its size.

    input:
        F_a - reference Frame (e.g. ground truth)
        F_b - Frame to compare (e.g. your prediction)
    output:
        (angle_rad, dist_mm) - ||alpha|| (the rotation angle of dF.R) and
            ||epsilon|| (the norm of dF.p)
    """

    # relative error frame from reference to predicted frame
    dF = F_a.inv() * F_b

    R = dF.R.matrix
    p = np.array([dF.p.x, dF.p.y, dF.p.z])

    # rotation angle from the relative rotation matrix
    cos_angle = (np.trace(R) - 1) / 2
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle_rad = np.arccos(cos_angle)

    # magnitude of translational error
    dist_mm = np.linalg.norm(p)

    return float(angle_rad), float(dist_mm) 


def summarize_errors(errors: Sequence[Tuple[float, float]]) -> Dict[str, float]:
    """
    TODO: implement.

    input:
        errors - list of (angle_rad, dist_mm) pairs from frame_error
    output:
        dict with at least the keys
            n, rms_angle, max_angle, rms_pos, max_pos
    """

    if len(errors) == 0:
        raise ValueError("At least one error is required")

    angles = np.array([e[0] for e in errors])
    positions = np.array([e[1] for e in errors])

    return {
        "n": len(errors),
        "rms_angle": float(np.sqrt(np.mean(angles**2))),
        "max_angle": float(np.max(angles)),
        "rms_pos": float(np.sqrt(np.mean(positions**2))),
        "max_pos": float(np.max(positions)),
    }
