"""Extract BVH End-Site world positions for fitting SMPL toe-tip / finger-tip joints.

The lafan_vendor BVH loader discards End Site offsets, so we re-parse the raw
BVH text for the specific parent bones we need, then run forward kinematics
(using lafan_vendor's own quat_fk) and apply the same Y-up -> Z-up rotation +
cm -> m scaling that lafan1.load_bvh_file applies to joint positions, so the
outputs are in the same world frame as the joint positions the fitter consumes.

Typical use:
    >>> from general_motion_retargeting.bvh_to_smpl.end_sites import compute_axis_bvh_end_site_world
    >>> positions = compute_axis_bvh_end_site_world(
    ...     bvh_path, parent_names=["LeftFoot", "RightFoot"],
    ...     drop_calibration_frame=True,
    ... )
    >>> positions["LeftFoot"].shape   # -> (T, 3) in the same frame as lafan1's joint positions
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

from general_motion_retargeting.utils.lafan_vendor.extract import read_bvh
from general_motion_retargeting.utils.lafan_vendor import utils as _lv_utils


# Same rotation as general_motion_retargeting.utils.lafan1.load_bvh_file (Y-up -> Z-up).
_YUP_TO_ZUP = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)


def parse_end_site_offsets(bvh_text: str) -> dict[str, np.ndarray]:
    """Return a dict mapping parent-bone name -> End Site local offset (Y-up, cm)."""
    offsets: dict[str, np.ndarray] = {}
    # Tokenize just what we need
    pattern = re.compile(
        r"\{|\}|(?:ROOT|JOINT)\s+\S+|End Site|OFFSET\s+[-\d.eE+]+\s+[-\d.eE+]+\s+[-\d.eE+]+"
    )
    stack: list[tuple[str, str | None]] = []  # entries: (kind, name) where kind in {"joint","end"}
    pending: tuple[str, str | None] | None = None
    for tok in pattern.findall(bvh_text):
        tok = tok.strip()
        if tok.startswith("ROOT") or tok.startswith("JOINT"):
            parts = tok.split()
            pending = ("joint", parts[1])
        elif tok == "End Site":
            parent = stack[-1][1] if stack else None
            pending = ("end", parent)
        elif tok == "{":
            if pending is None:
                stack.append(("unknown", None))
            else:
                stack.append(pending)
                pending = None
        elif tok == "}":
            if stack:
                stack.pop()
        elif tok.startswith("OFFSET"):
            if stack and stack[-1][0] == "end":
                nums = tok.split()[1:4]
                offsets[stack[-1][1]] = np.array([float(x) for x in nums], dtype=np.float64)
    return offsets


def compute_axis_bvh_end_site_world(
    bvh_path: str | Path,
    parent_names: list[str],
    drop_calibration_frame: bool = True,
) -> dict[str, np.ndarray]:
    """Compute world-frame positions (Z-up, metres) of End Sites attached to given parent bones.

    The output coordinate frame matches the joint positions returned by
    general_motion_retargeting.utils.lafan1.load_bvh_file so that the end-site
    positions can be mixed directly into a (T, N, 3) gt_joints array for fitting.
    """
    text = Path(bvh_path).read_text(encoding="utf-8")
    offsets_yup_cm = parse_end_site_offsets(text)

    data = read_bvh(str(bvh_path))
    # world FK in BVH's native Y-up frame
    global_quat_wxyz, global_pos_yup_cm = _lv_utils.quat_fk(
        data.quats, data.pos, data.parents
    )
    name_to_idx = {name: i for i, name in enumerate(data.bones)}

    result: dict[str, np.ndarray] = {}
    for parent in parent_names:
        if parent not in offsets_yup_cm:
            continue
        if parent not in name_to_idx:
            continue
        bi = name_to_idx[parent]
        # lafan_vendor stores quats as (w, x, y, z); scipy expects (x, y, z, w).
        pq_xyzw = global_quat_wxyz[:, bi][:, [1, 2, 3, 0]]
        rot_mat = R.from_quat(pq_xyzw).as_matrix()                        # (T, 3, 3)
        world_offset = np.einsum("tij,j->ti", rot_mat, offsets_yup_cm[parent])
        end_yup_cm = global_pos_yup_cm[:, bi] + world_offset              # (T, 3) in Y-up cm
        end_zup_m = (end_yup_cm @ _YUP_TO_ZUP.T) / 100.0                  # match lafan1's framing
        if drop_calibration_frame and end_zup_m.shape[0] > 1:
            end_zup_m = end_zup_m[1:]
        result[parent] = end_zup_m.astype(np.float32)
    return result
