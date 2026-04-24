"""BVH -> SMPL fitting pipeline.

Replaces the ad-hoc name-mapping conversion in
gear_sonic/scripts/blanket_sonic_utils.py::build_smpl_motion_data with a
proper PyTorch-based fit of SMPL-X parameters to BVH forward-kinematics
joint positions. Output matches SONIC's smpl_filtered schema.

Module-level imports are deliberately light; the heavy deps (smplx, torch)
live inside fitter.py so this package can be imported without them if only
the joint mapping / loss definitions are needed.
"""
from .joint_mapping import (
    AXIS_BVH_TO_SMPL,
    SMPL_NUM_JOINTS,
    SMPL_JOINT_NAMES,
    SMPL_PARENTS,
    SMPLX_NUM_BODY_JOINTS_FIT,
)
from .end_sites import compute_axis_bvh_end_site_world, parse_end_site_offsets

__all__ = [
    "AXIS_BVH_TO_SMPL",
    "SMPL_NUM_JOINTS",
    "SMPL_JOINT_NAMES",
    "SMPL_PARENTS",
    "SMPLX_NUM_BODY_JOINTS_FIT",
    "compute_axis_bvh_end_site_world",
    "parse_end_site_offsets",
]
