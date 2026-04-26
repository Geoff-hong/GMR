"""Axis Studio / Noitom BVH joint names -> SMPL 24-joint correspondence.

SMPL-X joints 0-21 coincide with SMPL joints 0-21 (same names, same order).
SMPL joints 22 / 23 (L_hand / R_hand) have no direct SMPL-X body-joint equivalent;
we approximate them with the hand end-site positions in BVH, used only as
fitting reference — the output pose_aa hand slots are left at zero.

Mapping adapted from gear_sonic/scripts/blanket_sonic_utils.py SMPL_POSITION_SOURCES
so that the new fitter consumes the same BVH semantics as the previous pipeline.
"""
from __future__ import annotations

SMPL_NUM_JOINTS = 24
SMPLX_NUM_BODY_JOINTS_FIT = 22  # we fit against indices 0..21 only

# Per SMPL 24-joint index, (source_kind, axis_bone_name):
#   ("joint",  <bone>) -> use FK world position of that bone
#   ("end",    <bone>) -> use the bone's end-site position (BVH offset at terminal)
AXIS_BVH_TO_SMPL: list[tuple[str, str]] = [
    ("joint", "Hips"),          # 0  pelvis
    ("joint", "LeftUpLeg"),     # 1  L_hip
    ("joint", "RightUpLeg"),    # 2  R_hip
    ("joint", "Spine"),         # 3  spine1
    ("joint", "LeftLeg"),       # 4  L_knee
    ("joint", "RightLeg"),      # 5  R_knee
    ("joint", "Spine1"),        # 6  spine2
    ("joint", "LeftFoot"),      # 7  L_ankle
    ("joint", "RightFoot"),     # 8  R_ankle
    ("joint", "Spine2"),        # 9  spine3
    ("end",   "LeftFoot"),      # 10 L_foot (toe end-site)
    ("end",   "RightFoot"),     # 11 R_foot
    ("joint", "Neck"),          # 12 neck
    ("joint", "LeftShoulder"),  # 13 L_collar
    ("joint", "RightShoulder"), # 14 R_collar
    ("joint", "Head"),          # 15 head
    ("joint", "LeftArm"),       # 16 L_shoulder
    ("joint", "RightArm"),      # 17 R_shoulder
    ("joint", "LeftForeArm"),   # 18 L_elbow
    ("joint", "RightForeArm"),  # 19 R_elbow
    ("joint", "LeftHand"),      # 20 L_wrist
    ("joint", "RightHand"),     # 21 R_wrist
    ("end",   "LeftHand"),      # 22 L_hand (thumb-tip proxy, not fit)
    ("end",   "RightHand"),     # 23 R_hand (thumb-tip proxy, not fit)
]

SMPL_PARENTS: list[int] = [
    -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8,
    9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21,
]

SMPL_JOINT_NAMES: list[str] = [
    "pelvis", "L_hip", "R_hip", "spine1", "L_knee", "R_knee", "spine2",
    "L_ankle", "R_ankle", "spine3", "L_foot", "R_foot", "neck",
    "L_collar", "R_collar", "head", "L_shoulder", "R_shoulder",
    "L_elbow", "R_elbow", "L_wrist", "R_wrist", "L_hand", "R_hand",
]
