#!/usr/bin/env python3
"""Convert an Axis-Studio BVH motion clip to a SONIC-compatible SMPL pkl.

Pipeline:
  1. Load BVH via general_motion_retargeting.utils.lafan1.load_bvh_file
     (drops the Axis calibration T-pose frame 0 by default).
  2. Extract world-space 3D joint positions for the 22 body joints that SMPL
     and SMPL-X share. End-site "foot tip" positions (SMPL 10 / 11) are
     down-weighted to zero during fitting so the SMPL-X skeleton handles them.
  3. Fit SMPL-X parameters (global_orient, body_pose, transl, shared betas)
     to those 3D targets via batch Adam (general_motion_retargeting.bvh_to_smpl.fitter).
  4. Save in SONIC's smpl_filtered schema: pose_aa (T, 72 axis-angle),
     transl (T, 3), smpl_joints (T, 24, 3), fps, original_fps, original_pose_aa.

Output schema mirrors data/smpl_filtered/<motion>.pkl in GR00T-WholeBodyControl.
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np

from general_motion_retargeting.bvh_to_smpl import (
    AXIS_BVH_TO_SMPL,
    SMPL_NUM_JOINTS,
    compute_axis_bvh_end_site_world,
)
from general_motion_retargeting.bvh_to_smpl.fitter import BatchSmplxFitter, FitterConfig
from general_motion_retargeting.utils.lafan1 import load_bvh_file


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--bvh_file", type=Path, required=True)
    p.add_argument("--format", type=str, default="axis",
                   choices=["axis", "lafan1", "nokov", "mixamo", "3dsmax"])
    p.add_argument("--human_height", type=float, default=None,
                   help="Override the subject height in metres for BVH scaling.")
    p.add_argument("--smpl_model_dir", type=Path,
                   default=Path(__file__).resolve().parent.parent / "assets" / "models",
                   help="Directory containing models/smplx/SMPLX_{NEUTRAL,MALE,FEMALE}.pkl")
    p.add_argument("--gender", type=str, default="neutral", choices=["neutral", "male", "female"])
    p.add_argument("--save_path", type=Path, required=True, help="Output .pkl path (SONIC schema).")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--n_iters_init", type=int, default=200)
    p.add_argument("--n_iters_refine", type=int, default=400)
    p.add_argument("--target_fps", type=float, default=50.0,
                   help="Output fps (SONIC standard = 50). Set equal to input BVH fps to skip resampling.")
    p.add_argument("--keep_axis_calibration_frame", action="store_true",
                   help="Do NOT drop BVH frame 0 (by default it's dropped for axis format).")
    return p.parse_args()


def extract_gt_joints(
    frames: list[dict],
    bvh_path: Path,
    drop_calibration_frame: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Build (T, 22, 3) array of 3D joint positions and per-joint weights.

    Uses lafan1's per-frame world joint positions for regular joints, and a
    dedicated BVH End-Site parser for the two foot-tip entries (SMPL 10 / 11).

    Returns:
        gt_joints: (T, 22, 3) float32 — positions for SMPL joints 0..21 (L_hand/R_hand skipped).
        weights:   (22,) float32 — 1 where we have a real target, 0 only if we had to
                   fall back to an ankle position (end-site parse failed).
    """
    T = len(frames)
    mapping = AXIS_BVH_TO_SMPL[:22]
    gt = np.zeros((T, 22, 3), dtype=np.float32)
    weights = np.ones(22, dtype=np.float32)

    end_parents = sorted({bone for (kind, bone) in mapping if kind == "end"})
    end_sites = compute_axis_bvh_end_site_world(
        bvh_path, end_parents, drop_calibration_frame=drop_calibration_frame,
    )

    for smpl_idx, (kind, bone) in enumerate(mapping):
        if kind == "joint":
            for t, frame in enumerate(frames):
                if bone not in frame:
                    raise KeyError(
                        f"BVH frame is missing joint '{bone}' (needed for SMPL index {smpl_idx})."
                    )
                gt[t, smpl_idx] = frame[bone][0]
        elif kind == "end":
            arr = end_sites.get(bone)
            if arr is None or arr.shape[0] != T:
                # End Site parse failed. Fall back to the parent joint and zero the weight
                # so this joint does not pollute the fit.
                print(f"[extract_gt_joints] end-site for {bone!r} unavailable "
                      f"(got shape {None if arr is None else arr.shape}); using ankle fallback "
                      f"with weight 0 for SMPL index {smpl_idx}.")
                for t, frame in enumerate(frames):
                    gt[t, smpl_idx] = frame[bone][0]
                weights[smpl_idx] = 0.0
            else:
                gt[:, smpl_idx] = arr
        else:
            raise ValueError(f"unknown source kind {kind!r}")
    return gt, weights


def pack_pose_aa_72(global_orient: np.ndarray, body_pose: np.ndarray) -> np.ndarray:
    """Build SONIC's (T, 72) pose_aa layout.

    SMPL pose layout (24 joints * 3):
      [0:3]   global_orient (pelvis root rotation)
      [3:66]  body_pose     (21 joints: L_hip..R_wrist)
      [66:72] hand slots    (L_hand, R_hand) — left as zeros; SONIC data fills these
                              identically for body-only motions.
    """
    T = global_orient.shape[0]
    pose = np.zeros((T, 72), dtype=np.float32)
    pose[:, 0:3] = global_orient
    pose[:, 3:66] = body_pose
    return pose


def maybe_resample(arr: np.ndarray, src_fps: float, tgt_fps: float) -> np.ndarray:
    """Nearest-index resampling to match SONIC's loader behaviour."""
    if abs(src_fps - tgt_fps) < 1e-6:
        return arr
    step = src_fps / tgt_fps
    idxs = np.arange(0, arr.shape[0], step).astype(int)
    idxs = idxs[idxs < arr.shape[0]]
    return arr[idxs]


def main() -> None:
    args = parse_args()
    args.save_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[bvh_to_smpl] loading {args.bvh_file}  format={args.format}")
    frames, _height, bvh_fps = load_bvh_file(
        str(args.bvh_file),
        format=args.format,
        human_height_override=args.human_height,
        drop_calibration_frame=not args.keep_axis_calibration_frame,
    )
    print(f"  -> {len(frames)} frames @ {bvh_fps} fps")

    gt_joints, weights = extract_gt_joints(
        frames, bvh_path=args.bvh_file,
        drop_calibration_frame=not args.keep_axis_calibration_frame,
    )
    print(f"  gt_joints: {gt_joints.shape}  dtype={gt_joints.dtype}")
    print(f"  fit-joint weights: 1={int((weights==1).sum())}  0={int((weights==0).sum())}")

    cfg = FitterConfig(
        smpl_model_dir=str(args.smpl_model_dir),
        gender=args.gender,
        device=args.device,
    )
    fitter = BatchSmplxFitter(cfg).with_joint_weights(weights)
    result = fitter.fit_clip(gt_joints, fps=float(bvh_fps))

    pose_aa = pack_pose_aa_72(result["global_orient"], result["body_pose"])
    transl = result["transl"]
    smpl_joints = result["smpl_joints"]
    original_fps = float(bvh_fps)
    pose_aa_out = maybe_resample(pose_aa, original_fps, args.target_fps)
    transl_out = maybe_resample(transl, original_fps, args.target_fps)
    joints_out = maybe_resample(smpl_joints, original_fps, args.target_fps)

    out = {
        "fps": float(args.target_fps),
        "original_fps": original_fps,
        "pose_aa": pose_aa_out.astype(np.float32),
        "transl": transl_out.astype(np.float32),
        "smpl_joints": joints_out.astype(np.float32),
        "original_pose_aa": pose_aa.astype(np.float32),
    }

    with args.save_path.open("wb") as f:
        pickle.dump(out, f)
    print(f"[bvh_to_smpl] saved {args.save_path}")
    print(f"  pose_aa={out['pose_aa'].shape}  transl={out['transl'].shape}"
          f"  smpl_joints={out['smpl_joints'].shape}  fps={out['fps']}")


if __name__ == "__main__":
    main()
