#!/usr/bin/env python3
"""Convert GMR robot PKLs to grounded 30 fps CSVs.

This is the canonical control-facing grounding implementation for GMR-generated
G1 motions. It replaces older global-minimum grounding logic with stance-aware
contact locking based on MuJoCo foot contact spheres.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import mujoco as mj
import numpy as np

from general_motion_retargeting import ROBOT_XML_DICT


_FOOT_BODY_NAMES = ("left_ankle_roll_link", "right_ankle_roll_link")
_CONTACT_SPHERE_MAX_RADIUS = 0.01
_STANCE_HEIGHT_MAX = 0.08
_STANCE_RELATIVE_HEIGHT_MAX = 0.035
_STANCE_XY_SPEED_MAX = 0.20
_STANCE_Z_SPEED_MAX = 0.35
_MIN_STANCE_FRAMES = 3
_FILL_STANCE_GAP_FRAMES = 4


def _foot_contact_geom_ids_by_side(model: mj.MjModel) -> dict[str, list[int]]:
    ids_by_side: dict[str, list[int]] = {}
    for name in _FOOT_BODY_NAMES:
        try:
            body_id = model.body(name).id
        except KeyError:
            continue
        ids = [
            geom_id
            for geom_id in range(model.ngeom)
            if model.geom_bodyid[geom_id] == body_id
            and model.geom_size[geom_id, 0] > 0
            and model.geom_size[geom_id, 0] <= _CONTACT_SPHERE_MAX_RADIUS
        ]
        if not ids:
            ids = [
                geom_id
                for geom_id in range(model.ngeom)
                if model.geom_bodyid[geom_id] == body_id
                and model.geom_size[geom_id, 0] > 0
            ]
        if ids:
            ids_by_side[name] = ids
    missing = [name for name in _FOOT_BODY_NAMES if name not in ids_by_side]
    if missing:
        raise RuntimeError(f"Could not find foot contact geoms for: {missing}")
    return ids_by_side


def _foot_contact_kinematics(
    model: mj.MjModel,
    data: mj.MjData,
    contact_ids_by_side: dict[str, list[int]],
    root_pos_all: np.ndarray,
    root_rot_xyzw_all: np.ndarray,
    dof_pos_all: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    frame_count = root_pos_all.shape[0]
    heights = np.zeros((frame_count, len(_FOOT_BODY_NAMES)), dtype=np.float64)
    positions = np.zeros((frame_count, len(_FOOT_BODY_NAMES), 3), dtype=np.float64)
    root_rot_wxyz_all = root_rot_xyzw_all[:, [3, 0, 1, 2]]

    for frame_idx in range(frame_count):
        data.qpos[:3] = root_pos_all[frame_idx]
        data.qpos[3:7] = root_rot_wxyz_all[frame_idx]
        data.qpos[7:] = dof_pos_all[frame_idx]
        mj.mj_forward(model, data)

        for side_idx, body_name in enumerate(_FOOT_BODY_NAMES):
            geom_ids = contact_ids_by_side[body_name]
            bottoms = np.asarray(
                [data.geom_xpos[geom_id].copy() for geom_id in geom_ids],
                dtype=np.float64,
            )
            bottoms[:, 2] -= np.asarray(
                [model.geom_size[geom_id, 0] for geom_id in geom_ids],
                dtype=np.float64,
            )
            heights[frame_idx, side_idx] = float(bottoms[:, 2].min())
            positions[frame_idx, side_idx] = bottoms.mean(axis=0)

    return heights, positions


def _finite_difference(values: np.ndarray, fps: float) -> np.ndarray:
    velocity = np.zeros_like(values, dtype=np.float64)
    if values.shape[0] <= 1:
        return velocity
    if values.shape[0] == 2:
        velocity[:] = (values[1] - values[0]) * fps
        return velocity
    velocity[1:-1] = (values[2:] - values[:-2]) * (0.5 * fps)
    velocity[0] = velocity[1]
    velocity[-1] = velocity[-2]
    return velocity


def _iter_segments(mask: np.ndarray, value: bool) -> list[tuple[int, int]]:
    segments: list[tuple[int, int]] = []
    start = None
    for idx, item in enumerate(mask):
        if bool(item) is value and start is None:
            start = idx
        elif bool(item) is not value and start is not None:
            segments.append((start, idx))
            start = None
    if start is not None:
        segments.append((start, len(mask)))
    return segments


def _fill_short_false_gaps(mask: np.ndarray, max_gap_frames: int) -> np.ndarray:
    output = mask.copy()
    for start, end in _iter_segments(output, False):
        if end - start > max_gap_frames:
            continue
        left_true = start == 0 or bool(output[start - 1])
        right_true = end == len(output) or bool(output[end])
        if left_true and right_true:
            output[start:end] = True
    return output


def _remove_short_true_islands(mask: np.ndarray, min_frames: int) -> np.ndarray:
    output = mask.copy()
    for start, end in _iter_segments(output, True):
        if end - start < min_frames:
            output[start:end] = False
    return output


def _infer_stance_mask(
    foot_heights: np.ndarray,
    foot_positions: np.ndarray,
    fps: float,
) -> tuple[np.ndarray, dict[str, object]]:
    normalized_heights = foot_heights - float(np.min(foot_heights))
    relative_heights = normalized_heights - np.min(normalized_heights, axis=1, keepdims=True)
    velocities = _finite_difference(foot_positions, fps)
    xy_speed = np.linalg.norm(velocities[..., :2], axis=-1)
    z_speed = np.abs(velocities[..., 2])
    stance = (
        (normalized_heights <= _STANCE_HEIGHT_MAX)
        & (relative_heights <= _STANCE_RELATIVE_HEIGHT_MAX)
        & (xy_speed <= _STANCE_XY_SPEED_MAX)
        & (z_speed <= _STANCE_Z_SPEED_MAX)
    )

    for side_idx in range(stance.shape[1]):
        stance[:, side_idx] = _fill_short_false_gaps(stance[:, side_idx], _FILL_STANCE_GAP_FRAMES)
        stance[:, side_idx] = _remove_short_true_islands(stance[:, side_idx], _MIN_STANCE_FRAMES)

    diagnostics = {
        "height_threshold_m": _STANCE_HEIGHT_MAX,
        "relative_height_threshold_m": _STANCE_RELATIVE_HEIGHT_MAX,
        "xy_speed_threshold_mps": _STANCE_XY_SPEED_MAX,
        "z_speed_threshold_mps": _STANCE_Z_SPEED_MAX,
        "left_stance_frames": int(np.count_nonzero(stance[:, 0])),
        "right_stance_frames": int(np.count_nonzero(stance[:, 1])),
        "any_stance_frames": int(np.count_nonzero(np.any(stance, axis=1))),
        "no_support_frames": int(np.count_nonzero(~np.any(stance, axis=1))),
    }
    return stance, diagnostics


def _support_root_z_correction(foot_heights: np.ndarray, stance: np.ndarray) -> np.ndarray:
    support = np.any(stance, axis=1)
    correction = np.full(foot_heights.shape[0], np.nan, dtype=np.float64)
    for frame_idx in np.flatnonzero(support):
        correction[frame_idx] = float(np.min(foot_heights[frame_idx, stance[frame_idx]]))

    support_idx = np.flatnonzero(np.isfinite(correction))
    if support_idx.size == 0:
        return np.full(foot_heights.shape[0], float(np.min(foot_heights)), dtype=np.float64)
    if support_idx.size == 1:
        return np.full(foot_heights.shape[0], correction[support_idx[0]], dtype=np.float64)
    return np.interp(
        np.arange(foot_heights.shape[0], dtype=np.float64),
        support_idx.astype(np.float64),
        correction[support_idx],
    )


def _summarize_grounding(
    foot_heights_before: np.ndarray,
    foot_heights_after: np.ndarray,
    stance: np.ndarray,
    correction: np.ndarray,
    stance_diagnostics: dict[str, object],
) -> dict[str, object]:
    support = np.any(stance, axis=1)
    stance_values = np.abs(foot_heights_after[stance])
    return {
        "mode": "contact_aware_foot_lock",
        "legacy_global_min_height_m": float(np.min(foot_heights_before)),
        "root_z_correction_min_m": float(np.min(correction)),
        "root_z_correction_median_m": float(np.median(correction)),
        "root_z_correction_max_m": float(np.max(correction)),
        "root_z_correction_max_step_m": (
            float(np.max(np.abs(np.diff(correction)))) if correction.shape[0] > 1 else 0.0
        ),
        "support_frames": int(np.count_nonzero(support)),
        "stance_abs_height_after_p95_m": float(np.percentile(stance_values, 95)) if stance_values.size else None,
        "stance_abs_height_after_max_m": float(np.max(stance_values)) if stance_values.size else None,
        "min_foot_height_after_m": float(np.min(foot_heights_after)),
        "stance_detector": stance_diagnostics,
    }


def compute_contact_aware_root_z_correction(
    root_pos_all: np.ndarray,
    root_rot_xyzw_all: np.ndarray,
    dof_pos_all: np.ndarray,
    fps: float,
    robot: str = "unitree_g1",
) -> tuple[np.ndarray, dict[str, object]]:
    model = mj.MjModel.from_xml_path(str(ROBOT_XML_DICT[robot]))
    data = mj.MjData(model)
    contact_ids_by_side = _foot_contact_geom_ids_by_side(model)
    heights_before, positions = _foot_contact_kinematics(
        model, data, contact_ids_by_side, root_pos_all, root_rot_xyzw_all, dof_pos_all
    )
    stance, stance_diagnostics = _infer_stance_mask(heights_before, positions, fps)
    correction = _support_root_z_correction(heights_before, stance)
    heights_after = heights_before - correction[:, None]
    return correction, _summarize_grounding(
        heights_before, heights_after, stance, correction, stance_diagnostics
    )


def apply_iterative_contact_aware_grounding(
    root_pos_all: np.ndarray,
    root_rot_xyzw_all: np.ndarray,
    dof_pos_all: np.ndarray,
    fps: float,
    robot: str = "unitree_g1",
    passes: int = 2,
) -> tuple[np.ndarray, dict[str, object]]:
    grounded_root = np.asarray(root_pos_all, dtype=np.float64).copy()
    total_correction = np.zeros(grounded_root.shape[0], dtype=np.float64)
    pass_reports: list[dict[str, object]] = []

    for _ in range(max(1, passes)):
        correction, report = compute_contact_aware_root_z_correction(
            grounded_root, root_rot_xyzw_all, dof_pos_all, fps, robot=robot
        )
        grounded_root[:, 2] -= correction
        total_correction += correction
        pass_reports.append(report)
        if float(np.max(np.abs(correction))) < 1e-6:
            break

    final_report = dict(pass_reports[-1])
    final_report["mode"] = "iterative_contact_aware_foot_lock"
    final_report["pass_count"] = len(pass_reports)
    final_report["passes"] = pass_reports
    final_report["legacy_global_min_height_m"] = float(pass_reports[0]["legacy_global_min_height_m"])
    final_report["root_z_correction_min_m"] = float(np.min(total_correction))
    final_report["root_z_correction_median_m"] = float(np.median(total_correction))
    final_report["root_z_correction_max_m"] = float(np.max(total_correction))
    final_report["root_z_correction_max_step_m"] = (
        float(np.max(np.abs(np.diff(total_correction)))) if total_correction.shape[0] > 1 else 0.0
    )
    return grounded_root, final_report


def convert_motion(
    pkl_path: Path,
    csv_path: Path,
    robot: str = "unitree_g1",
    grounding_passes: int = 2,
) -> dict[str, object]:
    with pkl_path.open("rb") as handle:
        motion_data = pickle.load(handle)

    grounded_root_pos, grounding_report = apply_iterative_contact_aware_grounding(
        np.asarray(motion_data["root_pos"], dtype=np.float64),
        np.asarray(motion_data["root_rot"], dtype=np.float64),
        np.asarray(motion_data["dof_pos"], dtype=np.float64),
        float(motion_data["fps"]),
        robot=robot,
        passes=grounding_passes,
    )

    motion = np.concatenate(
        [
            grounded_root_pos,
            np.asarray(motion_data["root_rot"], dtype=np.float64),
            np.asarray(motion_data["dof_pos"], dtype=np.float64),
        ],
        axis=1,
    )
    if motion.shape[1] != 36:
        raise ValueError(f"{pkl_path} produced {motion.shape[1]} columns, expected 36")

    source_fps = float(motion_data["fps"])
    output_fps = 30.0
    if abs(source_fps - output_fps) < 1e-6:
        indices = np.arange(motion.shape[0], dtype=int)
    else:
        step = source_fps / output_fps
        indices = np.arange(0, motion.shape[0], step).astype(int)
    motion_30fps = motion[indices]

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(csv_path, motion_30fps, delimiter=",")

    return {
        "source_fps": source_fps,
        "output_fps": output_fps,
        "grounding_mode": grounding_report["mode"],
        "grounding_report": grounding_report,
        "source_frames": int(motion.shape[0]),
        "output_frames": int(motion_30fps.shape[0]),
        "columns": int(motion_30fps.shape[1]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--take", type=str, default=None)
    parser.add_argument("--robot", type=str, default="unitree_g1")
    parser.add_argument("--grounding-passes", type=int, default=2)
    args = parser.parse_args()

    for pkl_path in sorted(args.input_root.glob("**/*.pkl")):
        if pkl_path.name == "metadata.pkl":
            continue
        if args.take is not None and pkl_path.stem != args.take:
            continue
        csv_path = pkl_path.with_suffix(".csv")
        report = convert_motion(
            pkl_path,
            csv_path,
            robot=args.robot,
            grounding_passes=args.grounding_passes,
        )
        report_path = pkl_path.with_name(f"{pkl_path.stem}_csv_report.json")
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(
            f"Converted {pkl_path.name} -> {csv_path.name} "
            f"({report['output_frames']} frames @ 30 fps, mode={report['grounding_mode']})"
        )


if __name__ == "__main__":
    main()
