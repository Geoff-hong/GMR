#!/usr/bin/env python3
"""Canonical BVH-folder -> grounded G1 pipeline for GMR."""

from __future__ import annotations

import argparse
import json
import pickle
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

import imageio.v2 as imageio
import mujoco as mj
import numpy as np

from general_motion_retargeting import (
    GeneralMotionRetargeting as GMR,
    ROBOT_BASE_DICT,
    ROBOT_XML_DICT,
    VIEWER_CAM_DISTANCE_DICT,
)
from general_motion_retargeting.utils.lafan1 import load_bvh_file


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from pkl_to_grounded_csv import apply_iterative_contact_aware_grounding


FINGER_TOKENS = ("Thumb", "Index", "Middle", "Ring", "Pinky")
HAND_MARKERS = ("Hand",)


@dataclass
class Joint:
    name: str
    kind: str
    parent: str | None
    indent: int
    offset: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    channels: list[str] = field(default_factory=list)
    children: list[str] = field(default_factory=list)
    end_site_offset: np.ndarray | None = None


@dataclass(frozen=True)
class TakePaths:
    source_bvh: Path
    take_dir: Path
    copied_bvh: Path
    clean_bvh: Path
    report_json: Path
    robot_pkl: Path
    robot_mp4: Path
    grounded_csv: Path
    csv_report_json: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motions-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pattern", default="take*_chr00.bvh")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument(
        "--robot",
        choices=sorted(ROBOT_XML_DICT.keys()),
        default="unitree_g1",
    )
    parser.add_argument(
        "--format",
        choices=["lafan1", "nokov", "axis", "mixamo", "3dsmax"],
        default="axis",
    )
    parser.add_argument("--human-height", type=float, default=1.85)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--camera-lookat-z-offset", type=float, default=0.10)
    parser.add_argument("--grounding-passes", type=int, default=2)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def parse_bvh(path: Path) -> tuple[list[Joint], np.ndarray, int, float]:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    joints: list[Joint] = []
    joint_by_name: dict[str, Joint] = {}
    stack: list[str | None] = []
    current_joint_name: str | None = None
    pending_end_site_for: str | None = None
    in_motion = False
    frames = 0
    frame_time = 0.0
    motion_rows: list[list[float]] = []

    for raw_line in lines:
        stripped = raw_line.strip()
        indent = len(raw_line) - len(raw_line.lstrip(" "))

        if stripped == "MOTION":
            in_motion = True
            continue

        if not in_motion:
            if stripped.startswith("ROOT "):
                name = stripped.split()[1]
                joint = Joint(name=name, kind="ROOT", parent=None, indent=indent)
                joints.append(joint)
                joint_by_name[name] = joint
                stack.append(name)
                current_joint_name = name
                continue
            if stripped.startswith("JOINT "):
                name = stripped.split()[1]
                parent = next((item for item in reversed(stack) if item is not None), None)
                joint = Joint(name=name, kind="JOINT", parent=parent, indent=indent)
                joints.append(joint)
                joint_by_name[name] = joint
                if parent is not None:
                    joint_by_name[parent].children.append(name)
                stack.append(name)
                current_joint_name = name
                continue
            if stripped.startswith("End Site"):
                pending_end_site_for = next((item for item in reversed(stack) if item is not None), None)
                stack.append(None)
                current_joint_name = None
                continue
            if stripped == "}":
                if stack:
                    popped = stack.pop()
                    if popped is None:
                        pending_end_site_for = None
                current_joint_name = next((item for item in reversed(stack) if item is not None), None)
                continue
            if stripped.startswith("OFFSET "):
                offset = np.array([float(x) for x in stripped.split()[1:4]], dtype=np.float64)
                if pending_end_site_for is not None:
                    joint_by_name[pending_end_site_for].end_site_offset = offset
                elif current_joint_name is not None:
                    joint_by_name[current_joint_name].offset = offset
                continue
            if stripped.startswith("CHANNELS ") and current_joint_name is not None:
                parts = stripped.split()
                count = int(parts[1])
                joint_by_name[current_joint_name].channels = parts[2 : 2 + count]
                continue
            continue

        if stripped.startswith("Frames:"):
            frames = int(stripped.split(":", 1)[1].strip())
            continue
        if stripped.startswith("Frame Time:"):
            frame_time = float(stripped.split(":", 1)[1].strip())
            continue
        if stripped:
            motion_rows.append([float(x) for x in stripped.split()])

    motion = np.asarray(motion_rows, dtype=np.float64)
    return joints, motion, frames, frame_time


def build_motion_column_map(joints: list[Joint]) -> dict[str, tuple[int, int]]:
    column = 0
    result: dict[str, tuple[int, int]] = {}
    for joint in joints:
        if joint.channels:
            result[joint.name] = (column, column + len(joint.channels))
            column += len(joint.channels)
    return result


def identify_finger_joints(joints: list[Joint]) -> set[str]:
    by_name = {joint.name: joint for joint in joints}
    hand_roots = [
        joint.name
        for joint in joints
        if any(marker in joint.name for marker in HAND_MARKERS)
        and joint.parent is not None
        and not any(token in joint.name for token in FINGER_TOKENS)
    ]
    finger_joints: set[str] = set()
    for root in hand_roots:
        stack = list(by_name[root].children)
        while stack:
            child = stack.pop()
            if child in finger_joints:
                continue
            finger_joints.add(child)
            stack.extend(by_name[child].children)
    return finger_joints


def finger_motion_stats(
    joints: list[Joint],
    motion: np.ndarray,
    finger_joints: set[str],
) -> tuple[dict[str, dict[str, float]], bool]:
    column_map = build_motion_column_map(joints)
    stats: dict[str, dict[str, float]] = {}
    all_static = True
    for joint_name in sorted(finger_joints):
        if joint_name not in column_map:
            continue
        start, end = column_map[joint_name]
        joint_motion = motion[:, start:end]
        value_range = float(np.max(joint_motion) - np.min(joint_motion)) if joint_motion.size else 0.0
        value_std = float(np.max(np.std(joint_motion, axis=0))) if joint_motion.size else 0.0
        is_static = bool(np.allclose(joint_motion, joint_motion[:1], atol=1e-4)) if joint_motion.size else True
        stats[joint_name] = {
            "max_range_deg": value_range,
            "max_std_deg": value_std,
            "static": is_static,
        }
        all_static = all_static and is_static
    return stats, all_static


def write_clean_bvh(
    dst_path: Path,
    joints: list[Joint],
    motion: np.ndarray,
    frames: int,
    frame_time: float,
    strip_finger_joints: bool,
) -> None:
    keep_names = {joint.name for joint in joints}
    if strip_finger_joints:
        keep_names -= identify_finger_joints(joints)
    by_name = {joint.name: joint for joint in joints}
    roots = [joint for joint in joints if joint.parent is None]
    column_map = build_motion_column_map(joints)

    def render_joint(name: str, indent: int, out_lines: list[str]) -> None:
        joint = by_name[name]
        prefix = " " * indent
        kind = "ROOT" if joint.parent is None else "JOINT"
        out_lines.append(f"{prefix}{kind} {joint.name}")
        out_lines.append(f"{prefix}{{")
        out_lines.append(
            f"{prefix}    OFFSET {joint.offset[0]:.6f} {joint.offset[1]:.6f} {joint.offset[2]:.6f}"
        )
        if joint.channels:
            out_lines.append(f"{prefix}    CHANNELS {len(joint.channels)} {' '.join(joint.channels)}")
        kept_children = [child for child in joint.children if child in keep_names]
        if kept_children:
            for child in kept_children:
                render_joint(child, indent + 4, out_lines)
        else:
            end_site_offset = joint.end_site_offset if joint.end_site_offset is not None else np.zeros(3)
            out_lines.append(f"{prefix}    End Site")
            out_lines.append(f"{prefix}    {{")
            out_lines.append(
                f"{prefix}        OFFSET {end_site_offset[0]:.6f} {end_site_offset[1]:.6f} {end_site_offset[2]:.6f}"
            )
            out_lines.append(f"{prefix}    }}")
        out_lines.append(f"{prefix}}}")

    output_lines = ["HIERARCHY"]
    for root in roots:
        render_joint(root.name, 0, output_lines)
    output_lines.append("MOTION")
    output_lines.append(f"Frames: {frames}")
    output_lines.append(f"Frame Time: {frame_time:.6f}")

    keep_indices: list[int] = []
    for joint in joints:
        if joint.name not in keep_names:
            continue
        start, end = column_map[joint.name]
        keep_indices.extend(range(start, end))
    reduced_motion = motion[:, keep_indices]
    for row in reduced_motion:
        output_lines.append(" ".join(f"{value:.6f}" for value in row))
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    dst_path.write_text("\n".join(output_lines) + "\n", encoding="utf-8")


def iter_source_bvhs(motions_dir: Path, pattern: str, recursive: bool) -> list[Path]:
    iterator = motions_dir.rglob(pattern) if recursive else motions_dir.glob(pattern)
    return sorted(path for path in iterator if path.is_file())


def build_take_paths(source_bvh: Path, output_root: Path) -> TakePaths:
    stem = source_bvh.stem
    take_dir = output_root / stem
    return TakePaths(
        source_bvh=source_bvh,
        take_dir=take_dir,
        copied_bvh=take_dir / source_bvh.name,
        clean_bvh=take_dir / f"{stem}_clean.bvh",
        report_json=take_dir / f"{stem}_report.json",
        robot_pkl=take_dir / f"{stem}.pkl",
        robot_mp4=take_dir / f"{stem}_robot.mp4",
        grounded_csv=take_dir / f"{stem}.csv",
        csv_report_json=take_dir / f"{stem}_csv_report.json",
    )


def collect_dependency_mtime() -> float:
    dependency_paths = [
        Path(__file__),
        SCRIPT_DIR / "pkl_to_grounded_csv.py",
        Path(__file__).resolve().parents[1] / "general_motion_retargeting" / "motion_retarget.py",
        SCRIPT_DIR / "bvh_to_robot.py",
    ]
    return max(path.stat().st_mtime for path in dependency_paths if path.exists())


def pkl_is_valid(pkl_path: Path) -> bool:
    if not pkl_path.exists() or pkl_path.stat().st_size == 0:
        return False
    try:
        with pkl_path.open("rb") as handle:
            motion_data = pickle.load(handle)
    except Exception:
        return False
    required = {"fps", "root_pos", "root_rot", "dof_pos"}
    if not required.issubset(motion_data):
        return False
    root_pos = np.asarray(motion_data["root_pos"])
    root_rot = np.asarray(motion_data["root_rot"])
    dof_pos = np.asarray(motion_data["dof_pos"])
    frame_count = root_pos.shape[0] if root_pos.ndim == 2 else 0
    return (
        frame_count > 1
        and root_pos.shape == (frame_count, 3)
        and root_rot.shape == (frame_count, 4)
        and dof_pos.ndim == 2
        and dof_pos.shape[0] == frame_count
    )


def csv_has_expected_columns(csv_path: Path, expected: int = 36) -> bool:
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return False
    first_line = csv_path.read_text(encoding="utf-8").splitlines()[0].strip()
    return bool(first_line) and len(first_line.split(",")) == expected


def required_outputs(paths: TakePaths) -> list[Path]:
    return [
        paths.copied_bvh,
        paths.clean_bvh,
        paths.report_json,
        paths.robot_pkl,
        paths.robot_mp4,
        paths.grounded_csv,
        paths.csv_report_json,
    ]


def outputs_are_valid(paths: TakePaths, dependency_mtime: float) -> bool:
    outputs = required_outputs(paths)
    if not all(path.exists() for path in outputs):
        return False
    if not csv_has_expected_columns(paths.grounded_csv, expected=36):
        return False
    if not pkl_is_valid(paths.robot_pkl):
        return False
    freshness_cutoff = max(paths.source_bvh.stat().st_mtime, dependency_mtime)
    return min(path.stat().st_mtime for path in outputs) >= freshness_cutoff


def write_clean_bvh_and_report(paths: TakePaths) -> dict[str, object]:
    joints, motion, frames, frame_time = parse_bvh(paths.source_bvh)
    fps = 1.0 / frame_time if frame_time else 30.0
    finger_joints = identify_finger_joints(joints)
    finger_stats, fingers_static = finger_motion_stats(joints, motion, finger_joints)
    strip_fingers = bool(finger_joints)

    paths.take_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(paths.source_bvh, paths.copied_bvh)
    write_clean_bvh(paths.clean_bvh, joints, motion, frames, frame_time, strip_fingers)
    report = {
        "file": paths.source_bvh.name,
        "joint_count": len(joints),
        "dof_count": int(motion.shape[1]),
        "frame_count": int(frames),
        "duration_s": float(frames * frame_time),
        "frame_time_s": float(frame_time),
        "fps": float(fps),
        "finger_joint_count": len(finger_joints),
        "fingers_static": bool(fingers_static),
        "fingers_stripped": bool(strip_fingers),
        "finger_stats": finger_stats,
    }
    _write_json(paths.report_json, report)
    return report


def retarget_take(
    retargeter: GMR,
    clean_bvh: Path,
    robot_pkl: Path,
    motion_format: str,
    human_height: float,
) -> dict[str, object]:
    frames, actual_human_height, bvh_fps = load_bvh_file(
        str(clean_bvh),
        format=motion_format,
        human_height_override=human_height,
    )
    motion_fps = 30 if int(round(bvh_fps)) == 30 else int(round(bvh_fps))

    retargeter.configuration.update(q=retargeter.model.qpos0.copy())
    qpos_list = []
    for frame in frames:
        qpos_list.append(retargeter.retarget(frame, offset_to_ground=True))
    qpos_array = np.asarray(qpos_list, dtype=np.float64)
    motion_data = {
        "fps": motion_fps,
        "root_pos": qpos_array[:, :3],
        "root_rot": qpos_array[:, 3:7][:, [1, 2, 3, 0]],
        "dof_pos": qpos_array[:, 7:],
        "local_body_pos": None,
        "link_body_list": None,
        "source_bvh_fps": float(bvh_fps),
        "actual_human_height": float(actual_human_height),
    }
    with robot_pkl.open("wb") as handle:
        pickle.dump(motion_data, handle)
    return motion_data


def apply_contact_aware_grounding(
    motion_data: dict[str, object],
    robot: str,
    grounding_passes: int,
) -> tuple[dict[str, object], dict[str, object]]:
    grounded_motion = dict(motion_data)
    grounded_root_pos, grounding_report = apply_iterative_contact_aware_grounding(
        np.asarray(motion_data["root_pos"], dtype=np.float64),
        np.asarray(motion_data["root_rot"], dtype=np.float64),
        np.asarray(motion_data["dof_pos"], dtype=np.float64),
        float(motion_data["fps"]),
        robot=robot,
        passes=grounding_passes,
    )
    root_z_correction = np.asarray(motion_data["root_pos"], dtype=np.float64)[:, 2] - grounded_root_pos[:, 2]
    grounded_motion["root_pos"] = grounded_root_pos
    grounded_motion["grounding_mode"] = grounding_report["mode"]
    grounded_motion["grounding_report"] = grounding_report
    grounded_motion["root_z_correction"] = root_z_correction
    return grounded_motion, grounding_report


def save_grounded_csv(
    motion_data: dict[str, object],
    csv_path: Path,
    csv_report_path: Path,
    grounding_report: dict[str, object],
) -> None:
    root_pos = np.asarray(motion_data["root_pos"], dtype=np.float64)
    root_rot = np.asarray(motion_data["root_rot"], dtype=np.float64)
    dof_pos = np.asarray(motion_data["dof_pos"], dtype=np.float64)
    source_fps = float(motion_data["fps"])

    motion = np.concatenate([root_pos, root_rot, dof_pos], axis=1)
    if motion.shape[1] != 36:
        raise ValueError(f"{csv_path} would have {motion.shape[1]} columns instead of 36")
    output_fps = 30.0
    if abs(source_fps - output_fps) < 1e-6:
        indices = np.arange(motion.shape[0], dtype=int)
    else:
        step = source_fps / output_fps
        indices = np.arange(0, motion.shape[0], step).astype(int)
    motion_30fps = motion[indices]

    np.savetxt(csv_path, motion_30fps, delimiter=",")
    _write_json(
        csv_report_path,
        {
            "source_fps": source_fps,
            "output_fps": output_fps,
            "grounding_mode": grounding_report["mode"],
            "grounding_report": grounding_report,
            "source_frames": int(motion.shape[0]),
            "output_frames": int(motion_30fps.shape[0]),
            "columns": int(motion_30fps.shape[1]),
        },
    )


def render_robot_video(
    model: mj.MjModel,
    data: mj.MjData,
    renderer: mj.Renderer,
    camera: mj.MjvCamera,
    base_body_id: int,
    motion_data: dict[str, object],
    video_path: Path,
    lookat_z_offset: float,
) -> None:
    root_pos = np.asarray(motion_data["root_pos"], dtype=np.float64)
    root_rot = np.asarray(motion_data["root_rot"], dtype=np.float64)
    dof_pos = np.asarray(motion_data["dof_pos"], dtype=np.float64)
    fps = float(motion_data["fps"])

    writer = imageio.get_writer(
        video_path,
        fps=max(1, round(fps)),
        format="FFMPEG",
        codec="libx264",
    )
    try:
        for idx in range(root_pos.shape[0]):
            data.qpos[:3] = root_pos[idx]
            data.qpos[3:7] = root_rot[idx][[3, 0, 1, 2]]
            data.qpos[7:] = dof_pos[idx]
            mj.mj_forward(model, data)
            camera.lookat[:] = data.xpos[base_body_id]
            camera.lookat[2] -= lookat_z_offset
            renderer.update_scene(data, camera=camera)
            writer.append_data(renderer.render())
    finally:
        writer.close()


def main() -> None:
    args = parse_args()
    motions_dir = args.motions_dir.resolve()
    output_dir = args.output_dir.resolve()
    source_bvhs = iter_source_bvhs(motions_dir, args.pattern, args.recursive)
    if not source_bvhs:
        raise FileNotFoundError(f"No BVHs matched pattern {args.pattern!r} under {motions_dir}")

    dependency_mtime = collect_dependency_mtime()
    output_dir.mkdir(parents=True, exist_ok=True)

    retargeter = GMR(
        src_human=f"bvh_{args.format}",
        tgt_robot=args.robot,
        actual_human_height=args.human_height,
        verbose=False,
    )
    render_model = mj.MjModel.from_xml_path(str(ROBOT_XML_DICT[args.robot]))
    render_data = mj.MjData(render_model)
    renderer = mj.Renderer(render_model, height=args.height, width=args.width)
    camera = mj.MjvCamera()
    camera.distance = VIEWER_CAM_DISTANCE_DICT[args.robot]
    camera.elevation = -10
    camera.azimuth = 180
    base_body_id = render_model.body(ROBOT_BASE_DICT[args.robot]).id

    processed: list[dict[str, object]] = []
    skipped: list[str] = []
    try:
        total = len(source_bvhs)
        for index, source_bvh in enumerate(source_bvhs, start=1):
            paths = build_take_paths(source_bvh, output_dir)
            label = source_bvh.stem
            print(f"Processing {label} [{index}/{total}]")
            if not args.force and outputs_are_valid(paths, dependency_mtime):
                print(f"Skipping {label}: outputs are already current")
                skipped.append(label)
                continue

            bvh_report = write_clean_bvh_and_report(paths)
            motion_data = retarget_take(
                retargeter,
                paths.clean_bvh,
                paths.robot_pkl,
                motion_format=args.format,
                human_height=args.human_height,
            )
            motion_data, grounding_report = apply_contact_aware_grounding(
                motion_data,
                robot=args.robot,
                grounding_passes=args.grounding_passes,
            )
            with paths.robot_pkl.open("wb") as handle:
                pickle.dump(motion_data, handle)
            render_robot_video(
                render_model,
                render_data,
                renderer,
                camera,
                base_body_id,
                motion_data,
                paths.robot_mp4,
                args.camera_lookat_z_offset,
            )
            save_grounded_csv(motion_data, paths.grounded_csv, paths.csv_report_json, grounding_report)
            processed.append(
                {
                    "take": label,
                    "original_bvh": str(paths.copied_bvh),
                    "clean_bvh": str(paths.clean_bvh),
                    "g1_motion_pkl": str(paths.robot_pkl),
                    "g1_video_mp4": str(paths.robot_mp4),
                    "g1_csv": str(paths.grounded_csv),
                    "frame_count": int(np.asarray(motion_data["root_pos"]).shape[0]),
                    "motion_fps": float(motion_data["fps"]),
                    "grounding_mode": motion_data.get("grounding_mode"),
                    "fingers_stripped": bool(bvh_report["fingers_stripped"]),
                }
            )
    finally:
        renderer.close()

    _write_json(
        output_dir / "retarget_manifest.json",
        {
            "motions_dir": str(motions_dir),
            "output_dir": str(output_dir),
            "robot": args.robot,
            "format": args.format,
            "human_height": args.human_height,
            "pattern": args.pattern,
            "recursive": bool(args.recursive),
            "grounding_passes": int(args.grounding_passes),
            "camera_lookat_z_offset_m": float(args.camera_lookat_z_offset),
            "source_count": len(source_bvhs),
            "processed_count": len(processed),
            "skipped_count": len(skipped),
            "entries": processed,
            "skipped": skipped,
        },
    )
    print(
        f"Retargeting complete: {len(processed)} processed, {len(skipped)} skipped, "
        f"{len(source_bvhs)} total. Manifest: {output_dir / 'retarget_manifest.json'}"
    )


if __name__ == "__main__":
    main()
