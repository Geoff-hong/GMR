"""Headless video renderer for GMR robot motion pkl files.
Uses MuJoCo offscreen rendering (EGL/osmesa) without GLFW/display.
"""
import argparse
import os
import pickle
import numpy as np
import mujoco as mj
import imageio
from tqdm import tqdm
from general_motion_retargeting import ROBOT_XML_DICT, ROBOT_BASE_DICT, VIEWER_CAM_DISTANCE_DICT


def load_robot_motion(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    fps = data["fps"]
    root_pos = data["root_pos"]
    # pkl stores xyzw, mujoco needs wxyz
    root_rot_xyzw = data["root_rot"]
    root_rot = root_rot_xyzw[:, [3, 0, 1, 2]]
    dof_pos = data["dof_pos"]
    return fps, root_pos, root_rot, dof_pos


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", type=str, default="unitree_g1")
    parser.add_argument("--robot_motion_path", type=str, required=True)
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()

    xml_path = str(ROBOT_XML_DICT[args.robot])
    model = mj.MjModel.from_xml_path(xml_path)
    data = mj.MjData(model)
    robot_base = ROBOT_BASE_DICT[args.robot]
    cam_distance = VIEWER_CAM_DISTANCE_DICT[args.robot]

    fps, root_pos, root_rot, dof_pos = load_robot_motion(args.robot_motion_path)
    n_frames = len(root_pos)

    renderer = mj.Renderer(model, height=args.height, width=args.width)

    # Setup camera
    cam = mj.MjvCamera()
    cam.distance = cam_distance
    cam.elevation = -10
    cam.azimuth = 180

    video_dir = os.path.dirname(args.video_path)
    if video_dir:
        os.makedirs(video_dir, exist_ok=True)
    writer = imageio.get_writer(args.video_path, fps=fps)

    for i in tqdm(range(n_frames), desc="Rendering"):
        data.qpos[:3] = root_pos[i]
        data.qpos[3:7] = root_rot[i]
        data.qpos[7:] = dof_pos[i]
        mj.mj_forward(model, data)

        # Follow camera
        cam.lookat[:] = data.xpos[model.body(robot_base).id]

        renderer.update_scene(data, camera=cam)
        img = renderer.render()
        writer.append_data(img)

    writer.close()
    print(f"Video saved to {args.video_path} ({n_frames} frames, {fps} fps)")
