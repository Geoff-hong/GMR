import numpy as np
from scipy.spatial.transform import Rotation as R

import general_motion_retargeting.utils.lafan_vendor.utils as utils
from general_motion_retargeting.utils.lafan_vendor.extract import read_bvh


def load_bvh_file(bvh_file, format="lafan1", human_height_override=None):
    """
    Load a BVH file and return per-frame joint data, human height, and fps.

    Returns:
        frames: list of dicts mapping joint name -> [position, orientation]
        human_height: float (meters)
        fps: int (frames per second from BVH header)
    """
    data = read_bvh(bvh_file)
    global_data = utils.quat_fk(data.quats, data.pos, data.parents)

    rotation_matrix = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    rotation_quat = R.from_matrix(rotation_matrix).as_quat(scalar_first=True)

    # Name mapping for 3DSMAX format -> LAFAN1-compatible names
    _3DSMAX_NAME_MAP = {
        "RightHip": "RightUpLeg", "RightKnee": "RightLeg", "RightAnkle": "RightFoot",
        "LeftHip": "LeftUpLeg", "LeftKnee": "LeftLeg", "LeftAnkle": "LeftFoot",
        "Chest": "Spine", "Chest2": "Spine1", "Chest3": "Spine2",
        "RightCollar": "RightShoulder", "RightShoulder": "RightArm",
        "RightElbow": "RightForeArm", "RightWrist": "RightHand",
        "LeftCollar": "LeftShoulder", "LeftShoulder": "LeftArm",
        "LeftElbow": "LeftForeArm", "LeftWrist": "LeftHand",
    }

    frames = []
    for frame in range(data.pos.shape[0]):
        result = {}
        for i, bone in enumerate(data.bones):
            orientation = utils.quat_mul(rotation_quat, global_data[0][frame, i])
            position = global_data[1][frame, i] @ rotation_matrix.T / 100  # cm to m

            if format == "mixamo":
                key = bone.replace("mixamorig:", "")
            elif format == "3dsmax":
                key = _3DSMAX_NAME_MAP.get(bone, bone)
            else:
                key = bone
            result[key] = [position, orientation]

        if format == "lafan1":
            # Add modified foot pose
            result["LeftFootMod"] = [result["LeftFoot"][0], result["LeftToe"][1]]
            result["RightFootMod"] = [result["RightFoot"][0], result["RightToe"][1]]
        elif format == "nokov":
            result["LeftFootMod"] = [result["LeftFoot"][0], result["LeftToeBase"][1]]
            result["RightFootMod"] = [result["RightFoot"][0], result["RightToeBase"][1]]
        elif format == "mixamo":
            result["LeftFootMod"] = [result["LeftFoot"][0], result["LeftToeBase"][1]]
            result["RightFootMod"] = [result["RightFoot"][0], result["RightToeBase"][1]]
        elif format in ("axis", "3dsmax"):
            # No toe joints available; use foot orientation as fallback
            result["LeftFootMod"] = [result["LeftFoot"][0], result["LeftFoot"][1]]
            result["RightFootMod"] = [result["RightFoot"][0], result["RightFoot"][1]]
        else:
            raise ValueError(f"Invalid format: {format}")

        frames.append(result)

    if human_height_override is not None:
        human_height = human_height_override
    else:
        human_height = 1.75

    fps = round(1.0 / data.frametime) if data.frametime else 30
    return frames, human_height, fps
