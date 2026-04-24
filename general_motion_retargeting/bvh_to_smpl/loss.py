"""3D joint-position fitting losses for SMPL-X, adapted from smplify-x/fitting.py.

SMPLify-X's original SMPLifyLoss projects 3D SMPL-X joints to 2D via a camera
and compares against 2D keypoint detections. Our fitter has direct 3D joint
positions (from BVH forward kinematics), so the data term is a Euclidean L2
between SMPL-X forward-pass joints and observed 3D joints. We keep SMPLify-X's
structure for:

  - shape prior (L2 on betas)
  - angle prior (penalise negative knee/elbow hyperextension)
  - temporal smoothness on pose + translation between frames

Pose prior (MoG / VPoser) from SMPLify-X is intentionally NOT included here;
fitting against well-defined 3D keypoints is already well-conditioned and the
prior adds a heavy dependency (VPoser weights / MoG .pkl). If motions drift
into implausible poses we can add it later.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class Smpl3DLossWeights:
    """Loss term weights. Defaults are sane for the blanket dataset scale (meters)."""

    data: float = 1.0              # L2 on joint positions
    shape: float = 0.01            # ||betas||^2
    angle: float = 0.1             # knee / elbow hyperextension penalty
    smoothness_pose: float = 0.05  # ||pose[t+1] - pose[t]||^2
    smoothness_transl: float = 0.01


class AnglePrior(nn.Module):
    """Penalise hyperextension of elbows and knees (from SMPLify-X prior.py)."""

    # SMPL body_pose is 23 joints × 3 axis-angle.
    # Body_pose joint indices (0-based within body_pose, i.e. excluding the pelvis root):
    #   3 -> L_knee   (SMPL joint 4)
    #   6 -> R_knee   (SMPL joint 5)
    #   17 -> L_elbow (SMPL joint 18)
    #   18 -> R_elbow (SMPL joint 19)
    # Axis indices along the axis-angle within each joint (the axis of hinge bending).
    # SMPLify-X uses the bend axis that, when negative, represents hyperextension.
    _ANGLE_JOINT_IDS = [55, 58, 12, 15]  # linear indices into body_pose flat
    _ANGLE_SIGNS = torch.tensor([1.0, -1.0, 1.0, -1.0])

    def forward(self, body_pose: torch.Tensor) -> torch.Tensor:
        # body_pose: (T, 69)
        signs = self._ANGLE_SIGNS.to(body_pose.device)
        vals = body_pose[:, self._ANGLE_JOINT_IDS] * signs  # (T, 4)
        return torch.exp(vals).pow(2).sum(dim=-1).mean()


class Smpl3DFittingLoss(nn.Module):
    """Compose data + priors into a single scalar loss."""

    def __init__(self, weights: Smpl3DLossWeights | None = None,
                 joint_weights: torch.Tensor | None = None):
        super().__init__()
        self.w = weights if weights is not None else Smpl3DLossWeights()
        self.angle_prior = AnglePrior()
        # joint_weights: (N_fit,) per-joint weighting for the L2 data term
        self.register_buffer("joint_weights",
                             joint_weights if joint_weights is not None else torch.ones(22))

    def forward(
        self,
        pred_joints: torch.Tensor,      # (T, N_fit, 3) from SMPL-X forward pass
        gt_joints: torch.Tensor,        # (T, N_fit, 3) observed 3D joints
        betas: torch.Tensor,            # (1, 10) shared shape
        body_pose: torch.Tensor,        # (T, 69)
        transl: torch.Tensor,           # (T, 3)
    ) -> tuple[torch.Tensor, dict[str, float]]:
        w_j = self.joint_weights.to(pred_joints.device).view(1, -1, 1)
        data = ((pred_joints - gt_joints) * w_j).pow(2).sum(-1).mean()

        shape = betas.pow(2).sum()
        angle = self.angle_prior(body_pose)

        if body_pose.shape[0] > 1:
            smooth_pose = (body_pose[1:] - body_pose[:-1]).pow(2).sum(-1).mean()
            smooth_transl = (transl[1:] - transl[:-1]).pow(2).sum(-1).mean()
        else:
            smooth_pose = torch.zeros((), device=pred_joints.device)
            smooth_transl = torch.zeros((), device=pred_joints.device)

        total = (
            self.w.data * data
            + self.w.shape * shape
            + self.w.angle * angle
            + self.w.smoothness_pose * smooth_pose
            + self.w.smoothness_transl * smooth_transl
        )
        logs = {
            "total": float(total.detach()),
            "data": float(data.detach()),
            "shape": float(shape.detach()),
            "angle": float(angle.detach()),
            "smooth_pose": float(smooth_pose.detach()),
            "smooth_transl": float(smooth_transl.detach()),
        }
        return total, logs
