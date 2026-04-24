"""3D joint fitting losses for SMPL-X. Adapted from smplify-x/fitting.py.

Key differences from the v2 version:
  * AnglePrior indices are the body-pose-local indices (post -3 shift from
    SMPLify-X's full-pose indices). The v2 code forgot the shift, which
    left knees and elbows unpenalised -- the optimiser then wrapped their
    axis-angle into 3-4 rad magnitudes (equivalent to >200 deg), producing
    the wildly spinning knees visible in the v2 comparison video.
  * Body-pose L2 term to keep joint rotations near rest instead of drifting
    into physically implausible, position-preserving but rotation-ambiguous
    solutions (e.g. the ankle that was twisting 100 deg to chase toe tips).
  * Hinge constraint for knees and elbows: their rotation matrix should
    represent a rotation around a single, joint-local hinge axis. We express
    this by penalising the off-axis components of the log-map (axis-angle).
  * Ankle twist constraint: penalise rotation component along the tibia axis
    (parent-to-child bone direction), which is the DOF under-constrained by
    joint position alone.
  * Rotation-matrix smoothness (geodesic L2 distance between consecutive
    per-joint rotation matrices) replaces axis-angle L2, which is discontinuous
    at |theta|=pi.

All loss terms work in rotation-matrix space (shape (T, N_joints, 3, 3)) so
the fitter can feed rotations computed from 6-D representation directly.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .rotations import matrix_to_axis_angle, geodesic_distance


# SMPL body_pose joint indices (within the 21-joint body_pose). Used to tag
# which joints are knees / elbows / ankles for specialised priors.
class BodyPoseIdx:
    L_HIP = 0
    R_HIP = 1
    SPINE1 = 2
    L_KNEE = 3
    R_KNEE = 4
    SPINE2 = 5
    L_ANKLE = 6
    R_ANKLE = 7
    SPINE3 = 8
    L_FOOT = 9
    R_FOOT = 10
    NECK = 11
    L_COLLAR = 12
    R_COLLAR = 13
    HEAD = 14
    L_SHOULDER = 15
    R_SHOULDER = 16
    L_ELBOW = 17
    R_ELBOW = 18
    L_WRIST = 19
    R_WRIST = 20


@dataclass
class Smpl3DLossWeights:
    """Loss term weights (m and rad units).

    Defaults tuned for the Axis/G1 blanket dataset: joint positions in metres,
    motion dynamics ~ ±50 cm / s for limbs, 30 Hz capture.

    hinge_knee / hinge_elbow / ankle_twist are OFF by default (weight 0).
    They require correct joint-local rotation-axis conventions per joint,
    which differ across SMPL skeletons and versions. Enabling them with the
    wrong axis biases the fit in a wrong direction (as happened in the first
    v3 attempt: knee hinge axis +X was the wrong sign for this SMPL-X build
    and the optimiser happily bent knees 115 deg backwards while satisfying
    hinge=0). Leave them off unless you have verified the axes empirically.
    """

    data: float = 1.0                  # L2 on joint positions (primary data term)
    body_pose_l2: float = 0.01         # keep body_pose rotations near rest (gentle)
    shape_l2: float = 0.01             # ||betas||^2
    angle_prior: float = 0.0           # SMPLify-X exp() prior disabled: without VPoser warm-start it blows
                                       # up stage C when aa magnitudes exceed ~1.5 rad.
    hinge_knee: float = 0.0
    hinge_elbow: float = 0.0
    ankle_twist: float = 0.0
    smooth_rot: float = 0.2            # geodesic L2 between consecutive frames
    smooth_transl: float = 0.05        # L2 on pelvis translation velocity


class Smpl3DFittingLoss(nn.Module):
    def __init__(
        self,
        weights: Smpl3DLossWeights | None = None,
        joint_weights: torch.Tensor | None = None,
    ):
        super().__init__()
        self.w = weights if weights is not None else Smpl3DLossWeights()
        # joint_weights: (N_fit,) per-joint positional weight
        self.register_buffer(
            "joint_weights",
            joint_weights if joint_weights is not None else torch.ones(22),
        )

    # ---------- individual terms ----------

    def _data_loss(self, pred_joints: torch.Tensor, gt_joints: torch.Tensor) -> torch.Tensor:
        w = self.joint_weights.to(pred_joints.device).view(1, -1, 1)
        return ((pred_joints - gt_joints) * w).pow(2).sum(-1).mean()

    def _body_pose_l2(self, body_pose_aa: torch.Tensor) -> torch.Tensor:
        """L2 on axis-angle magnitude across all body_pose entries.

        body_pose_aa: (T, 21, 3). Keeps joint rotations near rest.
        """
        return body_pose_aa.pow(2).sum(-1).mean()

    def _angle_prior_knee_elbow(self, body_pose_aa: torch.Tensor) -> torch.Tensor:
        """Hyperextension penalty replicating SMPLify-X's angle_prior.

        SMPLify-X (prior.py) uses indices [55, 58, 12, 15] into the *full*
        pose (root + body); the body-pose-local versions (subtract 3) are:
            L_knee  [full 12] -> body_pose flat 9  (x component of L_knee aa)
            R_knee  [full 15] -> body_pose flat 12 (x component of R_knee aa)
            L_elbow [full 55] -> body_pose flat 52 (y component of L_elbow aa)
            R_elbow [full 58] -> body_pose flat 55 (y component of R_elbow aa)
        Signs: [1, -1, -1, -1]. Penalty = sum_i exp(sign_i * value_i)^2.
        """
        flat = body_pose_aa.reshape(body_pose_aa.shape[0], -1)  # (T, 63)
        idxs = torch.tensor([9, 12, 52, 55], device=flat.device)
        signs = torch.tensor([1.0, -1.0, -1.0, -1.0], device=flat.device)
        vals = flat[:, idxs] * signs
        return torch.exp(vals).pow(2).sum(-1).mean()

    def _hinge_penalty(
        self,
        body_pose_aa: torch.Tensor,
        joint_ids: list[int],
        hinge_axis: torch.Tensor,
    ) -> torch.Tensor:
        """Penalty for rotation components not along a hinge axis.

        Physical knees and elbows are approximately 1-DOF hinges. The axis-angle
        of a pure hinge rotation has direction parallel to the joint's hinge
        axis (in the joint's local frame). Any component of the axis-angle
        orthogonal to the hinge axis is a non-physical bend.

        body_pose_aa: (T, 21, 3). joint_ids: list of indices into 21. hinge_axis:
        (3,) unit vector in the joint's local frame.
        """
        # Select joints
        aa = body_pose_aa[:, joint_ids, :]   # (T, K, 3)
        hinge = hinge_axis.to(aa.device).view(1, 1, 3)
        # Component along hinge, aligned axis, and perpendicular
        parallel = (aa * hinge).sum(-1, keepdim=True) * hinge   # (T, K, 3)
        perp = aa - parallel
        return perp.pow(2).sum(-1).mean()

    def _ankle_twist_penalty(
        self,
        body_pose_aa: torch.Tensor,
        tibia_axis: torch.Tensor,
    ) -> torch.Tensor:
        """Penalty for ankle rotation around the tibia axis.

        At rest, the tibia runs roughly along the joint-local Y axis in SMPL-X
        (long-bone direction). Twist around this axis is under-constrained by
        position alone: any twist that keeps the foot on the same circle of
        revolution around the tibia satisfies the joint-position data term.
        Penalising it forces a natural, twist-free ankle.
        """
        aa = body_pose_aa[:, [BodyPoseIdx.L_ANKLE, BodyPoseIdx.R_ANKLE], :]  # (T, 2, 3)
        axis = tibia_axis.to(aa.device).view(1, 1, 3)
        twist = (aa * axis).sum(-1)
        return twist.pow(2).mean()

    def _rotation_smoothness(self, joint_R: torch.Tensor) -> torch.Tensor:
        """Geodesic L2 between consecutive per-joint rotation matrices.

        joint_R: (T, N_rot, 3, 3).
        Avoids axis-angle wrapping that would occur with L2 on axis-angle.
        """
        if joint_R.shape[0] < 2:
            return joint_R.new_zeros(())
        d = geodesic_distance(joint_R[:-1], joint_R[1:])  # (T-1, N_rot)
        return d.pow(2).mean()

    def _transl_smoothness(self, transl: torch.Tensor) -> torch.Tensor:
        if transl.shape[0] < 2:
            return transl.new_zeros(())
        return (transl[1:] - transl[:-1]).pow(2).sum(-1).mean()

    # ---------- composite ----------

    def forward(
        self,
        pred_joints: torch.Tensor,        # (T, N_fit, 3)
        gt_joints: torch.Tensor,          # (T, N_fit, 3)
        betas: torch.Tensor,              # (1, 10)
        body_pose_aa: torch.Tensor,       # (T, 21, 3) axis-angle (for readable priors)
        joint_R: torch.Tensor,            # (T, 22, 3, 3) rotation matrices (global_orient + 21 body)
        transl: torch.Tensor,             # (T, 3)
    ) -> tuple[torch.Tensor, dict[str, float]]:

        data = self._data_loss(pred_joints, gt_joints)
        pose_l2 = self._body_pose_l2(body_pose_aa)
        shape_l2 = betas.pow(2).sum()
        angle = self._angle_prior_knee_elbow(body_pose_aa)

        # knee hinge axis: +X in joint-local frame is the canonical SMPL knee-bend axis
        hinge_x = torch.tensor([1.0, 0.0, 0.0])
        knee_hinge = self._hinge_penalty(
            body_pose_aa, [BodyPoseIdx.L_KNEE, BodyPoseIdx.R_KNEE], hinge_x,
        )
        elbow_hinge = self._hinge_penalty(
            body_pose_aa, [BodyPoseIdx.L_ELBOW, BodyPoseIdx.R_ELBOW], hinge_x,
        )
        # tibia axis ~ Y in ankle-local frame (long-bone direction)
        tibia_y = torch.tensor([0.0, 1.0, 0.0])
        ankle_twist = self._ankle_twist_penalty(body_pose_aa, tibia_y)

        smooth_rot = self._rotation_smoothness(joint_R)
        smooth_tr = self._transl_smoothness(transl)

        total = (
            self.w.data * data
            + self.w.body_pose_l2 * pose_l2
            + self.w.shape_l2 * shape_l2
            + self.w.angle_prior * angle
            + self.w.hinge_knee * knee_hinge
            + self.w.hinge_elbow * elbow_hinge
            + self.w.ankle_twist * ankle_twist
            + self.w.smooth_rot * smooth_rot
            + self.w.smooth_transl * smooth_tr
        )
        logs = {
            "total": float(total.detach()),
            "data": float(data.detach()),
            "pose_l2": float(pose_l2.detach()),
            "shape_l2": float(shape_l2.detach()),
            "angle": float(angle.detach()),
            "knee_hinge": float(knee_hinge.detach()),
            "elbow_hinge": float(elbow_hinge.detach()),
            "ankle_twist": float(ankle_twist.detach()),
            "smooth_rot": float(smooth_rot.detach()),
            "smooth_transl": float(smooth_tr.detach()),
        }
        return total, logs
