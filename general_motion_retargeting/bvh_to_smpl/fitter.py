"""Batched SMPL-X fitter: 6-D rotation parameterisation, multi-stage Adam.

Improvements vs v2:
  * **6-D rotation parameterisation** for global_orient and all 21 body joints.
    Axis-angle has a wrap discontinuity at |theta|=pi; Adam was drifting past
    this boundary (v2 produced a 3.9-rad R_knee and a 3.5-rad R_elbow because
    the optimiser moved along "the long way around"). 6-D is globally
    continuous, so the optimiser stays near identity at rest and smoothly
    accumulates rotation.
  * **Three-stage schedule** (A global translation/orient warm-up, B positional
    fit with gentle priors, C final tightening with smoothness + hinge).
  * **Warm-start global_orient** from the pelvis-to-chest axis of the observed
    joints so the body is roughly upright before stage B, which is critical
    since 6-D starting from identity is very far from a facing-backward pose.
  * Passes rotation matrices (not just axis-angle) to the loss so that
    temporal smoothness and hinge penalties operate in SO(3) where they are
    well-defined.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import smplx
import torch

from .loss import Smpl3DFittingLoss, Smpl3DLossWeights
from .rotations import (
    axis_angle_to_matrix,
    axis_angle_to_rot6d,
    identity_rot6d,
    matrix_to_axis_angle,
    rot6d_to_matrix,
)


@dataclass
class FitterConfig:
    smpl_model_dir: str
    gender: str = "neutral"
    num_betas: int = 10
    device: str = "cuda"
    n_iters_A: int = 120          # stage A: global warm-up only
    n_iters_B: int = 300          # stage B: body fit, gentle priors
    n_iters_C: int = 400          # stage C: full priors + smoothness
    lr_A: float = 0.05
    lr_B: float = 0.03
    lr_C: float = 0.005
    fit_joint_count: int = 22
    log_every: int = 50


class BatchSmplxFitter:
    def __init__(self, cfg: FitterConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    # ----- model / forward helpers -----

    def _make_model(self, batch_size: int) -> smplx.SMPLX:
        return smplx.create(
            self.cfg.smpl_model_dir,
            model_type="smplx",
            gender=self.cfg.gender,
            ext="pkl",
            num_betas=self.cfg.num_betas,
            use_pca=False,
            flat_hand_mean=True,
            batch_size=batch_size,
        ).to(self.device)

    def _forward(
        self,
        model,
        go_aa: torch.Tensor,        # (T, 3) axis-angle from 6-D
        body_aa: torch.Tensor,      # (T, 63)
        transl: torch.Tensor,       # (T, 3)
        betas: torch.Tensor,        # (1, num_betas)
    ):
        T = go_aa.shape[0]
        return model(
            betas=betas.expand(T, -1),
            global_orient=go_aa,
            body_pose=body_aa,
            transl=transl,
            return_verts=False,
        )

    # ----- initialisation -----
    # NOTE: we do NOT warm-start global_orient any more. An earlier heuristic
    # that aligned SMPL-X's rest spine (+Y in model frame) to the observed
    # spine direction in world Z-up was locally correct in pitch but left
    # the yaw (facing direction) unresolved, producing a body pointed 180 deg
    # away from the demonstrator. Stage A then had to bend hips/knees to
    # huge angles (>100 deg) to compensate. Leaving go_r6 at identity and
    # letting Adam walk the full SO(3) manifold during stage A is empirically
    # more reliable with the strong body_pose L2 + smooth_rot priors added
    # in this version.

    # ----- main fit -----

    def fit_clip(self, gt_joints_3d: np.ndarray, fps: float) -> dict:
        """
        gt_joints_3d: (T, 22, 3) float, same world frame as SMPL-X output (joints + transl).
        """
        assert gt_joints_3d.ndim == 3 and gt_joints_3d.shape[1] == self.cfg.fit_joint_count
        T = gt_joints_3d.shape[0]
        model = self._make_model(T)
        gt = torch.as_tensor(gt_joints_3d, dtype=torch.float32, device=self.device)

        # -------- initialise --------
        # Start from identity rotations; Adam finds global_orient in stage A.
        go_r6 = identity_rot6d(T, device=self.device).requires_grad_(True)             # (T, 6)
        body_r6 = identity_rot6d(T, 21, device=self.device).requires_grad_(True)       # (T, 21, 6)
        betas = torch.zeros(1, self.cfg.num_betas, device=self.device, requires_grad=True)

        # Initial transl: pelvis target minus rest pelvis offset, computed with betas=0
        with torch.no_grad():
            rest = model(betas=torch.zeros(1, self.cfg.num_betas, device=self.device))
            rest_pelvis = rest.joints[0, 0].clone()
        transl_init = gt[:, 0] - rest_pelvis.unsqueeze(0)
        transl = transl_init.detach().clone().requires_grad_(True)

        def pack_axis_angle():
            """Compute (go_aa, body_aa, joint_R) from the 6-D parameters."""
            go_R = rot6d_to_matrix(go_r6)                    # (T, 3, 3)
            body_R = rot6d_to_matrix(body_r6)                # (T, 21, 3, 3)
            go_aa = matrix_to_axis_angle(go_R)               # (T, 3)
            body_aa = matrix_to_axis_angle(body_R).reshape(T, 63)
            joint_R = torch.cat([go_R.unsqueeze(1), body_R], dim=1)  # (T, 22, 3, 3)
            return go_aa, body_aa, joint_R

        # -------- loss configurations per stage --------
        # Loss terms are averaged over (T, ...). data term is per-joint squared Euclidean
        # in metres^2, so realistic target magnitudes are O(1e-4) once per-joint RMS ~ 1cm.
        # Priors are in axis-angle rad^2 and are scaled up unless data weight dominates,
        # so we lift data weight relative to priors considerably from the previous version.
        # Stage A: warm-up global orient + translation with identity body pose, no priors.
        w_A = Smpl3DLossWeights(
            data=100.0, body_pose_l2=0.0, shape_l2=0.01, angle_prior=0.0,
            smooth_rot=0.0, smooth_transl=0.0,
        )
        # Stage B: activate body pose, almost pure data fit so positions match tightly.
        w_B = Smpl3DLossWeights(
            data=100.0, body_pose_l2=0.001, shape_l2=0.005, angle_prior=0.0,
            smooth_rot=0.01, smooth_transl=0.001,
        )
        # Stage C: preserve data fit, enable temporal smoothness (which regularises rotations
        # without biasing the mean pose); modest pose L2 to avoid wrap past pi.
        w_C = Smpl3DLossWeights(
            data=100.0, body_pose_l2=0.005, shape_l2=0.005, angle_prior=0.0,
            smooth_rot=0.1, smooth_transl=0.005,
        )

        loss_A = Smpl3DFittingLoss(w_A, self._jw_tensor).to(self.device)
        loss_B = Smpl3DFittingLoss(w_B, self._jw_tensor).to(self.device)
        loss_C = Smpl3DFittingLoss(w_C, self._jw_tensor).to(self.device)

        # -------- stage A: freeze body rotations, only optimise global_orient + transl --------
        # (equivalent to optimising just the "where is the body" part)
        body_r6_frozen = body_r6.detach()
        opt_A = torch.optim.Adam([go_r6, transl, betas], lr=self.cfg.lr_A)
        print(f"[fitter] stage A: global orient + transl, {self.cfg.n_iters_A} iters")
        for it in range(self.cfg.n_iters_A):
            opt_A.zero_grad()
            go_aa, body_aa, joint_R = pack_axis_angle()
            # replace body_r6 rotation with frozen identity (by nullifying its gradient path)
            body_aa_frozen = matrix_to_axis_angle(rot6d_to_matrix(body_r6_frozen)).reshape(T, 63)
            body_aa_frozen = body_aa_frozen.detach()
            out = self._forward(model, go_aa, body_aa_frozen, transl, betas)
            pred = out.joints[:, : self.cfg.fit_joint_count]
            body_pose_aa = body_aa_frozen.reshape(T, 21, 3)
            joint_R_A = torch.cat(
                [rot6d_to_matrix(go_r6).unsqueeze(1), rot6d_to_matrix(body_r6_frozen).expand(T, -1, -1, -1)],
                dim=1,
            )
            total, logs = loss_A(pred, gt, betas, body_pose_aa, joint_R_A, transl)
            total.backward()
            opt_A.step()
            if it % self.cfg.log_every == 0 or it == self.cfg.n_iters_A - 1:
                print(f"  A it={it:3d}  total={logs['total']:.4f}  data={logs['data']:.4f}")

        # -------- stage B: body pose + priors, gentle smoothness --------
        opt_B = torch.optim.Adam([go_r6, body_r6, transl, betas], lr=self.cfg.lr_B)
        print(f"[fitter] stage B: body fit, {self.cfg.n_iters_B} iters")
        for it in range(self.cfg.n_iters_B):
            opt_B.zero_grad()
            go_aa, body_aa, joint_R = pack_axis_angle()
            out = self._forward(model, go_aa, body_aa, transl, betas)
            pred = out.joints[:, : self.cfg.fit_joint_count]
            body_pose_aa = body_aa.reshape(T, 21, 3)
            total, logs = loss_B(pred, gt, betas, body_pose_aa, joint_R, transl)
            total.backward()
            opt_B.step()
            if it % self.cfg.log_every == 0 or it == self.cfg.n_iters_B - 1:
                print(
                    f"  B it={it:3d}  tot={logs['total']:.4f}  data={logs['data']:.4f} "
                    f"pose_l2={logs['pose_l2']:.3f} knee={logs['knee_hinge']:.3f} "
                    f"ankle_twist={logs['ankle_twist']:.3f} smooth_rot={logs['smooth_rot']:.3f}"
                )

        # -------- stage C: full priors + strong smoothness --------
        opt_C = torch.optim.Adam([go_r6, body_r6, transl, betas], lr=self.cfg.lr_C)
        print(f"[fitter] stage C: refine w/ priors + smoothness, {self.cfg.n_iters_C} iters")
        for it in range(self.cfg.n_iters_C):
            opt_C.zero_grad()
            go_aa, body_aa, joint_R = pack_axis_angle()
            out = self._forward(model, go_aa, body_aa, transl, betas)
            pred = out.joints[:, : self.cfg.fit_joint_count]
            body_pose_aa = body_aa.reshape(T, 21, 3)
            total, logs = loss_C(pred, gt, betas, body_pose_aa, joint_R, transl)
            total.backward()
            opt_C.step()
            if it % self.cfg.log_every == 0 or it == self.cfg.n_iters_C - 1:
                print(
                    f"  C it={it:3d}  tot={logs['total']:.4f}  data={logs['data']:.4f} "
                    f"pose_l2={logs['pose_l2']:.3f} knee={logs['knee_hinge']:.3f} "
                    f"elbow={logs['elbow_hinge']:.3f} ankle_twist={logs['ankle_twist']:.3f} "
                    f"smooth_rot={logs['smooth_rot']:.3f}"
                )

        # -------- final forward pass (transl=0 for SONIC model-frame joints) --------
        with torch.no_grad():
            go_aa, body_aa, _ = pack_axis_angle()
            final = model(
                betas=betas.expand(T, -1),
                global_orient=go_aa,
                body_pose=body_aa,
                transl=torch.zeros_like(transl),
                return_verts=False,
            )
            joints_mf = final.joints[:, : self.cfg.fit_joint_count].detach().cpu().numpy()
            hand_proxy = final.joints[:, 20:22].detach().cpu().numpy()
            smpl_joints_24 = np.concatenate([joints_mf, hand_proxy], axis=1)

        return {
            "global_orient": go_aa.detach().cpu().numpy(),
            "body_pose": body_aa.detach().cpu().numpy(),
            "transl": transl.detach().cpu().numpy(),
            "betas": betas.detach().cpu().numpy()[0],
            "smpl_joints": smpl_joints_24,
            "fps": float(fps),
        }

    # joint weights buffer (used when the CLI monkey-patches via attr)
    _jw_tensor: torch.Tensor | None = None

    def with_joint_weights(self, w: np.ndarray) -> "BatchSmplxFitter":
        self._jw_tensor = torch.as_tensor(w, dtype=torch.float32)
        return self
