"""Batched SMPL-X fitter from 3D joint observations.

Approach: cast one whole motion clip's SMPL-X parameters as learnable tensors
and run PyTorch Adam to jointly optimise all frames. Much faster than SMPLify-X's
per-frame L-BFGS (seconds per clip instead of minutes) at the cost of slightly
less tight convergence per frame.

Two-stage schedule (inspired by SMPLify-X's weight staging):
  Stage A (init):   high data weight, priors off / small -> pure positional fit
  Stage B (refine): data weight down a notch, priors + smoothness on
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import smplx
import torch

from .loss import Smpl3DFittingLoss, Smpl3DLossWeights


@dataclass
class FitterConfig:
    smpl_model_dir: str               # directory containing models/smplx/SMPLX_*.pkl
    gender: str = "neutral"           # "neutral" | "male" | "female"
    num_betas: int = 10
    device: str = "cuda"
    n_iters_init: int = 200
    n_iters_refine: int = 400
    lr_init: float = 0.05
    lr_refine: float = 0.01
    fit_joint_count: int = 22         # we fit against the 22 body joints (SMPL 0..21)
    log_every: int = 50


class BatchSmplxFitter:
    """Fit SMPL-X (global_orient, body_pose, transl, shared betas) to 3D joint targets."""

    def __init__(self, cfg: FitterConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

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

    def _forward_joints(self, model, global_orient, body_pose, transl, betas):
        """SMPL-X forward pass; return only the first N fit joints."""
        out = model(
            betas=betas.expand(global_orient.shape[0], -1),
            global_orient=global_orient,
            body_pose=body_pose,
            transl=transl,
            return_verts=False,
        )
        return out.joints[:, : self.cfg.fit_joint_count]  # (T, 22, 3)

    def fit_clip(self, gt_joints_3d: np.ndarray, fps: float) -> dict:
        """
        Args:
            gt_joints_3d: (T, 22, 3) float array — BVH-derived 3D joint world positions
                          (first 22 SMPL body joints). Y-up, metres.
            fps:          input fps
        Returns dict with fit parameters and forward-pass joints, all as numpy.
        """
        assert gt_joints_3d.ndim == 3 and gt_joints_3d.shape[1] == self.cfg.fit_joint_count
        T = gt_joints_3d.shape[0]

        model = self._make_model(batch_size=T)

        gt = torch.as_tensor(gt_joints_3d, dtype=torch.float32, device=self.device)

        # Parameters (learnable)
        global_orient = torch.zeros(T, 3, device=self.device, requires_grad=True)
        body_pose = torch.zeros(T, 21 * 3, device=self.device, requires_grad=True)
        betas = torch.zeros(1, self.cfg.num_betas, device=self.device, requires_grad=True)
        # Initialise transl with the pelvis target minus SMPL-X rest pelvis offset (-0.35m Y).
        with torch.no_grad():
            rest_out = model(betas=torch.zeros(1, self.cfg.num_betas, device=self.device))
            pelvis_rest = rest_out.joints[0, 0].clone()  # (3,)
        transl_init = gt[:, 0] - pelvis_rest.unsqueeze(0)
        transl = transl_init.detach().clone().requires_grad_(True)

        # ----- Stage A: positional-only, fast convergence -----
        loss_A = Smpl3DFittingLoss(weights=Smpl3DLossWeights(
            data=1.0, shape=0.005, angle=0.0, smoothness_pose=0.0, smoothness_transl=0.0,
        ))
        opt_A = torch.optim.Adam([global_orient, body_pose, transl, betas], lr=self.cfg.lr_init)
        print(f"[fitter] Stage A: {self.cfg.n_iters_init} iters, lr={self.cfg.lr_init}")
        for it in range(self.cfg.n_iters_init):
            opt_A.zero_grad()
            pred = self._forward_joints(model, global_orient, body_pose, transl, betas)
            total, logs = loss_A(pred, gt, betas, body_pose, transl)
            total.backward()
            opt_A.step()
            if it % self.cfg.log_every == 0 or it == self.cfg.n_iters_init - 1:
                print(f"  A it={it:4d}  total={logs['total']:.4f}  data={logs['data']:.4f}")

        # ----- Stage B: priors + smoothness -----
        loss_B = Smpl3DFittingLoss(weights=Smpl3DLossWeights())  # defaults
        opt_B = torch.optim.Adam([global_orient, body_pose, transl, betas], lr=self.cfg.lr_refine)
        print(f"[fitter] Stage B: {self.cfg.n_iters_refine} iters, lr={self.cfg.lr_refine}")
        for it in range(self.cfg.n_iters_refine):
            opt_B.zero_grad()
            pred = self._forward_joints(model, global_orient, body_pose, transl, betas)
            total, logs = loss_B(pred, gt, betas, body_pose, transl)
            total.backward()
            opt_B.step()
            if it % self.cfg.log_every == 0 or it == self.cfg.n_iters_refine - 1:
                print(
                    f"  B it={it:4d}  total={logs['total']:.4f}  data={logs['data']:.4f}"
                    f"  smooth_pose={logs['smooth_pose']:.4f}  shape={logs['shape']:.4f}"
                )

        with torch.no_grad():
            # Forward with transl=0 so joints are in SMPL-X model frame.
            # SONIC's smpl_filtered schema stores joints WITHOUT transl applied
            # (pelvis ends up at ~(0, -0.35, 0), the SMPL-X rest pelvis offset),
            # and pairs it with transl separately. World position = joints + transl.
            final = model(
                betas=betas.expand(T, -1),
                global_orient=global_orient,
                body_pose=body_pose,
                transl=torch.zeros_like(transl),
                return_verts=False,
            )
            all_joints = final.joints[:, : self.cfg.fit_joint_count].detach().cpu().numpy()
            hand_proxy = final.joints[:, 20:22].detach().cpu().numpy()
            smpl_joints_24 = np.concatenate([all_joints, hand_proxy], axis=1)  # (T, 24, 3)

        return {
            "global_orient": global_orient.detach().cpu().numpy(),       # (T, 3)
            "body_pose": body_pose.detach().cpu().numpy(),               # (T, 63)
            "transl": transl.detach().cpu().numpy(),                     # (T, 3)
            "betas": betas.detach().cpu().numpy()[0],                    # (10,)
            "smpl_joints": smpl_joints_24,                               # (T, 24, 3)
            "fps": float(fps),
        }
