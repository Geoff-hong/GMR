"""Differentiable rotation conversions used by the SMPL-X fitter.

We optimise joint rotations in the 6-D representation from
Zhou et al. 2019 ("On the Continuity of Rotation Representations in Neural
Networks") because axis-angle has a discontinuous wrap at |theta|=pi, which
lets Adam drift into wrapped, physically implausible solutions such as a knee
with axis-angle magnitude ~3.9 rad (~222 deg).

The 6-D representation is the first two columns of a 3x3 rotation matrix.
It is mapped back to SO(3) via Gram-Schmidt on those two columns and is
globally continuous. We convert to axis-angle only at output time for
compatibility with the SONIC pkl schema.

All functions are torch-native and differentiable.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


EPS = 1e-7


def rot6d_to_matrix(rot6d: torch.Tensor) -> torch.Tensor:
    """Convert a 6-D rotation representation to a 3x3 rotation matrix.

    Zhou et al. (2019) representation: `rot6d[..., :3]` is the first column
    of the rotation matrix (before normalisation) and `rot6d[..., 3:]` is a
    vector from which we Gram-Schmidt orthogonalise the second column.
    The third column is the cross product, guaranteeing a right-handed rotation.

    Args:
        rot6d: tensor of shape (..., 6)
    Returns:
        tensor of shape (..., 3, 3)
    """
    a1 = rot6d[..., 0:3]
    a2 = rot6d[..., 3:6]
    b1 = F.normalize(a1, dim=-1, eps=EPS)
    # remove b1 projection from a2
    dot = (b1 * a2).sum(dim=-1, keepdim=True)
    b2 = F.normalize(a2 - dot * b1, dim=-1, eps=EPS)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)  # last dim = columns


def matrix_to_rot6d(R: torch.Tensor) -> torch.Tensor:
    """Inverse of `rot6d_to_matrix`: the first two columns of `R` stacked.

    Args:
        R: tensor of shape (..., 3, 3)
    Returns:
        tensor of shape (..., 6)
    """
    return R[..., :, 0:2].reshape(*R.shape[:-2], 6)


def axis_angle_to_matrix(aa: torch.Tensor) -> torch.Tensor:
    """Rodrigues' formula: (..., 3) axis-angle -> (..., 3, 3) rotation matrix."""
    angle = aa.norm(dim=-1, keepdim=True)
    safe = angle + EPS
    axis = aa / safe  # (..., 3)
    x, y, z = axis.unbind(dim=-1)
    s = torch.sin(angle.squeeze(-1))
    c = torch.cos(angle.squeeze(-1))
    C = 1 - c
    # Each rotation matrix element via Rodrigues
    R = torch.stack([
        torch.stack([c + x * x * C,     x * y * C - z * s, x * z * C + y * s], dim=-1),
        torch.stack([y * x * C + z * s, c + y * y * C,     y * z * C - x * s], dim=-1),
        torch.stack([z * x * C - y * s, z * y * C + x * s, c + z * z * C],     dim=-1),
    ], dim=-2)
    # Near-zero angle fallback: identity
    small = angle.squeeze(-1) < EPS
    if small.any():
        eye = torch.eye(3, device=aa.device, dtype=aa.dtype).expand_as(R)
        R = torch.where(small.unsqueeze(-1).unsqueeze(-1), eye, R)
    return R


def matrix_to_axis_angle(R: torch.Tensor) -> torch.Tensor:
    """Stable conversion (..., 3, 3) -> (..., 3) axis-angle.

    Uses `acos((trace-1)/2)` with clamp and an `atan2`-based near-pi branch
    to avoid numerical issues at the `theta == pi` discontinuity. Output is
    always canonicalised with `theta in [0, pi]`.
    """
    # Trace and clamped cos
    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    cos_t = (trace - 1) * 0.5
    cos_t = torch.clamp(cos_t, -1.0 + 1e-6, 1.0 - 1e-6)
    theta = torch.acos(cos_t)                           # (...,), in [0, pi]

    # Skew part gives 2*sin(theta)*axis
    r = torch.stack([
        R[..., 2, 1] - R[..., 1, 2],
        R[..., 0, 2] - R[..., 2, 0],
        R[..., 1, 0] - R[..., 0, 1],
    ], dim=-1)                                           # (..., 3)
    sin_t = torch.sin(theta).unsqueeze(-1)
    axis = r / (2 * sin_t + EPS)

    # Near pi: sin(theta) ~ 0; use diagonal method. |theta - pi| < small tolerance
    near_pi = (theta > torch.pi - 1e-3)
    if near_pi.any():
        # Extract axis from (R+I)/2 whose columns are (axis_x^2, x*y, x*z), etc.
        diag = torch.stack([R[..., 0, 0], R[..., 1, 1], R[..., 2, 2]], dim=-1)
        axis_pi = torch.sqrt(torch.clamp((diag + 1) * 0.5, min=0.0))
        # sign pattern from off-diagonal
        sgn = torch.sign(torch.stack([
            R[..., 2, 1] - R[..., 1, 2],
            R[..., 0, 2] - R[..., 2, 0],
            R[..., 1, 0] - R[..., 0, 1],
        ], dim=-1))
        sgn = torch.where(sgn == 0, torch.ones_like(sgn), sgn)
        axis_pi = axis_pi * sgn
        axis = torch.where(near_pi.unsqueeze(-1), axis_pi, axis)

    return axis * theta.unsqueeze(-1)


def rot6d_to_axis_angle(rot6d: torch.Tensor) -> torch.Tensor:
    """Compose the two converters."""
    return matrix_to_axis_angle(rot6d_to_matrix(rot6d))


def axis_angle_to_rot6d(aa: torch.Tensor) -> torch.Tensor:
    return matrix_to_rot6d(axis_angle_to_matrix(aa))


def identity_rot6d(*shape, device=None, dtype=torch.float32) -> torch.Tensor:
    """Return a 6-D representation of identity rotations, shape (*shape, 6)."""
    r = torch.zeros(*shape, 6, device=device, dtype=dtype)
    r[..., 0] = 1.0   # first column = (1, 0, 0)
    r[..., 4] = 1.0   # second column = (0, 1, 0)
    return r


def geodesic_distance(R1: torch.Tensor, R2: torch.Tensor) -> torch.Tensor:
    """Geodesic distance on SO(3): angle of the relative rotation.

    Returns: (...,) tensor of angles in [0, pi].
    """
    R = torch.matmul(R1.transpose(-1, -2), R2)
    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    cos_t = torch.clamp((trace - 1) * 0.5, -1.0 + 1e-6, 1.0 - 1e-6)
    return torch.acos(cos_t)
