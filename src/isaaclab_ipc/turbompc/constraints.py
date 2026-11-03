"""Fixed smooth stage constraints for native Torch SQP."""

import torch

from .dynamics import AutoDiffDynamics


class PathConstraints:
    """Represent ``lower <= function([state, control]) <= upper``.

    Bounds are fixed float64 vectors. ``stage_mask[T,m]`` disables individual
    rows, including constraints on terminal dummy controls. The function must
    be smooth on the evaluated trajectory; its physical parameters are fixed.
    Learned linear geometry continues to use ``MPCInputs.G/lower/upper``.
    """

    def __init__(self, function, *, lower, upper, stage_mask=None):
        if lower.ndim != 1 or not lower.numel() or upper.shape != lower.shape:
            raise ValueError("Path bounds must be nonempty vectors of the same shape")
        if lower.dtype != torch.float64 or upper.dtype != torch.float64 or lower.device != upper.device:
            raise ValueError("Path bounds must be float64 on the same device")
        if lower.requires_grad or upper.requires_grad:
            raise ValueError("Learnable path bounds are unsupported; physical bounds must be fixed")
        if torch.isnan(lower).any() or torch.isnan(upper).any() or (lower > upper).any():
            raise ValueError("Path bounds must be ordered and cannot contain NaN")
        if torch.isposinf(lower).any() or torch.isneginf(upper).any():
            raise ValueError("Infinite path bounds may only open a lower or upper side")
        if stage_mask is not None:
            if (
                stage_mask.ndim != 2
                or stage_mask.shape[0] < 1
                or stage_mask.shape[1] != lower.numel()
                or stage_mask.dtype != torch.bool
                or stage_mask.device != lower.device
            ):
                raise ValueError("stage_mask must be bool[T,m] on the bounds' device")
            stage_mask = stage_mask.detach().clone()
        owner = getattr(function, "__self__", function)
        if isinstance(owner, torch.nn.Module) and any(parameter.requires_grad for parameter in owner.parameters()):
            raise ValueError("Learnable path physics are unsupported")
        self.lower, self.upper = lower.detach().clone(), upper.detach().clone()
        self.stage_mask = stage_mask
        self._bounded = torch.isfinite(self.lower) | torch.isfinite(self.upper)
        self.map = AutoDiffDynamics(function, nx=lower.numel())

    def _check(self, z):
        if z.ndim != 3 or z.dtype != self.lower.dtype or z.device != self.lower.device:
            raise ValueError("Path trajectories must be float64[B,T,n] on the bounds' device")
        if self.stage_mask is not None and self.stage_mask.shape[0] != z.shape[1]:
            raise ValueError("stage_mask does not match the trajectory's node count")

    def enabled(self, z):
        self._check(z)
        mask = self._bounded if self.stage_mask is None else self.stage_mask & self._bounded
        return mask.expand(z.shape[0], z.shape[1], self.lower.numel())

    def bounds(self, z):
        enabled = self.enabled(z)
        lower = self.lower.expand_as(enabled).masked_fill(~enabled, -torch.inf)
        upper = self.upper.expand_as(enabled).masked_fill(~enabled, torch.inf)
        return lower, upper

    def __call__(self, z):
        self._check(z)
        return self.map(z)

    def linearize(self, z):
        self._check(z)
        return self.map.linearize(z)

    def violations(self, z):
        value = self(z)
        violation = torch.maximum(self.lower - value, value - self.upper).clamp_min(0)
        return torch.where(self.enabled(z), violation, 0.0)

    def normal_cone_residual(self, z, multipliers):
        # A fixed unit step checks dual feasibility/complementarity independently
        # of ADMM rho and the translated bounds of a local linearization.
        value = self(z)
        enabled = self.enabled(z)
        y = torch.where(enabled, multipliers, 0.0)
        projected = torch.clamp(value + y, min=self.lower, max=self.upper)
        return torch.where(enabled, (value - projected).abs(), 0.0)

    def active_sets(self, z, multipliers, eps_abs, eps_rel):
        value, jacobian = self.linearize(z)
        lower_finite, upper_finite = torch.isfinite(self.lower), torch.isfinite(self.upper)
        lower_scale = torch.where(lower_finite, self.lower.abs(), 0.0)
        upper_scale = torch.where(upper_finite, self.upper.abs(), 0.0)
        dl, du = value - self.lower, self.upper - value
        touching_lower = lower_finite & (dl <= eps_abs + eps_rel * lower_scale)
        touching_upper = upper_finite & (du <= eps_abs + eps_rel * upper_scale)
        prefer_upper = (multipliers > eps_abs) | ((multipliers.abs() <= eps_abs) & (du < dl))
        active_lower = touching_lower & (~touching_upper | ~prefer_upper)
        active_upper = touching_upper & (~touching_lower | prefer_upper)
        eligible = self.enabled(z) & (jacobian.abs().amax(-1) > 0)
        active_lower, active_upper = active_lower & eligible, active_upper & eligible
        return active_lower | active_upper, active_lower, active_upper

    def lagrangian_hessian(self, z, multipliers):
        enabled = self.enabled(z)
        y = torch.where(enabled, multipliers, 0.0)
        result = self.map.lagrangian_hessian(z, y)
        return torch.where(enabled.any(-1)[..., None, None], result, 0.0)
