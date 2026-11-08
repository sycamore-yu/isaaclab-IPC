"""AC-MPC actor: observations -> 28N costs -> native nonlinear CTBR MPC."""

from dataclasses import asdict, fields, replace
from pathlib import Path

import torch
from rsl_rl.models import MLPModel
from rsl_rl.modules import MLP

from isaaclab_ipc.assets.quadrotor import MASS
from isaaclab_ipc.turbompc import ADMMConfig, SQPConfig, TorchMPC
from isaaclab_ipc.turbompc.ctbr import CTBRConfig, CTBRQPBuilder
from isaaclab_ipc.turbompc.native import _extension
from isaaclab_ipc.turbompc.quadrotor import controls_to_actions


DEFAULT_CTBR_QP_LINEAR_SOLVER = "cudss_inverse_chunk"


class CTBRMPCActor(MLPModel):
    """A frozen observation determines both the MPC problem and its warm start.

    No online planner, cross-batch mutable warm start, jerk model or recovery
    controller is used by this policy. Failed solver statuses remain visible;
    they are not replaced by a second controller.
    """

    def __init__(
        self, obs, obs_groups, obs_set, output_dim, hidden_dims=(64, 64), activation="elu",
        obs_normalization=False, distribution_cfg=None, cost_nodes=5, prediction_dt=0.1,
        model_config=None, compile_kernels=True, qp_iterations=4096, sqp_iterations=25,
        sqp_tolerance=1e-4, admm_tolerance=1e-5, sqp_linesearch="none",
        qp_linear_solver=DEFAULT_CTBR_QP_LINEAR_SOLVER,
    ):
        if output_dim != 4 or obs_normalization:
            raise ValueError("CTBRMPCActor requires four physical actions and fixed feature scaling")
        distribution_cfg = distribution_cfg or dict(
            class_name="GaussianDistribution", init_std=0.03, std_type="log", learn_std=False
        )
        super().__init__(
            obs, obs_groups, obs_set, output_dim, hidden_dims, activation, obs_normalization, distribution_cfg
        )
        config = (
            CTBRConfig(cost_nodes=cost_nodes, dt=prediction_dt, mass=MASS)
            if model_config is None else CTBRConfig(**model_config)
        )
        self.ctbr_problem = CTBRQPBuilder(config, compile_kernels=compile_kernels)
        self.mlp = MLP(self.obs_dim, config.cost_dim, hidden_dims, activation)
        torch.nn.init.zeros_(self.mlp[-1].weight)
        torch.nn.init.zeros_(self.mlp[-1].bias)
        self.register_buffer("physical_action_lower", self.ctbr_problem.action_lower.float())
        self.register_buffer("physical_action_upper", self.ctbr_problem.action_upper.float())
        # Hydra parses an unquoted CLI `sqp_linesearch=none` as Python None.
        sqp_linesearch = "none" if sqp_linesearch is None else sqp_linesearch
        if sqp_linesearch not in ("none", "merit"):
            raise ValueError("CTBR SQP linesearch must be none or merit")
        self.sqp_config = SQPConfig(
            max_iter=sqp_iterations, tolerance=sqp_tolerance, linesearch=sqp_linesearch,
            compile_kernels=compile_kernels, require_qp_convergence=False,
        )
        self.admm_config = ADMMConfig(
            max_iter=qp_iterations, eps_abs=admm_tolerance, eps_rel=0.0,
            linear_solver=qp_linear_solver,
        )
        self.record_checkpoint_linear_solver(None, resolution="actor_configuration")
        self._model = self.ctbr_problem.dynamics()
        if compile_kernels:
            self._action_postprocess = torch.compile(
                self._action_postprocess, fullgraph=True, dynamic=False,
                options={"triton.cudagraphs": False},
            )
        self._native_solver = None
        self.last_native_result = None
        self.last_result = None
        self.native_solve_calls = 0
        self.fixed_cost = False

    @property
    def native_solver(self):
        if self._native_solver is None:
            self._native_solver = TorchMPC(
                self._model, self.sqp_config, self.admm_config,
            )
        return self._native_solver

    def get_extra_state(self):
        return {
            "contract": "ctbr-node-costs-v2",
            "cost_outputs": self.ctbr_problem.config.cost_dim,
            "cost_layout": "all_node_diagonal14_then_all_node_linear14",
            "policy_features": self.obs_dim,
            "model": asdict(self.ctbr_problem.config),
            "sqp_linesearch": self.sqp_config.linesearch,
        }

    def set_extra_state(self, state):
        # Earlier node-cost checkpoints used merit and did not record this field.
        # Loading a checkpoint preserves its algorithm even if defaults change.
        linesearch = state.get("sqp_linesearch", "merit")
        if linesearch not in ("none", "merit"):
            raise ValueError("CTBR checkpoint SQP linesearch must be none or merit")
        expected = {**self.get_extra_state(), "sqp_linesearch": linesearch}
        if {**state, "sqp_linesearch": linesearch} != expected:
            raise ValueError("CTBR checkpoint model, units, cost contract or features differ")
        if self.sqp_config.linesearch != linesearch:
            self.close()
            self.sqp_config = replace(self.sqp_config, linesearch=linesearch)

    def _action_postprocess(self, controls, initial_state):
        """Return a differentiable mean and physical diagnostics without gradients."""
        mass = self.ctbr_problem.config.mass
        mean = controls_to_actions(controls[:, 0], mass).float()
        mean = mean.clamp(self.physical_action_lower, self.physical_action_upper)
        with torch.no_grad():
            applied = mean.double().clone()
            applied[:, 0] *= mass
            current = torch.cat((initial_state, applied), -1)
            predicted = self._model(current)
            acceleration = self._model.acceleration(current)
        return mean, applied, predicted, acceleration

    def forward(self, obs, masks=None, hidden_state=None, stochastic_output=False):
        costs = None if self.fixed_cost else self.mlp(self.get_latent(obs))
        data, guess = self.ctbr_problem(obs, costs)
        result = self.native_solver.solve(data, guess)
        self.last_native_result = result
        self.native_solve_calls += 1
        controls = result.controls
        mean, applied, predicted, acceleration = self._action_postprocess(
            controls, data.initial_state,
        )
        self.distribution.update(mean)
        valid = obs["sfc_valid"].flatten().bool()
        solved = (result.status == 0) & valid
        self.last_result = dict(
            model="ctbr",
            state=data.initial_state.detach(),
            solution=controls[:, :-1].flatten(1).detach(),
            controls=controls[:, :-1].detach(),
            trajectory=result.trajectory.detach(),
            prediction_positions=result.states[:, 1:, :3].detach(),
            predicted=predicted,
            applied_control=applied,
            predicted_acceleration=acceleration,
            solved=solved,
            braking=torch.zeros_like(solved),
            primal=result.errors[:, 1:3].amax(-1),
            dual=result.errors[:, 0],
            normal_cone=result.normal_cone_error,
            raw_qp_status=result.admm.status,
            sqp_status=result.status,
            iterations=result.admm.iterations.max(),
            sqp_iterations=result.iterations,
            qp_iterations_total=result.qp_iterations_total,
            mean=mean.detach(),
            sfc_valid=valid,
        )
        return self.distribution.sample() if stochastic_output else mean

    def problem_snapshot(self, obs, costs):
        data, guess = self.ctbr_problem(obs, costs)
        return {
            **{field.name: getattr(data, field.name) for field in fields(data)},
            "initial_guess": guess,
            "acceleration_lower": self.ctbr_problem.acceleration_lower,
            "acceleration_upper": self.ctbr_problem.acceleration_upper,
        }

    def close(self):
        if self._native_solver is not None:
            self._native_solver.close()
            self._native_solver = None
        self.last_native_result = None

    def record_checkpoint_linear_solver(self, checkpoint_actor_config, *, resolution):
        """Record selection provenance without changing a checkpoint or solver."""
        present = checkpoint_actor_config is not None
        field_present = present and "qp_linear_solver" in checkpoint_actor_config
        recorded = checkpoint_actor_config.get("qp_linear_solver") if present else None
        effective = self.admm_config.linear_solver
        self._linear_solver_selection = {
            "effective": effective,
            "resolution": resolution,
            "checkpoint_present": present,
            "checkpoint_field_present": field_present,
            "checkpoint_linear_solver": recorded,
            "differs_from_recorded_checkpoint": effective != recorded if field_present else None,
        }

    def solver_metadata(self):
        cfg = self.ctbr_problem.config
        return {
            "backend": "turbompc-cuda",
            "implementation_file": __file__,
            "dynamics": "ac-mpc-ctbr",
            "state_order": "position_world,quaternion_wxyz,velocity_world",
            "control_units": "total_thrust_N,body_rates_rad_s",
            "action_units": "specific_thrust_m_s2,body_rates_rad_s",
            "cost_nodes": cfg.cost_nodes,
            "prediction_intervals": cfg.horizon,
            "prediction_window_s": cfg.horizon * cfg.dt,
            "cost_outputs": cfg.cost_dim,
            "cost_coordinates": "reference_centered",
            "hard_constraints": ["input_bounds", "velocity_bounds", "sfc"],
            "native_solve_calls": self.native_solve_calls,
            "native_extension_path": str(Path(_extension().__file__).resolve()) if self.native_solve_calls > 0 else None,
            "torch_version": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "model": asdict(cfg),
            "sqp": asdict(self.sqp_config),
            "admm": asdict(self.admm_config),
            "linear_solver_selection": dict(self._linear_solver_selection),
            "polish": None,
        }

    def as_jit(self):
        raise NotImplementedError("The saved CTBR actor needs its native CUDA solver")

    def as_onnx(self, verbose=False):
        raise NotImplementedError("The saved CTBR actor needs its native CUDA solver")
