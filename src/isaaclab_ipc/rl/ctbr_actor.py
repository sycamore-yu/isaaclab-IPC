"""Native nonlinear CTBR policy with immutable observations and IPC recovery."""

from dataclasses import asdict, fields

import torch

from isaaclab_ipc.assets.quadrotor import MASS
from isaaclab_ipc.execution import brake_jerk, next_braking
from isaaclab_ipc.turbompc import ADMMConfig, SQPConfig, TorchMPC
from isaaclab_ipc.turbompc.ctbr import CTBRConfig, CTBRQPBuilder
from isaaclab_ipc.turbompc.ctbr_polish import CTBRPolisher
from isaaclab_ipc.turbompc.quadrotor import controls_to_actions

from .actor import MPCActor, flatness_torch


class CTBRMPCActor(MPCActor):
    """Learn CTBR cost parameters; execute total thrust and body rates directly.

    The seven outputs are [position, velocity, attitude, input, bias_xyz].
    This contract and the additional body-rate features require a CTBR checkpoint.
    The inherited IPC tensors are used only by the established braking controller.
    """

    def __init__(
        self,
        *args,
        compile_kernels=True,
        qp_iterations=32768,
        sqp_iterations=60,
        sqp_tolerance=1e-7,
        admm_tolerance=1e-8,
        polish_iterations=4,
        polish_tolerance=1e-11,
        polish_linear_refinements=2,
        **kwargs,
    ):
        super().__init__(*args, qp_iterations=qp_iterations, **kwargs)
        self.ctbr_problem = CTBRQPBuilder(CTBRConfig(mass=MASS))
        # The physical task consumes float32. Round limits into the admissible
        # interval so conversion of an active float64 bound cannot cross it.
        low, high = self.ctbr_problem.action_lower, self.ctbr_problem.action_upper
        low32, high32 = low.float(), high.float()
        low32 = torch.where(low32.double() < low, torch.nextafter(low32, torch.full_like(low32, torch.inf)), low32)
        high32 = torch.where(
            high32.double() > high, torch.nextafter(high32, torch.full_like(high32, -torch.inf)), high32
        )
        self.register_buffer("physical_action_lower", low32)
        self.register_buffer("physical_action_upper", high32)
        self.sqp_config = SQPConfig(
            max_iter=sqp_iterations,
            tolerance=sqp_tolerance,
            linesearch="merit",
            compile_kernels=compile_kernels,
            require_qp_convergence=True,
        )
        self.admm_config = ADMMConfig(max_iter=qp_iterations, eps_abs=admm_tolerance, eps_rel=0.0)
        self.polisher = CTBRPolisher(
            max_iter=polish_iterations,
            tolerance=polish_tolerance,
            max_linear_refinements=polish_linear_refinements,
        )
        self._model = self.ctbr_problem.dynamics()
        self._native_solver = None
        self.native_solve_calls = 0
        self.last_native_result = None

    @property
    def native_solver(self):
        if self._native_solver is None:
            self._native_solver = TorchMPC(
                self._model,
                self.sqp_config,
                self.admm_config,
                constraints=self.ctbr_problem.path_constraints(),
                polisher=self.polisher,
            )
        return self._native_solver

    def get_extra_state(self):
        return {
            "contract": "ctbr-p-wxyz-v-total-newtons-body-rates-v1",
            "cost_heads": ("position", "velocity", "attitude", "input", "bias_x", "bias_y", "bias_z"),
            "policy_features": self.obs_dim,
            "model": asdict(self.ctbr_problem.config),
        }

    def set_extra_state(self, state):
        if state != self.get_extra_state():
            raise ValueError("CTBR checkpoint model, units, cost-head contract or observation features differ")

    def forward(self, obs, masks=None, hidden_state=None, stochastic_output=False):
        costs = self.mlp(self.get_latent(obs))
        state = obs["ctbr_state"].double()
        navigation = obs["mpc_state"].double()
        navigation = torch.cat((navigation[:, :3], navigation[:, 3:].clamp(self.lower, self.upper)), -1)
        if state.shape != (len(costs), 10) or not torch.isfinite(state).all():
            raise ValueError("CTBR requires a finite [position, quaternion WXYZ, world velocity] state")
        if ((state[:, 3:7].norm(dim=-1) - 1).abs() > 1e-4).any():
            raise ValueError("CTBR requires a near-unit physical quaternion")

        valid = obs["sfc_valid"].flatten().bool()
        horizon = self.ctbr_problem.config.horizon
        controls = costs.double().sum(-1)[:, None, None].expand(-1, horizon + 1, 4) * 0
        controls = controls + self.ctbr_problem.hover_control
        prediction = state.detach()[:, None].expand(-1, horizon + 1, -1).clone()
        solved = torch.zeros_like(valid)
        primal = state.new_full((len(state),), torch.inf)
        dual, normal_cone = primal.clone(), primal.clone()
        raw_qp_status = torch.full((len(state),), -1, device=state.device, dtype=torch.int32)
        iterations = 0
        self.last_native_result = None
        if valid.any():
            selected = {name: value[valid] for name, value in obs.items()}
            data, guess = self.ctbr_problem(selected, costs[valid])
            result = self.native_solver.solve(data, guess)
            self.native_solve_calls += 1
            self.last_native_result = result
            controls[valid] = result.controls
            prediction[valid] = result.states.detach()
            # Both the physical SQP check and the inner QP must converge before execution.
            solved[valid] = (result.status == 0) & (result.admm.status == 0)
            primal[valid] = result.errors[:, 1:3].amax(-1)
            dual[valid] = result.errors[:, 0]
            normal_cone[valid] = result.normal_cone_error
            raw_qp_status[valid] = result.admm.status
            iterations = result.admm.iterations.max()

        braking = next_braking(
            obs["braking"].flatten().bool(), solved, navigation[:, 3:6].norm(dim=-1), navigation[:, 6:].norm(dim=-1)
        )
        jerk = brake_jerk(navigation, self.lower, self.upper, self.jlow, self.jhigh, 0.1)
        backup_state = navigation @ self.a.T + jerk @ self.b.T
        backup = flatness_torch(
            backup_state[:, 6:], jerk, obs["rotation"].reshape(-1, 3, 3).double(), obs["yaw_target"].flatten().double()
        ).clamp(self.ctbr_problem.action_lower, self.ctbr_problem.action_upper)
        optimized = controls_to_actions(controls[:, 0], self.ctbr_problem.config.mass)
        mean = torch.where(braking[:, None], backup, optimized).float()
        mean = mean.clamp(self.physical_action_lower, self.physical_action_upper)
        self._backup = backup.float().clamp(self.physical_action_lower, self.physical_action_upper).detach()
        self._braking = braking
        self.distribution.update(mean)
        with torch.no_grad():
            applied = mean.double().clone()
            applied[:, 0] *= self.ctbr_problem.config.mass
            current = torch.cat((state, applied), -1)
            predicted = self._model(current)
            acceleration = self._model.acceleration(current)
        self.last_result = dict(
            model="ctbr",
            state=state.detach(),
            navigation_state=navigation.detach(),
            solution=controls[:, :-1].flatten(1).detach(),
            trajectory=torch.cat((prediction, controls.detach()), -1),
            prediction_positions=prediction[:, 1:, :3],
            predicted=predicted,
            applied_control=applied,
            predicted_acceleration=acceleration,
            solved=solved,
            braking=braking,
            primal=primal,
            dual=dual,
            normal_cone=normal_cone,
            raw_qp_status=raw_qp_status,
            iterations=iterations,
            mean=mean.detach(),
            sfc_valid=valid,
        )
        if stochastic_output:
            return torch.where(braking[:, None], self._backup, self.distribution.sample())
        return mean

    def problem_snapshot(self, obs, costs):
        data, guess = self.ctbr_problem(obs, costs)
        return {
            **{field.name: getattr(data, field.name) for field in fields(data)},
            "initial_guess": guess,
            "acceleration_lower": self.ctbr_problem.acceleration_lower,
            "acceleration_upper": self.ctbr_problem.acceleration_upper,
            "model_dt_mass_gravity": data.q.new_tensor(
                (self.ctbr_problem.config.dt, self.ctbr_problem.config.mass, self.ctbr_problem.config.gravity)
            ),
        }

    def close(self):
        if self._native_solver is not None:
            self._native_solver.close()
            self._native_solver = None
        self.last_native_result = None

    def solver_metadata(self):
        return {
            "backend": "turbompc-cuda",
            "implementation_file": __file__,
            "dynamics": "ac-mpc-ctbr",
            "state_order": "position_world,quaternion_wxyz,velocity_world",
            "control_units": "total_thrust_N,body_rates_rad_s",
            "action_units": "specific_thrust_m_s2,body_rates_rad_s",
            "native_solve_calls": self.native_solve_calls,
            "torch_version": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "model": asdict(self.ctbr_problem.config),
            "sqp": asdict(self.sqp_config),
            "admm": asdict(self.admm_config),
            "polish": {
                "max_iter": self.polisher.max_iter,
                "tolerance": self.polisher.tolerance,
                "max_linear_refinements": self.polisher.max_linear_refinements,
                "gradient_scope": "cost parameters D/E/q",
            },
        }
