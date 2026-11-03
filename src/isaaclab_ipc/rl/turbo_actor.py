"""Native Torch SQP actor with the existing IPC jerk/SFC task contract."""

from dataclasses import asdict, fields

import torch

from isaaclab_ipc.turbompc import ADMMConfig, SQPConfig, TorchMPC
from isaaclab_ipc.turbompc.ipc import IPCQPBuilder

from .actor import MPCActor


class TurboMPCActor(MPCActor):
    """Keep the seven IPC cost outputs and execute the native solver in forward."""

    def __init__(self, *args, compile_kernels=True, qp_iterations=16384, **kwargs):
        super().__init__(*args, qp_iterations=qp_iterations, **kwargs)
        self.ipc_problem = IPCQPBuilder()
        self.compile_kernels = bool(compile_kernels)
        self.qp_iterations = int(qp_iterations)
        self._native_solver = None
        self.native_solve_calls = 0
        self.last_native_result = None

    @property
    def native_solver(self):
        if self._native_solver is None:
            self._native_solver = TorchMPC(
                self.ipc_problem.dynamics(),
                SQPConfig(max_iter=15, tolerance=1e-6, linesearch="none", compile_kernels=self.compile_kernels),
                # IPC position weights are O(1e3). A relative ADMM threshold can
                # stop above the absolute SQP stationarity target on active SFCs.
                ADMMConfig(max_iter=self.qp_iterations, eps_abs=1e-7, eps_rel=0.0),
            )
        return self._native_solver

    def _solve_context(self, obs, costs):
        state = obs["mpc_state"].double()
        state = torch.cat((state[:, :3], state[:, 3:].clamp(self.lower, self.upper)), -1)
        valid = obs["sfc_valid"].flatten().bool()
        # Keep an exact zero derivative when every row takes the recovery branch.
        solution = costs.double().sum(-1, keepdim=True).expand(-1, 45) * 0
        solved = torch.zeros_like(valid)
        primal = state.new_full((len(state),), torch.inf)
        dual = primal.clone()
        iterations = 0
        self.last_native_result = None
        if valid.any():
            selected = {name: value[valid] for name, value in obs.items()}
            data, guess = self.ipc_problem(selected, costs[valid])
            result = self.native_solver.solve(data, guess)
            self.native_solve_calls += 1
            self.last_native_result = result
            solution[valid] = result.controls[:, :-1].flatten(1)
            solved[valid] = result.status == 0
            primal[valid] = result.errors[:, 1:].amax(-1)
            dual[valid] = result.errors[:, 0]
            iterations = result.admm.iterations.max()
        return solution, solved, primal, dual, iterations, state

    def problem_snapshot(self, obs, costs):
        data, guess = self.ipc_problem(obs, costs)
        return {**{field.name: getattr(data, field.name) for field in fields(data)}, "initial_guess": guess}

    def close(self):
        if self._native_solver is not None:
            self._native_solver.close()
            self._native_solver = None
        self.last_native_result = None

    def solver_metadata(self):
        result = {
            "backend": "turbompc-cuda",
            "implementation_file": __file__,
            "dynamics": "ipc-jerk",
            "horizon": 15,
            "prediction_dt": 0.1,
            "native_solve_calls": self.native_solve_calls,
            "torch_version": torch.__version__,
            "torch_cuda": torch.version.cuda,
        }
        if self._native_solver is not None:
            result["sqp"] = asdict(self._native_solver.config)
            result["admm"] = asdict(self._native_solver.admm.config)
        return result
