"""RSL-RL PPO with rollout-only MPC diagnostics delivery to the task."""

from rsl_rl.algorithms import PPO


class MPCPPO(PPO):
    @staticmethod
    def construct_algorithm(obs, env, cfg, device):
        algorithm = PPO.construct_algorithm(obs, env, cfg, device)
        algorithm.mpc_env = env.unwrapped
        return algorithm

    def act(self, obs):
        actions = super().act(obs)
        self.mpc_env.record_mpc_result(self.actor.last_result)
        return actions
