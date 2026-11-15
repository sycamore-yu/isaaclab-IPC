"""Isaac Lab registry/Hydra configuration for the standard RSL-RL runner."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@configclass
class MPCActorCfg(RslRlMLPModelCfg):
    class_name = "isaaclab_ipc.rl.actor:MPCActor"
    hidden_dims = [64, 64]
    activation = "elu"
    obs_normalization = False
    qp_iterations = 16384
    compile_kernels = True
    # MPCActor supplies its fixed Gaussian/point-mass distribution.
    distribution_cfg = None
    planner_params = None
    thrust_limits = (0.08 * 9.81 / 0.7, 0.9 * 9.81 / 0.7)
    rate_limit = None


@configclass
class MPCTrainCfg(RslRlOnPolicyRunnerCfg):
    seed = 0
    device = "cuda:0"
    num_steps_per_env = 64
    max_iterations = 32
    save_interval = 8
    init_at_random_ep_len = False
    experiment_name = "ipc_acmpc"
    logger = "tensorboard"
    check_for_nan = True
    obs_groups = {"actor": ["policy"], "critic": ["policy"]}
    actor = MPCActorCfg()
    critic = RslRlMLPModelCfg(hidden_dims=[64, 64], activation="elu", obs_normalization=False)
    algorithm = RslRlPpoAlgorithmCfg(
        class_name="isaaclab_ipc.rl.ppo:MPCPPO",
        num_learning_epochs=2,
        num_mini_batches=4,
        learning_rate=1e-4,
        schedule="fixed",
        desired_kl=None,
        clip_param=0.2,
        gamma=0.999,
        lam=0.99,
        entropy_coef=0.0,
        value_loss_coef=1.0,
        max_grad_norm=1.0,
        use_clipped_value_loss=True,
    )


@configclass
class CTBRActorCfg(RslRlMLPModelCfg):
    class_name = "isaaclab_ipc.rl.ctbr_actor:CTBRMPCActor"
    hidden_dims = [64, 64]
    activation = "elu"
    obs_normalization = False
    distribution_cfg = None
    cost_nodes = 5
    prediction_dt = 0.1
    model_config = None
    progress_config = None
    compile_kernels = True
    qp_iterations = 4096
    sqp_iterations = 25
    sqp_tolerance = 1e-4
    sqp_linesearch = "none"
    admm_tolerance = 1e-5
    qp_linear_solver = "cudss_inverse_chunk"


@configclass
class CTBRTrainCfg(MPCTrainCfg):
    experiment_name = "acmpc_native_ctbr"
    actor = CTBRActorCfg()
