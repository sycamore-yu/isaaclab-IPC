"""Standard structured trajectory recording, separate from Rerun visualization."""

from pathlib import Path

import torch
from isaaclab.managers import DatasetExportMode, RecorderManager, RecorderManagerBaseCfg, RecorderTerm, RecorderTermCfg
from isaaclab.utils import configclass


class FlightRecorder(RecorderTerm):
    def record_pre_step(self):
        env = self._env
        values = {"action": env.controller.command}
        context = getattr(env, "current_context", None)
        if context is not None:
            values["mpc_context"] = context
        return "inputs", values

    def record_post_step(self):
        env = self._env
        pose, twist = env.body_state()
        success = env.extras["success"] & ~env.hit
        if hasattr(env, "command_violations"):
            success &= torch.as_tensor(env.command_violations == 0, device=env.device)
        env.recorder_manager.set_success_to_episodes(None, success[:, None])
        return "outputs", dict(
            pose=pose,
            twist=twist,
            acceleration=env.acceleration(),
            time=env.elapsed,
            success=success,
            collision=env.hit,
            physical_contact=env.extras["physical_contact"],
        )


class RerunRecorder(RecorderTerm):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.recording = None

    def record_pre_step(self):
        if self.recording is None:
            from isaaclab_ipc.recording import ForestRecording

            self.recording = ForestRecording(self._env.cfg.rerun_path, self._env, self._env.cfg.rerun_label)
        self.recording.plan()
        return None, None

    def record_post_step(self):
        self.recording.state()
        return None, None

    def close(self, dataset_path):
        if self.recording is not None:
            self.recording.close()


@configclass
class FlightRecorderCfg(RecorderManagerBaseCfg):
    flight = None
    rerun = None
    dataset_filename = "flight"
    export_in_close = True


def recorder_config(cfg):
    if not (cfg.recorder_dir or cfg.rerun_path):
        return None
    if cfg.recorder_dir and (Path(cfg.recorder_dir) / "flight.hdf5").exists():
        raise FileExistsError(Path(cfg.recorder_dir) / "flight.hdf5")
    if cfg.rerun_path and Path(cfg.rerun_path).exists():
        raise FileExistsError(cfg.rerun_path)
    return FlightRecorderCfg(
        dataset_export_dir_path=cfg.recorder_dir or ".",
        dataset_export_mode=DatasetExportMode.EXPORT_ALL if cfg.recorder_dir else DatasetExportMode.EXPORT_NONE,
        flight=RecorderTermCfg(class_type=FlightRecorder) if cfg.recorder_dir else None,
        rerun=RecorderTermCfg(class_type=RerunRecorder) if cfg.rerun_path else None,
    )


class FlightRecorderManager(RecorderManager):
    def close(self):
        # DirectRLEnv closes before scene teardown; the base destructor closes again.
        if not getattr(self, "_closed", False):
            super().close()
            self._closed = True
