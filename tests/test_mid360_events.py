from types import SimpleNamespace

import numpy as np
import torch

from isaaclab_ipc.mid360 import Mid360PatternCfg, update_mid360_directions
from isaaclab_ipc.tasks.events import advance_scan


def test_random_window_updates_only_selected_environments(tmp_path):
    path = tmp_path / "directions.npy"
    np.save(path, np.arange(1, 181).reshape(60, 3).astype(np.float32))
    cfg = Mid360PatternCfg(raydirs_path=str(path), rays_per_scan=20, ray_decimation=2)
    buf = torch.zeros(3, 10, 3)
    sensor = SimpleNamespace(cfg=SimpleNamespace(pattern_cfg=cfg), ray_directions=SimpleNamespace(torch=buf))
    env = SimpleNamespace(
        scene=SimpleNamespace(sensors={"mid360": sensor}),
        device="cpu",
        mid360_rng=torch.Generator().manual_seed(0),
        mid360_window_start=torch.full((3,), -1),
    )
    update_mid360_directions(env, torch.tensor([0, 2]))
    start = env.mid360_window_start[0]
    torch.testing.assert_close(buf[0], cfg.load_table("cpu")[start : start + 20 : 2])
    torch.testing.assert_close(buf[0], buf[2])
    assert not buf[1].any() and env.mid360_window_start[1] == -1


def test_scan_event_uses_per_environment_reset_clock(monkeypatch):
    calls = []
    monkeypatch.setattr("isaaclab_ipc.mid360.update_mid360_directions", lambda env, ids: calls.append(ids.tolist()))
    advance_scan(SimpleNamespace(elapsed=torch.tensor([0.09, 0.1, 0.2])), None)
    assert calls == [[1, 2]]


def test_ctbr_ten_hertz_has_one_scan_window_per_control_step(monkeypatch):
    calls = []
    monkeypatch.setattr("isaaclab_ipc.mid360.update_mid360_directions", lambda env, ids: calls.append(ids.tolist()))
    for step in range(1, 11):
        advance_scan(SimpleNamespace(elapsed=torch.tensor([step * 0.1])), None)
    assert calls == [[0]] * 10
