import numpy as np
import pytest
import torch

from isaaclab_ipc.mid360 import Mid360PatternCfg, mid360_config, mid360_pattern


def test_mid360_table_normalization_and_decimation(tmp_path):
    path = tmp_path / "dirs.npy"
    np.save(path, np.arange(1, 61).reshape(20, 3).astype(np.float32))
    cfg = Mid360PatternCfg(raydirs_path=str(path), rays_per_scan=10, ray_decimation=2)
    a = cfg.load_table("cpu")
    torch.testing.assert_close(a.norm(dim=-1), torch.ones(20))
    starts, dirs = mid360_pattern(cfg, "cpu")
    assert dirs.shape == (5, 3) and not starts.any()
    torch.testing.assert_close(dirs, a[:10:2])
    assert mid360_config().max_distance == 40
    np.save(path, np.zeros((20, 3)))
    with pytest.raises(ValueError, match="invalid"):
        Mid360PatternCfg(raydirs_path=str(path), rays_per_scan=10).load_table("cpu")


def test_official_multimesh_configuration():
    from isaaclab.sensors import MultiMeshRayCasterCfg

    for scenario in ("forest", "avoid"):
        cfg = mid360_config(scenario)
        assert type(cfg) is MultiMeshRayCasterCfg
        assert cfg.update_period == 0.1 and cfg.ray_alignment == "base"
        assert cfg.offset.pos == (0.0, 0.0, 0.0) and cfg.offset.rot == (0.0, 0.0, 0.0, 1.0)
        assert cfg.pattern_cfg.rays_per_scan == 20000 and cfg.pattern_cfg.ray_decimation == 1
        assert all(t.prim_expr.startswith("{ENV_REGEX_NS}/") for t in cfg.mesh_prim_paths)


def test_task_configuration_exposes_official_sensors():
    from isaaclab.sensors import ContactSensorCfg, MultiMeshRayCasterCfg, PvaCfg

    from isaaclab_ipc.tasks.env import make_cfg

    cfg = make_cfg()
    assert type(cfg.scene.mid360) is MultiMeshRayCasterCfg
    assert type(cfg.scene.pva) is PvaCfg and type(cfg.scene.contacts) is ContactSensorCfg
    assert "mid360" not in vars(make_cfg(scenario="empty").scene)
