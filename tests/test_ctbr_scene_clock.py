from types import SimpleNamespace

import pytest
import torch

from isaaclab_ipc.tasks.events import advance_scene


@pytest.mark.parametrize("step_dt", [0.01, 0.1, 0.3, 1 / 300])
def test_obstacle_clock_matches_discrete_millisecond_events(step_dt):
    """Compare the batched count with the original event-by-event definition."""
    elapsed = torch.tensor([0.0, 0.2, 0.4, 0.104, 0.1, 0.2])
    position = torch.tensor([[6.9, 0.3, 1.0], [6.9, 0.4, 1.0], [0.0, 0.5, 1.0],
                             [6.9, -0.2, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    origins = torch.arange(18).reshape(6, 3).float()
    pose = torch.cat((position + origins, torch.zeros(6, 4)), -1)
    default_obstacle_pose = torch.zeros(6, 7)
    default_obstacle_pose[:, 0], default_obstacle_pose[:, 2], default_obstacle_pose[:, 6] = 7.0, 2.0, 1.0
    writes = []
    obstacle = SimpleNamespace(
        data=SimpleNamespace(default_root_pose=SimpleNamespace(torch=default_obstacle_pose)),
        write_root_pose_to_sim_index=lambda **kwargs: writes.append(kwargs),
    )

    class Scene(dict):
        env_origins = origins

    env = SimpleNamespace(
        cfg=SimpleNamespace(scenario="avoid", obstacle_distance=1.0),
        body_state=lambda: (pose, None), scene=Scene(obstacle=obstacle),
        elapsed=elapsed.clone(), step_dt=step_dt, ids=torch.arange(6),
        obstacle_y=torch.tensor([1.5, 1.5, 4.0, 1.5, -0.5, 4.0]),
        target_y=torch.tensor([0.0, 0.0, 0.5, 0.0, 0.0, 0.5]),
        obstacle_active=torch.tensor([False, False, True, False, False, True]),
        triggered=torch.tensor([False, False, True, False, False, True]),
    )
    expected = {name: getattr(env, name).clone() for name in ("obstacle_y", "target_y", "obstacle_active", "triggered")}
    ends = torch.round(elapsed * 1000).long()
    starts = torch.round((elapsed - step_dt) * 1000).long() + 1
    for row in range(6):
        near = abs(float(position[row, 0]) - 6.9) <= 1.0
        for tick in range(max(0, int(starts[row])), int(ends[row]) + 1):
            periodic = tick % 101 == 0
            if periodic and near and not expected["obstacle_active"][row]:
                expected["target_y"][row] = position[row, 1]
                expected["obstacle_y"][row] = position[row, 1] + 1
                expected["obstacle_active"][row] = True
            if periodic and expected["obstacle_active"][row]:
                expected["obstacle_y"][row] = max(expected["obstacle_y"][row] - 1, expected["target_y"][row])
            if near and not expected["triggered"][row]:
                expected["obstacle_y"][row] = max(expected["obstacle_y"][row] - 1, expected["target_y"][row])
                expected["triggered"][row] = True

    advance_scene(env, None)
    for name, value in expected.items():
        torch.testing.assert_close(getattr(env, name), value)
    torch.testing.assert_close(env.elapsed, elapsed + step_dt)
    torch.testing.assert_close(writes[0]["root_pose"][:, 1], expected["obstacle_y"] + origins[:, 1])
