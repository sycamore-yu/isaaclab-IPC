import numpy as np
import pytest

pytest.importorskip("rerun")

from isaaclab_ipc.recording import corridor_edges  # noqa: E402


def test_sfc_edges_preserve_halfspaces_and_duplicate_planes():
    planes = np.c_[np.r_[np.eye(3), -np.eye(3)], -np.ones(6)]
    edges = corridor_edges(np.r_[planes, planes[:1] * 3])
    assert edges.shape == (12, 2, 3)
    assert np.unique(edges.reshape(-1, 3), axis=0).shape == (8, 3)
    assert (edges @ planes[:, :3].T + planes[:, 3] <= 1e-8).all()
    assert corridor_edges([]).shape == (0, 2, 3)
    with pytest.raises(ValueError, match="nonfinite"):
        corridor_edges([[np.nan] * 4])
