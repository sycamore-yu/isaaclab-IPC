from types import SimpleNamespace

import pytest

from isaaclab_ipc.tasks.recorders import recorder_config


def test_disabled_recording_has_no_dataset():
    assert recorder_config(SimpleNamespace(recorder_dir=None, rerun_path=None)) is None


@pytest.mark.parametrize("kind", ["hdf5", "rrd"])
def test_recording_rejects_existing_data(tmp_path, kind):
    path = tmp_path / ("flight.hdf5" if kind == "hdf5" else "flight.rrd")
    path.write_bytes(b"historical recording")
    cfg = SimpleNamespace(
        recorder_dir=str(tmp_path) if kind == "hdf5" else None,
        rerun_path=str(path) if kind == "rrd" else None,
    )
    with pytest.raises(FileExistsError):
        recorder_config(cfg)
    assert path.read_bytes() == b"historical recording"
