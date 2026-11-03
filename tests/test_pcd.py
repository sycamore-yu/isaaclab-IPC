import numpy as np
import pytest

from isaaclab_ipc.scenes import read_pcd


@pytest.mark.parametrize('encoding', ['ascii', 'binary'])
def test_pcd_formats_preserve_xyz_and_field_order(tmp_path, encoding):
    path = tmp_path / 'map.pcd'
    header = ('FIELDS intensity z x y\nSIZE 4 4 4 4\nTYPE F F F F\nCOUNT 1 1 1 1\n'
              f'WIDTH 2\nHEIGHT 1\nPOINTS 2\nDATA {encoding}\n')
    values = np.array([[4, 3, 1, 2], [8, 7, 5, 6]], dtype=np.float32)
    with path.open('wb') as f:
        f.write(header.encode())
        if encoding == 'ascii':
            np.savetxt(f, values)
        else:
            f.write(values.tobytes())
    np.testing.assert_array_equal(read_pcd(path), [[1, 2, 3], [5, 6, 7]])


def test_ascii_pcd_rejects_wrong_point_count(tmp_path):
    path = tmp_path / 'bad.pcd'
    path.write_text('FIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nPOINTS 2\nDATA ascii\n1 2 3\n')
    with pytest.raises(ValueError, match='point count'):
        read_pcd(path)


def test_nan_removal_is_explicit_and_does_not_allow_infinity(tmp_path):
    path = tmp_path / 'map.pcd'
    header = 'FIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nPOINTS 2\nDATA ascii\n'
    path.write_text(header + 'nan nan nan\n1 2 3\n')
    with pytest.raises(ValueError, match='Nonfinite'):
        read_pcd(path)
    np.testing.assert_array_equal(read_pcd(path, drop_nan=True), [[1, 2, 3]])
    path.write_text(header + 'inf 0 0\n1 2 3\n')
    with pytest.raises(ValueError, match='Nonfinite'):
        read_pcd(path, drop_nan=True)
