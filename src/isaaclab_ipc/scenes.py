"""Published IPC scene data and deterministic 1 kHz obstacle event schedule."""

from pathlib import Path

import numpy as np


def cpp_range(start, stop, step, inclusive=False):
    values = []
    while start <= stop if inclusive else start < stop:
        values.append(start)
        start += step
    return np.array(values)


def point_box(axes):
    return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3).astype(np.float32)


class AvoidScene:
    def __init__(self, distance=1.0):
        self.distance = distance
        self.time = 0
        self.first = False
        self.active = False
        self.y = 1.5
        self.target_y = 0.0
        self.obstacle = np.empty((0, 3))
        self.trigger_speed = None
        self.wall = point_box([cpp_range(0, 10, 0.1), cpp_range(0.8, 1.0, 0.1), cpp_range(0, 3.0, 0.1)])

    def _publish(self, move):
        if move:
            self.y = max(self.y - 1.0, self.target_y)
            self.obstacle = point_box(
                [
                    cpp_range(6.9, 7.1, 0.1, True),
                    cpp_range(self.y - 0.1, self.y + 0.1, 0.1, True),
                    cpp_range(0.0, 4.0, 0.1, True),
                ]
            )
        return np.vstack((self.wall, self.obstacle))

    def advance(self, time, position, speed):
        clouds = []
        while self.time <= int(round(time * 1000)):
            near = abs(position[0] - 6.9) <= self.distance
            if self.time % 101 == 0:
                if near and not self.active:
                    self.active = True
                    self.target_y = position[1]
                    self.y = position[1] + 1.0
                clouds.append(self._publish(self.active))
            if near and not self.first:
                clouds.append(self._publish(True))
                self.first = True
                self.trigger_speed = float(speed)
            self.time += 1
        return clouds

    def clearance(self, position):
        if not len(self.obstacle):
            return float("inf")
        return float(np.linalg.norm(self.obstacle[:, :2] - position[:2], axis=1).min() - 0.25)


def read_pcd(path):
    with Path(path).open("rb") as f:
        header = {}
        while True:
            line = f.readline()
            if not line:
                raise ValueError("Truncated PCD header")
            line = line.decode("ascii").strip()
            if not line or line.startswith("#"):
                continue
            key, _, value = line.partition(" ")
            header[key] = value
            if key == "DATA":
                break
        offset = f.tell()
    if header["DATA"] != "binary" or set(header["TYPE"].split()) != {"F"} or set(header["SIZE"].split()) != {"4"}:
        raise ValueError("Expected binary float32 PCD")
    fields = header["FIELDS"].split()
    count = int(header["POINTS"])
    data = np.memmap(path, dtype=np.float32, mode="r", offset=offset, shape=(count, len(fields)))
    xyz = np.array(data[:, [fields.index(k) for k in "xyz"]])
    if not np.isfinite(xyz).all():
        raise ValueError("Nonfinite map")
    return xyz
