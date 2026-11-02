"""Build the pinned IPC geometry locally; remove ROS I/O, not geometry math."""

import argparse
import difflib
import json
import re
import subprocess
import sysconfig
from pathlib import Path

PIN = "0f959ca7995e71a1fcc4942bd4a86869ab6ee975"


def remove_block(text, marker, semicolon=False):
    start = text.index(marker)
    start = text.rfind("\n", 0, start) + 1
    cursor = text.index("{", start) + 1
    depth = 1
    while depth:
        depth += (text[cursor] == "{") - (text[cursor] == "}")
        cursor += 1
    return text[:start] + text[cursor + int(semicolon) :]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    args = parser.parse_args()
    import pybind11

    root = Path(__file__).resolve().parents[1]
    out = root / "build/native/include"
    out.mkdir(parents=True, exist_ok=True)
    patches = []

    def read(name):
        return subprocess.check_output(
            ["git", "-C", str(args.upstream), "show", f"{PIN}:IPC/include/{name}"], text=True
        )

    def save(name, old, new):
        (out / name).write_text(new)
        patches.extend(difflib.unified_diff(old.splitlines(True), new.splitlines(True), fromfile=name, tofile=name))

    for name in [
        "common_type_name.hpp",
        "ellipsoid.hpp",
        "emvp.hpp",
        "geo_utils.hpp",
        "lbfgs.hpp",
        "line_segment.hpp",
        "polytope.hpp",
        "quickhull.hpp",
        "sdlp.hpp",
    ]:
        old = read("polytope/" + name)
        s = old
        if name in ("ellipsoid.hpp", "polytope.hpp", "line_segment.hpp"):
            s = remove_block(s, "void Visualize(")
        s = re.sub(r"^#include [<\"](?:ros/|visualization_msgs/|std_msgs/)[^\n]*\n", "", s, flags=re.M)
        if name == "common_type_name.hpp":
            s = s.replace('#include "fmt/color.h"\n', "").replace("using namespace fmt;\n", "")
            s = remove_block(s, "class Color :", True)
        if name == "geo_utils.hpp":
            s = re.sub(
                r"(bool operator\(\)\(const Eigen::Vector3d &l,\s*const Eigen::Vector3d &r\))\s*\{", r"\1 const {", s
            )
        s = re.sub(r'print\(fg\(color::gold\), ("[^\n]*?")\);', r'std::fprintf(stderr, "%s", \1);', s)
        save(name, old, "#include <cstdio>\n" + s)
    old = read("astar.h")
    s = (
        old[: old.index("class AstarClass")].replace("#include <ros/ros.h>", "#include <cstdint>\n#include <cstring>")
        + "\n#endif\n"
    )
    save("astar.h", old, s)
    old = read("local_astar.h")
    s = old
    for marker in ["void setObsPcl(", "void GetOccupyPcl("]:
        s = remove_block(s, marker)
    s = re.sub(r"^#include <(?:ros/|pcl)[^\n]*\n", "", s, flags=re.M)
    s = s.replace(
        "~LoaclAstarClass(){}",
        """~LoaclAstarClass(){
        for (int i=0;i<GL_SIZE_.x();++i){
            for(int j=0;j<GL_SIZE_.y();++j){
                for(int k=0;k<GL_SIZE_.z();++k) delete GridNodeMap_[i][j][k];
                delete[] GridNodeMap_[i][j];
            }
            delete[] GridNodeMap_[i];
        }
        delete[] GridNodeMap_; delete[] grid_map_;
    }""",
    )
    save("local_astar.h", old, s)
    old = read("local_astar.cpp")
    s = old
    s = remove_block(s, "if ((time_2 - time_1).toSec()")
    s = re.sub(r"^.*(?:ros::Time|ROS_WARN\(|ROS_ERROR\().*\n", "", s, flags=re.M)
    save("local_astar.cpp", old, s)
    (out.parent / "port.patch").write_text("".join(patches))
    (out.parent / "source.json").write_text(
        json.dumps(
            {
                "upstream": str(args.upstream.resolve()),
                "revision": PIN,
                "changes": "ROS I/O removed; C++17 const comparator; release A* allocations",
            },
            indent=2,
        )
    )
    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    target = root / "src/isaaclab_ipc" / ("_geometry" + suffix)
    subprocess.run(
        [
            "g++",
            "-O3",
            "-DNDEBUG",
            "-std=c++17",
            "-shared",
            "-fPIC",
            f"-I{out}",
            "-I/usr/include/eigen3",
            f"-I{pybind11.get_include()}",
            f"-I{sysconfig.get_path('include')}",
            str(root / "native/geometry.cpp"),
            "-o",
            str(target),
        ],
        check=True,
    )
    print(target)


if __name__ == "__main__":
    main()
