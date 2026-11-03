"""Build the pinned IPC A* and the vendored PathCover corridor kernel."""

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
    s = s.replace('#include "astar.h"', '#include "astar.h"\n#include <memory>\n#include <unordered_map>')
    allocation = s.index("        GridNodeMap_ = new")
    allocation_end = s.index("    void Reset(void)", allocation)
    s = s[:allocation] + "        nodes_.clear();\n    }\n" + s[allocation_end:]
    s = remove_block(s, "void Reset(void)")
    s = s.replace("    void setObs(", """    GridNodePtr NodeAt(int x, int y, int z) {
        const size_t key = (size_t(x)*GL_SIZE_(1)+y)*GL_SIZE_(2)+z;
        auto &node = nodes_[key];
        if (!node) node = std::make_unique<GridNode>(Eigen::Vector3i(x,y,z));
        return node.get();
    }
    void Reset() {
        for (auto &item: nodes_) {
            auto &node = item.second;
            node->id=0; node->cameFrom=nullptr; node->gScore=inf; node->fScore=inf;
        }
    }
    void setObs(""", 1)
    s = s.replace("    void setObs(", """    void setObsMatrixFast(const Eigen::Ref<const Eigen::Matrix<double,Eigen::Dynamic,3,Eigen::RowMajor>>& cloud,double radius);
    void loadPackedOccupancy(const uint32_t* words, size_t count) {
        const size_t cells = size_t(GL_SIZE_(0))*GL_SIZE_(1)*GL_SIZE_(2);
        if (count != (cells + 31)/32) throw std::invalid_argument("Packed occupancy shape mismatch");
        packed_.assign(words, words + count);
    }
    inline bool occupiedAt(size_t index) const {
        return packed_.empty() ? grid_map_[index]!=0 : (packed_[index/32] >> (index%32)) & 1u;
    }
    bool sameOccupancy(const LoaclAstarClass &other) const {
        if (resolution_!=other.resolution_ || center_!=other.center_ || gl_low_!=other.gl_low_ ||
            gl_upp_!=other.gl_upp_ || GL_SIZE_!=other.GL_SIZE_) return false;
        if (!packed_.empty() && !other.packed_.empty()) return packed_==other.packed_;
        const size_t cells=size_t(GL_SIZE_(0))*GL_SIZE_(1)*GL_SIZE_(2);
        for(size_t i=0;i<cells;++i) if (occupiedAt(i)!=other.occupiedAt(i)) return false;
        return true;
    }
    void setObs(""", 1)
    s = s.replace("GridNodePtr ***GridNodeMap_;", "std::unordered_map<size_t, std::unique_ptr<GridNode>> nodes_;\n    std::vector<uint32_t> packed_;")
    s = s.replace("(grid_map_[idx_x*GL_SIZE_(1)*GL_SIZE_(2) + idx_y*GL_SIZE_(2) + idx_z] == 1)",
                  "occupiedAt((size_t(idx_x)*GL_SIZE_(1)+idx_y)*GL_SIZE_(2)+idx_z)")
    s = s.replace("(grid_map_[idx_x*GL_SIZE_(1)*GL_SIZE_(2) + idx_y*GL_SIZE_(2) + idx_z] < 1)",
                  "!occupiedAt((size_t(idx_x)*GL_SIZE_(1)+idx_y)*GL_SIZE_(2)+idx_z)")
    s = s.replace("void setObsVector(std::vector<Eigen::Vector3d> &cloud, double radius = 0.2) {",
                  "void setObsVector(std::vector<Eigen::Vector3d> &cloud, double radius = 0.2) {\n        packed_.clear();")
    # Configurable SUPER extents need the final partial voxel, unlike IPC's divisible defaults.
    for axis in "xyz":
        s = s.replace(f"(gl_upp.{axis}() - gl_low.{axis}()) / resolution_;",
                      f"std::ceil((gl_upp.{axis}() - gl_low.{axis}()) / resolution_);")
    s = s.replace("    double resolution_;", """    int heuristic_=3;
    bool virtual_limits_=false;
    double virtual_floor_=0, virtual_ceiling_=0;
    double resolution_;""")
    s = s.replace("    inline bool isOccupied(int idx_x, int idx_y, int idx_z) {", """    inline bool isOccupied(int idx_x, int idx_y, int idx_z) {
        if (virtual_limits_) {
            double height=(idx_z+.5)*resolution_+gl_low_.z()+center_.z();
            if(height<virtual_floor_ || height>virtual_ceiling_) return true;
        }""")
    s = s.replace(
        "~LoaclAstarClass(){}",
        "~LoaclAstarClass(){ delete[] grid_map_; }",
    )
    save("local_astar.h", old, s)
    old = read("local_astar.cpp")
    s = old
    s = remove_block(s, "if ((time_2 - time_1).toSec()")
    s = re.sub(r"^.*(?:ros::Time|ROS_WARN\(|ROS_ERROR\().*\n", "", s, flags=re.M)
    s = s.replace("GridNodeMap_[nx][ny][nz]", "NodeAt(nx,ny,nz)")
    s = s.replace("int distance_norm = 3;", "int distance_norm = heuristic_;")
    for name in ("start", "end"):
        s = s.replace(f"GridNodeMap_[{name}_idx(0)][{name}_idx(1)][{name}_idx(2)]",
                      f"NodeAt({name}_idx(0),{name}_idx(1),{name}_idx(2))")
    save("local_astar.cpp", old, s)
    (out.parent / "port.patch").write_text("".join(patches))
    (out.parent / "source.json").write_text(
        json.dumps(
            {
                "upstream": str(args.upstream.resolve()),
                "revision": PIN,
                "corridor": "PathCover/RISP@8723a05d1993bf0199a453b47dc6ee167d493c23",
                "changes": "ROS I/O removed; C++17 const comparator; lazy A* nodes with unchanged occupancy/search",
            },
            indent=2,
        )
    )
    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    target = root / "src/isaaclab_ipc" / ("_geometry" + suffix)
    compiled = target.with_name(target.name + ".tmp")
    subprocess.run(
        [
            "g++",
            "-O3",
            "-DNDEBUG",
            "-std=c++17",
            "-shared",
            "-fPIC",
            f"-I{out}",
            f"-I{root / 'third_party/pathcover'}",
            "-I/usr/include/eigen3",
            f"-I{pybind11.get_include()}",
            f"-I{sysconfig.get_path('include')}",
            str(root / "native/geometry.cpp"),
            "-lqhullcpp",
            "-lqhull_r",
            "-o",
            str(compiled),
        ],
        check=True,
    )
    compiled.replace(target)
    print(target)


if __name__ == "__main__":
    main()
