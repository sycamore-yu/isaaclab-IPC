"""Build the pinned ROS-free PathCover front end, including its CUDA map kernels."""

import argparse
import json
import os
import subprocess
import sys
import sysconfig
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu", action="store_true", help="Build an explicit CPU-only numerical reference")
    parser.add_argument("--cuda-home", type=Path, default=os.environ.get("CUDA_HOME"))
    parser.add_argument("--cuda-arch", default="89")
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--build-dir", type=Path, help="Independent object directory for a candidate build")
    parser.add_argument("--output-dir", type=Path, help="Write the candidate extension here without publishing it")
    args = parser.parse_args()
    import pybind11

    root = Path(__file__).resolve().parents[1]
    vendor = root / "third_party/pathcover_native"
    build = args.build_dir or root / "build/pathcover_native" / ("cpu" if args.cpu else "cuda")
    build.mkdir(parents=True, exist_ok=True)
    cuda = args.cuda_home or root / "build/turbompc/toolchains/cuda-13.0.2"
    if not args.cpu and not (cuda / "bin/nvcc").is_file():
        raise SystemExit("Pass --cuda-home matching PyTorch's toolkit, or --cpu for the CPU reference")
    includes = [
        vendor, vendor / "jps/include", vendor / "trajectory",
        vendor / "nanovoxmap/include", vendor / "occupancy_inflation/include",
        Path("/usr/include/eigen3"), Path(pybind11.get_include()),
        Path(sysconfig.get_path("include")),
    ]
    if not args.cpu:
        includes.append(cuda / "include/cccl")
    flags = [
        "-O3", "-DNDEBUG", "-std=c++17", "-fPIC", "-fopenmp", "-pthread",
        "-DPATHCOVER_DIM=3", *(f"-I{path}" for path in includes),
    ]
    definitions = [] if args.cpu else ["-DNANOVOXMAP_WITH_CUDA", "-DOCCUPANCY_INFLATION_WITH_CUDA"]
    sources = [
        root / "native/pathcover_native.cpp",
        vendor / "jps/src/jps_planner/graph_search.cpp",
        vendor / "jps/src/jps_planner/jps_planner.cpp",
        vendor / "jps/src/distance_map_planner/graph_search.cpp",
        vendor / "jps/src/distance_map_planner/distance_map_planner.cpp",
        vendor / "occupancy_inflation/src/voxel_bit_grid.cpp",
        vendor / "occupancy_inflation/src/point_cloud_inflater.cpp",
    ]
    if not args.cpu:
        sources += [
            root / "native/pathcover_distance_map.cu",
            root / "native/pathcover_boxes_cuda.cu",
            root / "native/pathcover_grid_cuda.cu",
            vendor / "nanovoxmap/src/nanovoxmap_cuda.cu",
            vendor / "occupancy_inflation/src/inflation_cuda.cu",
        ]
    header_times = {
        folder: max(path.stat().st_mtime_ns for path in (vendor / folder).rglob("*.h*"))
        for folder in ("corridor", "trajectory", "jps", "nanovoxmap", "occupancy_inflation")
    }
    header_times["native"] = max(path.stat().st_mtime_ns for path in (root / "native").glob("pathcover*.h"))
    manifest = {
        "python": sys.version, "cuda_home": str(cuda), "cuda_arch": args.cuda_arch,
        "flags": flags, "definitions": definitions,
    }
    manifest_path = build / "configuration.json"
    changed = not manifest_path.exists() or json.loads(manifest_path.read_text()) != manifest

    def compile_source(source):
        obj = build / (source.relative_to(root).as_posix().replace("/", "_") + ".o")
        latest_header = (
            header_times[source.relative_to(vendor).parts[0]]
            if source.is_relative_to(vendor) else max(header_times.values())
        )
        if not changed and obj.exists() and obj.stat().st_mtime_ns >= max(source.stat().st_mtime_ns, latest_header):
            return obj
        if source.suffix == ".cu":
            command = [
                str(cuda / "bin/nvcc"), "-O3", "-std=c++17", f"-arch=sm_{args.cuda_arch}",
                "--expt-relaxed-constexpr", "-Xcompiler=-fPIC,-fopenmp,-pthread",
                *(f"-I{path}" for path in includes), *definitions,
            ]
        else:
            command = [os.environ.get("CXX", "g++"), *flags, *definitions]
        subprocess.run([*command, "-c", str(source), "-o", str(obj)], check=True)
        return obj

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        objects = list(pool.map(compile_source, sources))
    output = args.output_dir or root / "src/isaaclab_ipc"
    output.mkdir(parents=True, exist_ok=True)
    target = output / ("_pathcover_native" + sysconfig.get_config_var("EXT_SUFFIX"))
    temporary = target.with_name(target.name + ".tmp")
    libraries = ["-lqhullcpp", "-lqhull_r"]
    if not args.cpu:
        cuda_library = cuda / "lib" if (cuda / "lib/libcudart.so").exists() else cuda / "lib64"
        libraries += [f"-L{cuda_library}", f"-Wl,-rpath,{cuda_library}", "-lcudart"]
    subprocess.run(
        [os.environ.get("CXX", "g++"), "-shared", "-fopenmp", "-pthread",
         *(str(obj) for obj in objects), *libraries, "-o", str(temporary)], check=True,
    )
    temporary.replace(target)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(target)


if __name__ == "__main__":
    main()
