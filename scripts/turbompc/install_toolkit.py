"""Install pinned NVIDIA CUDA build components in this worktree, without a driver."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import urllib.request
from pathlib import Path

BASE_URL = "https://developer.download.nvidia.com/compute/cuda/redist/"
VERSIONS = {"cu128": "12.8.1", "cu130": "13.0.2"}
COMPONENTS = (
    "cuda_nvcc", "cuda_cudart", "cuda_cccl", "cuda_crt", "libnvvm",
    "libcusparse", "libnvjitlink", "cuda_nvrtc",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", choices=VERSIONS)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    version = VERSIONS[args.profile]
    base = root / "build" / "turbompc"
    cache = base / "downloads"
    prefix = base / "toolchains" / f"cuda-{version}"
    cache.mkdir(parents=True, exist_ok=True)
    prefix.mkdir(parents=True, exist_ok=True)
    manifest_path = cache / f"redistrib_{version}.json"
    if not manifest_path.exists():
        with urllib.request.urlopen(BASE_URL + manifest_path.name, timeout=60) as response:
            manifest_path.write_bytes(response.read())
    manifest = json.loads(manifest_path.read_text())
    installed = {}
    for component in COMPONENTS:
        if component not in manifest:
            continue  # CUDA 12 bundles CRT and NVVM into the nvcc component.
        spec = manifest[component]["linux-x86_64"]
        archive = cache / Path(spec["relative_path"]).name
        marker = prefix / "manifests" / f"{component}.json"
        if marker.exists() and json.loads(marker.read_text()) == spec:
            installed[component] = manifest[component]
            continue
        if not archive.exists():
            print(f"Downloading {component} {manifest[component]['version']}", flush=True)
            partial = archive.with_suffix(archive.suffix + ".part")
            subprocess.run([
                "curl", "--fail", "--location", "--retry", "5", "--retry-all-errors",
                "--continue-at", "-", "--connect-timeout", "30",
                "--silent", "--show-error", "--output", str(partial), BASE_URL + spec["relative_path"],
            ], check=True)
            partial.rename(archive)
        with archive.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != spec["sha256"]:
            raise RuntimeError(f"SHA256 mismatch: {archive}")
        print(f"Installing {component} (SHA256 verified)", flush=True)
        subprocess.run(["tar", "-xJf", str(archive), "--strip-components=1", "-C", str(prefix)], check=True)
        license_path = prefix / "LICENSE"
        if license_path.is_file():
            licenses = prefix / "licenses"
            licenses.mkdir(exist_ok=True)
            shutil.copyfile(license_path, licenses / f"{component}.txt")
        marker.parent.mkdir(exist_ok=True)
        marker.write_text(json.dumps(spec, indent=2) + "\n")
        installed[component] = manifest[component]
    # NVIDIA redistributable archives use lib/, while Torch expects lib64/.
    if not (prefix / "lib64").exists():
        (prefix / "lib64").symlink_to("lib", target_is_directory=True)
    (prefix / "installed-components.json").write_text(json.dumps({
        "cuda": {"name": "CUDA Toolkit", "version": version},
        "manifest_url": BASE_URL + manifest_path.name,
        "components": installed,
    }, indent=2) + "\n")
    subprocess.run([str(prefix / "bin" / "nvcc"), "--version"], check=True)
    print(f"Toolkit installed: {prefix}", flush=True)


if __name__ == "__main__":
    main()
