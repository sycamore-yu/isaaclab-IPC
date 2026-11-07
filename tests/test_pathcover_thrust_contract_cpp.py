"""Compile the native option/bounds contract without loading a stale planner DSO."""

import importlib.util
import re
import shutil
import subprocess
import sysconfig
from pathlib import Path

import numpy as np
import pytest

from isaaclab_ipc.pathcover_native import PathCoverConfig


@pytest.fixture(scope="module")
def native_contract(tmp_path_factory):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("native compiler is unavailable")
    pybind = pytest.importorskip("pybind11")
    root = Path(__file__).resolve().parents[1]
    text = (root / "native/pathcover_native.cpp").read_text()
    options = text[text.index("struct Options {"):text.index("struct MappingStats {")]
    cost_bounds = re.search(r"bounds << options_\.nominal_velocity,[\s\S]*?;", text)[0]
    feasibility = re.search(r"result\.optimizer_quadrature_feasible = [\s\S]*?;", text)[0]
    tolerance = re.search(r"constexpr double kSfcFeasibilityToleranceM = [^;]+;", text)[0]
    # These are the actual production parser and call-site expressions. The
    # fixture does not duplicate a hand-written cost/feasibility implementation.
    source = r"""
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <Eigen/Core>
#include <cmath>
#include <stdexcept>
namespace py = pybind11;
using Vec3 = Eigen::Vector3d;
""" + tolerance + options + r"""
PYBIND11_MODULE(_pathcover_thrust_contract, module) {
    module.def("evaluate", [](const py::dict &values,
                              const Eigen::Matrix<double, 6, 1> &m,
                              double speed_acceptance_limit) {
        Options options_(values);
        Eigen::VectorXd bounds(5);
""" + cost_bounds + r"""
        struct Result {
            double speed_acceptance_limit;
            bool optimizer_quadrature_feasible = false;
        } result{speed_acceptance_limit};
""" + feasibility + r"""
        return py::make_tuple(bounds, result.optimizer_quadrature_feasible);
    });
}
"""
    folder = tmp_path_factory.mktemp("pathcover-thrust-contract")
    cpp = folder / "contract.cpp"
    binary = folder / ("_pathcover_thrust_contract" + sysconfig.get_config_var("EXT_SUFFIX"))
    cpp.write_text(source)
    subprocess.run(
        [compiler, "-O2", "-std=c++17", "-shared", "-fPIC", "-I/usr/include/eigen3",
         f"-I{pybind.get_include()}", f"-I{sysconfig.get_path('include')}", str(cpp), "-o", str(binary)],
        check=True, capture_output=True, text=True,
    )
    spec = importlib.util.spec_from_file_location("_pathcover_thrust_contract", binary)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_native_cost_target_does_not_replace_physical_floor(native_contract):
    cfg = PathCoverConfig(thrust_min=0.0, nominal_thrust_min=2.5, thrust_max=22.0)
    metrics = np.array([-0.1, 1.0, 1.0, 8.0, 0.2, 1.0])
    bounds, feasible = native_contract.evaluate(cfg.native_options(), metrics, 6.006)
    np.testing.assert_array_equal(bounds, [3.0, 2.99, 0.7854, 2.5, 22.0])
    assert feasible  # A 1.5 N nominal shortfall is not a physical failure.
    metrics[2] = -0.01
    _, feasible = native_contract.evaluate(cfg.native_options(), metrics, 6.006)
    assert not feasible

    historical = cfg.native_options()
    historical.pop("nominal_thrust_min")
    bounds, _ = native_contract.evaluate(historical, metrics, 6.006)
    assert bounds[3] == historical["thrust_min"] == 0.0


def test_native_parser_rejects_invalid_nominal_force_order(native_contract):
    original = PathCoverConfig(thrust_min=0.5, nominal_thrust_min=2.5, thrust_max=22.0).native_options()
    for value in (0.0, 22.0, np.nan, np.inf):
        values = dict(original, nominal_thrust_min=value)
        with pytest.raises(ValueError, match="nominal_thrust_min"):
            native_contract.evaluate(values, np.zeros(6), 6.006)
