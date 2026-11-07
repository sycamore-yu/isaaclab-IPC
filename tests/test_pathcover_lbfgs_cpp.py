"""Contracts for optional objective-relative L-BFGS gradient stopping."""

import shutil
import subprocess
from pathlib import Path

import pytest

SOURCE = r"""
#ifndef PATHCOVER_LBFGS_HEADER
#define PATHCOVER_LBFGS_HEADER "gcopter/lbfgs.hpp"
#endif
#include PATHCOVER_LBFGS_HEADER
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

struct Context {
    int evaluations = 0, accepted = 0, kind = 0;
    double offset = 0, curvature = 1, target = 0, scale = 1;
    std::vector<double> trace;
};

double objective(void* raw, const Eigen::VectorXd& x, Eigen::VectorXd& gradient) {
    auto& context = *static_cast<Context*>(raw);
    ++context.evaluations;
    double value;
    if (context.kind == 1) {
        gradient[0] = 3 * x[0] + x[1];
        gradient[1] = x[0] + 2 * x[1];
        value = .25 + .5 * (3*x[0]*x[0] + 2*x[0]*x[1] + 2*x[1]*x[1]);
    } else if (context.kind == 2 || context.kind == 3) {
        gradient[0] = 1;
        value = context.kind == 2 ? std::numeric_limits<double>::infinity()
                                  : std::numeric_limits<double>::quiet_NaN();
    } else if (context.kind == 4) {
        gradient[0] = 1;
        value = x[0];  // No point can satisfy the weak Wolfe curvature condition.
    } else {
        const double distance = x[0] - context.target;
        gradient[0] = context.scale * context.curvature * distance;
        value = context.scale * (context.offset + .5*context.curvature*distance*distance);
    }
    context.trace.push_back(value);
    for (int i = 0; i < x.size(); ++i) {
        context.trace.push_back(x[i]);
        context.trace.push_back(gradient[i]);
    }
    return value;
}

int progress(void* raw, const Eigen::VectorXd& x, const Eigen::VectorXd& gradient,
             double value, double step, int iteration, int evaluations) {
    auto& context = *static_cast<Context*>(raw);
    ++context.accepted;
    context.trace.insert(context.trace.end(), {value, step, double(iteration), double(evaluations)});
    for (int i = 0; i < x.size(); ++i) {
        context.trace.push_back(x[i]);
        context.trace.push_back(gradient[i]);
    }
    return 0;
}

std::string legacy_trace(bool explicit_zero) {
    Context context;
    context.kind = 1;
    Eigen::VectorXd x(2);
    x << 2, -1;
    lbfgs::lbfgs_parameter_t parameters;
    parameters.max_iterations = 100;
#ifndef PATHCOVER_LEGACY_TRACE
    require(parameters.g_relative_epsilon == 0, "public default must disable relative stopping");
    if (explicit_zero) parameters.g_relative_epsilon = 0;
#endif
    double value;
    const int status = lbfgs::lbfgs_optimize(x, value, objective, nullptr, progress, &context, parameters);
    require(status >= 0 && x.norm() < 1e-5, "legacy quadratic did not converge");
    std::ostringstream out;
    out << std::hexfloat << status << ' ' << context.evaluations << ' ' << context.accepted
        << ' ' << value << ' ' << x[0] << ' ' << x[1];
    for (const double entry : context.trace) out << ' ' << entry;
    return out.str();
}

#ifndef PATHCOVER_LEGACY_TRACE
void initial_criteria() {
    // This finite point fails the absolute test but satisfies the relative one.
    // Positive objective scaling leaves the relative ratio unchanged here.
    for (const double scale : {1.0, 1000.0}) {
        Context context;
        context.offset = 1000;
        context.scale = scale;
        Eigen::VectorXd x(1); x << 2e-5;
        lbfgs::lbfgs_parameter_t parameters;
        parameters.g_relative_epsilon = 1e-7;
        parameters.past = 0;
        double value;
        const int status = lbfgs::lbfgs_optimize(x, value, objective, nullptr, progress, &context, parameters);
        require(status == lbfgs::LBFGS_CONVERGENCE && context.evaluations == 1 && context.accepted == 0,
                "relative initial criterion was not applied before line search");
        require(x[0] == 2e-5 && value == scale*(1000 + .5*x[0]*x[0]),
                "relative criterion changed the raw objective or initial point");
    }
    // The absolute test remains an independent OR branch even when the relative
    // test is stricter. This case must not enter line search.
    Context context;
    Eigen::VectorXd x(1); x << 5e-6;
    lbfgs::lbfgs_parameter_t parameters;
    parameters.g_relative_epsilon = 1e-7;
    double value;
    const int status = lbfgs::lbfgs_optimize(x, value, objective, nullptr, progress, &context, parameters);
    require(status == lbfgs::LBFGS_CONVERGENCE && context.evaluations == 1 && context.accepted == 0,
            "absolute gradient convergence was replaced instead of retained");
}

void accepted_criterion() {
    Context context;
    context.offset = 5000;
    context.curvature = 1e-3;
    context.target = .2;
    Eigen::VectorXd x(1); x << 1;
    lbfgs::lbfgs_parameter_t parameters;
    parameters.g_relative_epsilon = 1e-7;
    parameters.past = 0;
    double value;
    const int status = lbfgs::lbfgs_optimize(x, value, objective, nullptr, progress, &context, parameters);
    require(status == lbfgs::LBFGS_CONVERGENCE && context.evaluations == 2 && context.accepted == 1,
            "relative criterion must run after an accepted Wolfe step");
    const double gradient = context.curvature * (x[0] - context.target);
    require(std::abs(gradient) > parameters.g_epsilon && std::abs(x[0]) < 1e-14,
            "fixture did not isolate the post-acceptance relative branch");
    require(value == context.offset + .5*context.curvature*(x[0]-.2)*(x[0]-.2),
            "returned value is not the raw objective");
}

void invalid_thresholds() {
    for (const double epsilon : {-1.0, std::numeric_limits<double>::infinity(),
                                  std::numeric_limits<double>::quiet_NaN()}) {
        Context context;
        Eigen::VectorXd x(1); x << 1;
        lbfgs::lbfgs_parameter_t parameters;
        parameters.g_relative_epsilon = epsilon;
        double value = -1;
        const int status = lbfgs::lbfgs_optimize(x, value, objective, nullptr, progress, &context, parameters);
        require(status == lbfgs::LBFGSERR_INVALID_GEPSILON && context.evaluations == 0,
                "invalid relative threshold must fail before objective evaluation");
    }
}

void unsuccessful_searches() {
    for (const int kind : {2, 3, 4}) {
        Context context;
        context.kind = kind;
        Eigen::VectorXd x(1); x << 0;
        lbfgs::lbfgs_parameter_t parameters;
        parameters.g_relative_epsilon = 1e-7;
        parameters.max_linesearch = 2;
        double value;
        const int status = lbfgs::lbfgs_optimize(x, value, objective, nullptr, progress, &context, parameters);
        require(status == lbfgs::LBFGSERR_MAXIMUMLINESEARCH && context.evaluations == 3
                    && context.accepted == 0 && x[0] == 0,
                "nonfinite objective or failed Wolfe search was reclassified as success");
    }
}
#endif

int main(int argc, char** argv) {
    static_assert(lbfgs::LBFGSERR_INVALID_GEPSILON == -1021);
    static_assert(lbfgs::LBFGSERR_MAXIMUMLINESEARCH == -1009);
    static_assert(lbfgs::LBFGSERR_MAXIMUMITERATION == -1008);
#ifdef PATHCOVER_LEGACY_TRACE
    std::cout << legacy_trace(false) << '\n';
#else
    const std::string mode = argc == 2 ? argv[1] : "";
    if (mode == "legacy") {
        const auto implicit = legacy_trace(false);
        require(implicit == legacy_trace(true), "default and explicit zero changed callback traces");
        std::cout << implicit << '\n';
    } else if (mode == "initial") initial_criteria();
    else if (mode == "accepted") accepted_criterion();
    else if (mode == "invalid") invalid_thresholds();
    else if (mode == "failure") unsuccessful_searches();
    else throw std::runtime_error("unknown contract case");
#endif
}
"""


@pytest.fixture(scope="module")
def lbfgs_contract_binary(tmp_path_factory):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("native compiler is unavailable")
    root = Path(__file__).resolve().parents[1]
    directory = tmp_path_factory.mktemp("pathcover_lbfgs")
    source = directory / "contract.cpp"
    binary = directory / "contract"
    source.write_text(SOURCE)
    subprocess.run([
        compiler, "-O2", "-std=c++17", "-I/usr/include/eigen3",
        f"-I{root / 'third_party/pathcover_native/trajectory'}", str(source), "-o", str(binary),
    ], check=True, capture_output=True, text=True)
    return binary


@pytest.mark.parametrize("case", ["legacy", "initial", "accepted", "invalid", "failure"])
def test_lbfgs_gradient_stopping_contract(lbfgs_contract_binary, case):
    subprocess.run([str(lbfgs_contract_binary), case], check=True, capture_output=True, text=True)
