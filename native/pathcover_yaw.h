#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

namespace pathcover_yaw {

using Vector3 = std::array<double, 3>;

class ProfileError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

struct Geometry {
    double heading, speed, zx, zy, zz, t1, t2;
};

inline Geometry geometry(const Vector3& velocity, const Vector3& acceleration,
                         const Vector3& jerk, double gravity) {
    for (int axis = 0; axis < 3; ++axis) {
        if (!std::isfinite(velocity[axis]) || !std::isfinite(acceleration[axis])
            || !std::isfinite(jerk[axis])) {
            throw ProfileError("PathCover heading requires finite trajectory derivatives");
        }
    }
    const Vector3 force{acceleration[0], acceleration[1], acceleration[2] + gravity};
    const double norm = std::sqrt(force[0] * force[0] + force[1] * force[1] + force[2] * force[2]);
    if (!std::isfinite(norm) || norm <= 0 || force[2] <= 0) {
        throw ProfileError("PathCover heading requires a finite upright thrust direction");
    }
    const Vector3 z{force[0] / norm, force[1] / norm, force[2] / norm};
    const double projection = z[0] * jerk[0] + z[1] * jerk[1] + z[2] * jerk[2];
    const Vector3 zdot{(jerk[0] - z[0] * projection) / norm,
                       (jerk[1] - z[1] * projection) / norm,
                       (jerk[2] - z[2] * projection) / norm};
    const double t1 = z[2] * zdot[0] - z[0] * zdot[2];
    const double t2 = z[1] * zdot[2] - z[2] * zdot[1];
    if (!std::isfinite(t1) || !std::isfinite(t2)) {
        throw ProfileError("PathCover heading thrust-direction derivative is non-finite");
    }
    return {std::atan2(velocity[1], velocity[0]), std::hypot(velocity[0], velocity[1]),
            z[0], z[1], z[2], t1, t2};
}

struct Settings {
    double body_r_max = 1.5;
    double base_dt = 0.02;
    double low_speed = 0.1;
    double angle_atol = 1e-7;
    double rate_atol = 1e-4;
};

// With h=(cos(yaw),sin(yaw),0), c=z.h and s²=1-c²,
// body r=(z_z*yaw_rate+offset)/s². Clip r pointwise, then integrate
// yaw_rate=(r*s²-offset)/z_z; no constant-rate intersection is required.
inline std::array<double, 2> rateGain(const Geometry& g, double yaw, const Settings& settings) {
    const double cosine = std::cos(yaw), sine = std::sin(yaw);
    const double c = g.zx * cosine + g.zy * sine;
    const double s2 = 1 - c * c;
    if (!std::isfinite(yaw) || s2 <= 0) {
        throw ProfileError("PathCover heading frame is singular or non-finite");
    }
    const double offset = c * (g.t1 * sine + g.t2 * cosine);
    // Periodic feedback avoids a rate jump when the shortest angle crosses pi.
    // The small-error gain is unchanged; the antipodal equilibrium is unstable,
    // so this does not assert global heading convergence.
    const double error = std::sin(g.heading - yaw);
    // Continuous fading avoids the old derivative jump at the low-speed cutoff.
    const double nominal = error / settings.base_dt * std::min(g.speed / settings.low_speed, 1.0);
    const double body_r = std::clamp((g.zz * nominal + offset) / s2,
                                     -settings.body_r_max, settings.body_r_max);
    const double rate = (body_r * s2 - offset) / g.zz;
    if (!std::isfinite(rate)) throw ProfileError("PathCover heading rate is non-finite");
    return {rate, g.zz / s2};
}

inline std::array<double, 2> hermite(double y0, double y1, double d0, double d1,
                                     double duration, double fraction) {
    const double s = fraction, s2 = s * s, s3 = s2 * s;
    return {(2 * s3 - 3 * s2 + 1) * y0 + (s3 - 2 * s2 + s) * duration * d0
                + (-2 * s3 + 3 * s2) * y1 + (s3 - s2) * duration * d1,
            6 * s * (1 - s) * (y1 - y0) / duration
                + (3 * s2 - 4 * s + 1) * d0 + (3 * s2 - 2 * s) * d1};
}

struct Profile {
    std::vector<double> times, values, rates;
};

template <class Evaluate>
class Builder {
    Evaluate evaluate_;
    Settings settings_;
    Profile output_;

    double rhs(const Geometry& geometry, double yaw) const {
        return rateGain(geometry, yaw, settings_)[0];
    }

    double rk4(double yaw, double step, const Geometry& g0, const Geometry& gm,
               const Geometry& g1) const {
        const double a = rhs(g0, yaw);
        const double b = rhs(gm, yaw + step * a / 2);
        const double c = rhs(gm, yaw + step * b / 2);
        const double d = rhs(g1, yaw + step * c);
        return yaw + step / 6 * (a + 2 * b + 2 * c + d);
    }

    double advance(double time, double step, double yaw, const std::array<Geometry, 5>& g, int depth) {
        const double whole = rk4(yaw, step, g[0], g[2], g[4]);
        const double middle = rk4(yaw, step / 2, g[0], g[1], g[2]);
        const double end = rk4(middle, step / 2, g[2], g[3], g[4]);
        const double d0 = rhs(g[0], yaw), d1 = rhs(g[4], end);
        double defect = 0;
        for (int i = 1; i < 4; ++i) {
            const auto sample = hermite(yaw, end, d0, d1, step, 0.25 * i);
            const auto exact = rateGain(g[i], sample[0], settings_);
            defect = std::max(defect, std::abs(sample[1] - exact[0]) * exact[1]);
        }
        // These local integration/interpolation errors are not a global
        // certificate of the translational or attitude trajectory constraints.
        if (std::abs(end - whole) / 15 > settings_.angle_atol || defect > settings_.rate_atol) {
            if (depth >= 20) throw ProfileError("PathCover heading ODE local error did not converge");
            const Geometry a = evaluate_(time + step * 0.125);
            const Geometry b = evaluate_(time + step * 0.375);
            const Geometry c = evaluate_(time + step * 0.625);
            const Geometry d = evaluate_(time + step * 0.875);
            const double refined_middle = advance(time, step / 2, yaw, {g[0], a, g[1], b, g[2]}, depth + 1);
            return advance(time + step / 2, step / 2, refined_middle, {g[2], c, g[3], d, g[4]}, depth + 1);
        }
        output_.times.push_back(time + step);
        output_.values.push_back(end);
        output_.rates.push_back(d1);
        return end;
    }

public:
    Builder(Evaluate evaluate, Settings settings) : evaluate_(std::move(evaluate)), settings_(settings) {}

    Profile run(double duration, double initial_yaw, const std::vector<double>& boundaries) {
        if (!std::isfinite(initial_yaw) || !std::isfinite(duration) || duration <= 0
            || !std::isfinite(settings_.body_r_max) || settings_.body_r_max <= 0
            || !std::isfinite(settings_.base_dt) || settings_.base_dt <= 0
            || !std::isfinite(settings_.low_speed) || settings_.low_speed <= 0) {
            throw std::invalid_argument("Invalid PathCover heading configuration");
        }
        const int count = std::max(1, static_cast<int>(std::ceil(duration / settings_.base_dt)));
        std::vector<double> times = boundaries;
        for (int i = 0; i <= count; ++i) times.push_back(duration * i / count);
        std::sort(times.begin(), times.end());
        times.erase(std::unique(times.begin(), times.end()), times.end());
        output_.times = {0};
        output_.values = {initial_yaw};
        output_.rates = {rhs(evaluate_(0), initial_yaw)};
        double yaw = initial_yaw;
        for (std::size_t i = 0; i + 1 < times.size(); ++i) {
            const double time = times[i], step = times[i + 1] - time;
            std::array<Geometry, 5> g;
            for (int j = 0; j < 5; ++j) g[j] = evaluate_(time + step * 0.25 * j);
            yaw = advance(time, step, yaw, g, 0);
        }
        return std::move(output_);
    }
};

// Zero-copy access to [piece, axis, descending quintic coefficient]. The caller
// owns the coefficient array for the duration of the synchronous build.
class PolynomialGeometry {
    const double* coefficients_;
    double gravity_;

public:
    std::vector<double> boundaries{0};

    PolynomialGeometry(const double* coefficients, const double* durations, std::size_t count,
                       double gravity) : coefficients_(coefficients), gravity_(gravity) {
        if (!std::isfinite(gravity) || gravity <= 0 || count == 0) {
            throw std::invalid_argument("Invalid PathCover polynomial or gravity");
        }
        boundaries.reserve(count + 1);
        for (std::size_t piece = 0; piece < count; ++piece) {
            if (!std::isfinite(durations[piece]) || durations[piece] <= 0) {
                throw std::invalid_argument("MINCO durations must be finite and positive");
            }
            for (int coefficient = 0; coefficient < 18; ++coefficient) {
                if (!std::isfinite(coefficients[18 * piece + coefficient])) {
                    throw std::invalid_argument("MINCO coefficients must be finite");
                }
            }
            boundaries.push_back(boundaries.back() + durations[piece]);
        }
    }

    Geometry operator()(double time) const {
        const auto iterator = std::lower_bound(boundaries.begin() + 1, boundaries.end(), time);
        const auto piece = std::min(static_cast<std::size_t>(iterator - boundaries.begin() - 1),
                                    boundaries.size() - 2);
        const double t = time - boundaries[piece];
        Vector3 velocity, acceleration, jerk;
        for (int axis = 0; axis < 3; ++axis) {
            const double* c = coefficients_ + piece * 18 + axis * 6;
            velocity[axis] = (((5 * c[0] * t + 4 * c[1]) * t + 3 * c[2]) * t + 2 * c[3]) * t + c[4];
            acceleration[axis] = ((20 * c[0] * t + 12 * c[1]) * t + 6 * c[2]) * t + 2 * c[3];
            jerk[axis] = (60 * c[0] * t + 24 * c[1]) * t + 6 * c[2];
        }
        return geometry(velocity, acceleration, jerk, gravity_);
    }
};

}  // namespace pathcover_yaw
