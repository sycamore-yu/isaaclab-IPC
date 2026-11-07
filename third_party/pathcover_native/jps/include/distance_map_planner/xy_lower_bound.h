#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <queue>
#include <stdexcept>
#include <utility>
#include <vector>

namespace DMP {

// Exact reverse distances on a relaxed XY graph. This owns no search state:
// GraphSearch continues to enforce the original 3D occupancy and edge costs.
class XYLowerBound3D {
public:
    struct Goal { int x, y; double connector_cost; };
    struct Update { bool cache_hit; double milliseconds; };

    void mapChanged() { projection_dirty_ = true; }
    const double* data() const { return distances_.data(); }

    Update update(const int8_t* potential, int nx, int ny, int nz,
                  double cweight, double epsilon, const std::vector<Goal>& goals) {
        const auto begin = std::chrono::steady_clock::now();
        if (nx <= 0 || ny <= 0 || nz <= 0 || !std::isfinite(cweight) || cweight < 0
            || !std::isfinite(epsilon) || epsilon < 0 || goals.empty()) {
            throw std::invalid_argument("XY lower bound requires valid dimensions, nonnegative weights and goals");
        }
        const std::array<int, 3> dimensions{nx, ny, nz};
        const bool dimensions_changed = dimensions != dimensions_;
        bool changed = !valid_ || dimensions_changed;
        const int plane = nx * ny;
        if (projection_dirty_ || dimensions_changed) {
            std::vector<int8_t> minimum(plane, 100);
            for (int z = 0; z < nz; ++z) {
                for (int i = 0; i < plane; ++i) {
                    const int8_t value = potential[i + plane * z];
                    if (value < 0) {
                        throw std::invalid_argument("XY lower bound requires nonnegative potential values");
                    }
                    if (value < 100) minimum[i] = std::min(minimum[i], value);
                }
            }
            changed = changed || minimum != minimum_;
            minimum_ = std::move(minimum);
            dimensions_ = dimensions;
            projection_dirty_ = false;
        }
        // A later invalid goal must not leave old reverse distances marked
        // valid for a projection that has already changed.
        if (changed) valid_ = false;
        // Sort/deduplicate only the relaxed target columns. The real 3D anchor
        // list, indices and connector fees remain unchanged in GraphSearch.
        std::vector<std::pair<int, double>> seeds;
        seeds.reserve(goals.size());
        for (const auto& goal : goals) {
            if (goal.x < 0 || goal.x >= nx || goal.y < 0 || goal.y >= ny
                || !std::isfinite(goal.connector_cost) || goal.connector_cost < 0
                || minimum_[goal.x + nx * goal.y] >= 100) {
                throw std::invalid_argument("XY goals require free columns and nonnegative connector costs");
            }
            seeds.emplace_back(goal.x + nx * goal.y, goal.connector_cost);
        }
        std::sort(seeds.begin(), seeds.end());
        seeds.erase(std::unique(seeds.begin(), seeds.end(),
            [](const auto& a, const auto& b) { return a.first == b.first; }), seeds.end());
        changed = changed || seeds != seeds_ || cweight != cweight_ || epsilon != epsilon_;
        if (changed) {
            valid_ = false;
            seeds_ = std::move(seeds);
            cweight_ = cweight;
            epsilon_ = epsilon;
            distances_.assign(plane, std::numeric_limits<double>::infinity());
            using Entry = std::pair<double, int>;
            std::priority_queue<Entry, std::vector<Entry>, std::greater<Entry>> open;
            for (const auto& seed : seeds_) {
                distances_[seed.first] = seed.second;
                open.emplace(seed.second, seed.first);
            }
            const auto free = [&](int x, int y) {
                return x >= 0 && x < nx && y >= 0 && y < ny && minimum_[x + nx * y] < 100;
            };
            while (!open.empty()) {
                const auto [cost, current] = open.top();
                open.pop();
                if (cost != distances_[current]) continue;
                const int vx = current % nx, vy = current / nx;
                for (int dx = -1; dx <= 1; ++dx) {
                    for (int dy = -1; dy <= 1; ++dy) {
                        if (!dx && !dy) continue;
                        const int ux = vx + dx, uy = vy + dy;
                        if (!free(ux, uy)) continue;
                        if (dx && dy && (!free(ux, vy) || !free(vx, uy))) continue;
                        const int predecessor = ux + nx * uy;
                        // Reverse relaxation pays the original forward
                        // destination v, not the predecessor u.
                        const double candidate = cost + std::sqrt(double(dx * dx + dy * dy))
                            + cweight * minimum_[current];
                        if (candidate < distances_[predecessor]) {
                            distances_[predecessor] = candidate;
                            open.emplace(candidate, predecessor);
                        }
                    }
                }
            }
            valid_ = true;
        }
        return {!changed, std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - begin).count()};
    }

private:
    bool projection_dirty_ = true, valid_ = false;
    std::array<int, 3> dimensions_{0, 0, 0};
    double cweight_ = 0.0, epsilon_ = 0.0;
    std::vector<int8_t> minimum_;
    std::vector<std::pair<int, double>> seeds_;
    std::vector<double> distances_;
};

} // namespace DMP
