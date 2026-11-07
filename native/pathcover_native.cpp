// Native PathCover front end. Upstream provenance: third_party/pathcover_native/SOURCE.md.
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <Eigen/Dense>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "corridor/PathCover.hpp"
#include "corridor/VoxelGrid.hpp"
#include "nanovoxmap/nanovoxmap.hpp"
#include "occupancy_inflation/point_cloud_inflater.h"
#include "distance_map_planner/distance_map_planner.h"
#include "gcopter_solver.hpp"
#include "pathcover_boxes.h"
#include "pathcover_grid.h"
#include "pathcover_endpoint.h"
#include "pathcover_yaw.h"
#ifdef OCCUPANCY_INFLATION_WITH_CUDA
#include "pathcover_distance_map.h"
#include "pathcover_boxes_cuda.h"
#endif

namespace py = pybind11;
using Vec3 = Eigen::Vector3d;
using Points = Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>;
using Clock = std::chrono::steady_clock;
using HPoly = Eigen::Matrix<double, Eigen::Dynamic, 4>;
using Coefficients = Eigen::Matrix<double, 3, 6>;
using TimeInitialization = trajectory_server::GcopterSolver<3>::TimeInitialization;
// Numerical diagnostic allowance for the original, unit-normal SFC.
// This does not change the 1e-6 m position-penalty smoothing or CTBR hard bounds.
constexpr double kSfcFeasibilityToleranceM = 1e-4;
class PathCoverPlanningError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

static double milliseconds(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

static double speedAcceptanceLimit(double physical_limit) {
    return 1.001 * physical_limit;
}

static py::dict timeInitializationDict(const TimeInitialization &value) {
    py::dict result;
    result["status"] = value.status;
    result["evaluations"] = value.evaluations;
    result["scale"] = value.scale;
    result["elapsed_ms"] = value.elapsedMs;
    return result;
}

struct Options {
    Vec3 lower, upper;
    double resolution, search_resolution, radius, radius_z, max_velocity, nominal_velocity, mass, gravity;
    double thrust_min, thrust_max, body_rate_max, reference_body_rate_margin, tilt_max;
    double potential_radius, dmp_heuristic_weight, time_weight, safety_margin, endpoint_connection_radius;
    double relative_tolerance, alpha, position_penalty_smoothing;
    int horizon, pieces_per_polytope, integral_resolution, max_iterations, num_threads;
    std::size_t cuda_min_grid_voxels;
    unsigned int random_seed;
    bool use_cuda, dmp_xy_heuristic;
    Eigen::VectorXd penalty_weights;
    explicit Options(const py::dict &values) {
        lower = values["bounds_lower"].cast<Vec3>();
        upper = values["bounds_upper"].cast<Vec3>();
        resolution = values["resolution"].cast<double>();
        search_resolution = values["search_resolution"].cast<double>();
        endpoint_connection_radius = values["endpoint_connection_radius"].cast<double>();
        radius = values["robot_radius"].cast<double>();
        radius_z = values["robot_radius_z"].cast<double>();
        max_velocity = values["max_velocity"].cast<double>();
        nominal_velocity = values["nominal_velocity"].cast<double>();
        mass = values["mass"].cast<double>();
        gravity = values["gravity"].cast<double>();
        thrust_min = values["thrust_min"].cast<double>();
        thrust_max = values["thrust_max"].cast<double>();
        body_rate_max = values["body_rate_max"].cast<double>();
        reference_body_rate_margin = values["reference_body_rate_margin"].cast<double>();
        if (!std::isfinite(body_rate_max) || !std::isfinite(reference_body_rate_margin)
            || reference_body_rate_margin < 0 || reference_body_rate_margin >= body_rate_max) {
            throw std::invalid_argument("PathCover rate limits require finite 0 <= reference_body_rate_margin < body_rate_max");
        }
        tilt_max = values["tilt_max"].cast<double>();
        potential_radius = values["potential_radius"].cast<double>();
        dmp_heuristic_weight = values["dmp_heuristic_weight"].cast<double>();
        dmp_xy_heuristic = values["dmp_xy_heuristic"].cast<bool>();
        time_weight = values["time_weight"].cast<double>();
        safety_margin = values["safety_margin"].cast<double>();
        relative_tolerance = values["relative_tolerance"].cast<double>();
        position_penalty_smoothing = values["position_penalty_smoothing_m"].cast<double>();
        alpha = values["risp_alpha"].cast<double>();
        horizon = values["max_corridors"].cast<int>();
        pieces_per_polytope = values["pieces_per_polytope"].cast<int>();
        integral_resolution = values["integral_resolution"].cast<int>();
        max_iterations = values["max_iterations"].cast<int>();
        num_threads = values["num_threads"].cast<int>();
        random_seed = values["random_seed"].cast<unsigned int>();
        use_cuda = values["use_cuda"].cast<bool>();
        cuda_min_grid_voxels = values["cuda_min_grid_voxels"].cast<std::size_t>();
        penalty_weights = values["penalty_weights"].cast<Eigen::VectorXd>();
    }
};

struct MappingStats {
    double mapping_ms = 0.0, inflation_ms = 0.0, grid_ms = 0.0;
    std::size_t occupied_voxels = 0, inflated_voxels = 0, geometry_boxes = 0;
    bool gpu_mapping = false, gpu_inflation = false, gpu_box_compression = false;
};

struct PlanResult {
    Points path;
    Vec3 local_goal;
    std::vector<HPoly> corridors;
    std::vector<Coefficients> coefficients;
    Eigen::VectorXd durations;
    // The legacy scalar is the minimum actual face offset per corridor.
    Eigen::VectorXd applied_reference_margins;
    Eigen::VectorXd reference_margin_caps;
    std::vector<Eigen::VectorXd> applied_reference_face_margins;
    Eigen::VectorXi corridor_indices;
    Eigen::Matrix<double, 6, 1> constraint_metrics = Eigen::Matrix<double, 6, 1>::Zero();
    double cost = 0.0, search_ms = 0.0, corridor_ms = 0.0, trajectory_ms = 0.0;
    double jps_ms = 0.0, distance_map_ms = 0.0, dmp_ms = 0.0, dmp_cost = 0.0;
    double dmp_xy_preprocessing_ms = 0.0; // Included in dmp_ms, not an extra leaf.
    double endpoint_connection_ms = 0.0;
    std::size_t endpoint_connection_visited_cells = 0, start_endpoint_anchors = 0, goal_endpoint_anchors = 0;
    double initial_speed = 0.0, speed_acceptance_limit = 0.0;
    int optimizer_status = 0;
    bool gpu_distance_map = false, gpu_legal_moves = false;
    bool dmp_xy_heuristic = false, dmp_xy_cache_hit = false;
    bool optimizer_quadrature_feasible = false;
    TimeInitialization time_initialization;
};

struct OptimizationFailure {
    Eigen::Matrix3d initial, terminal;
    std::vector<HPoly> corridors;
    int status;
    std::string stage;
    Eigen::VectorXd trial_durations;
    Eigen::Matrix<double, 6, 1> constraint_metrics = Eigen::Matrix<double, 6, 1>::Constant(NAN);
    bool constraint_finite = false;
    std::vector<Coefficients> coefficients;
    Eigen::VectorXd durations;
    Eigen::VectorXi corridor_indices;
    double cost = NAN;
    TimeInitialization time_initialization;
};

class NativeFrontend {
    Options options_;
    std::unique_ptr<NanoVoxMap::OccupancyMapd> map_;
    occupancy_inflation::PointCloudInflater inflater_;
    std::shared_ptr<JPS::VoxelMapUtil> grid_;
    std::unique_ptr<DMPlanner<3>> dmp_;
#ifdef OCCUPANCY_INFLATION_WITH_CUDA
    std::unique_ptr<pathcover_cuda::DistanceMap> distance_cuda_;
    std::unique_ptr<pathcover_cuda::BoxCompressor> boxes_cuda_;
    std::vector<std::array<int, 4>> distance_mask_;
#endif
    Eigen::Vector3i dimensions_, grid_origin_index_;
    pathcover_grid::ResolutionRatio grid_resolution_ratio_;
    Vec3 center_lower_, center_upper_, grid_origin_;
    std::vector<int> occupied_indices_;
    std::vector<Vec3> geometry_box_centers_, geometry_box_halfwidths_;
    MappingStats last_update_;
    bool map_dirty_ = true;
    std::optional<OptimizationFailure> failed_problem_;

    void make_map() {
        map_ = std::make_unique<NanoVoxMap::OccupancyMapd>(
            options_.resolution, 0.7, 0.4, 0.1192, 0.971,
            options_.use_cuda ? NanoVoxMap::Backend::kCuda : NanoVoxMap::Backend::kCpu,
            false);
        map_->setCudaMergeThreads(options_.num_threads > 0
            ? static_cast<unsigned>(options_.num_threads) : 0U);
    }

public:
    explicit NativeFrontend(const py::dict &values) : options_(values) {
        if (options_.resolution <= 0 || options_.search_resolution < options_.resolution
            || !std::isfinite(options_.endpoint_connection_radius) || options_.endpoint_connection_radius <= 0
            || (options_.upper.array() <= options_.lower.array()).any()
            || options_.horizon < 1 || options_.pieces_per_polytope < 1
            || options_.mass <= 0 || !std::isfinite(options_.max_velocity) || options_.max_velocity <= 0
            || !std::isfinite(options_.nominal_velocity) || options_.nominal_velocity <= 0
            || options_.nominal_velocity > options_.max_velocity
            || options_.radius < 0 || options_.radius_z < 0
            || options_.thrust_max <= options_.thrust_min
            || !std::isfinite(options_.dmp_heuristic_weight) || options_.dmp_heuristic_weight < 0
            || options_.penalty_weights.size() != 5
            || !std::isfinite(options_.position_penalty_smoothing)
            || options_.position_penalty_smoothing <= 0) {
            throw std::invalid_argument("Invalid native PathCover configuration");
        }
        const Vec3 clearance(options_.radius, options_.radius, options_.radius_z);
        // Both lattices are world-zero anchored. Physical SFC/seed/connector
        // bounds are exact; only the coarse search rounds inward to free cells.
        grid_origin_index_ = (options_.lower.array() / options_.search_resolution)
            .floor().cast<int>().matrix();
        grid_origin_ = grid_origin_index_.cast<double>() * options_.search_resolution;
        grid_resolution_ratio_ = pathcover_grid::resolutionRatio(options_.resolution, options_.search_resolution);
        center_lower_ = options_.lower + clearance;
        center_upper_ = options_.upper - clearance;
        const Vec3 search_lower = ((center_lower_.array() / options_.search_resolution)
            .ceil()).matrix() * options_.search_resolution;
        const Vec3 search_upper = ((center_upper_.array() / options_.search_resolution)
            .floor()).matrix() * options_.search_resolution;
        if ((search_upper.array() <= search_lower.array()).any()) {
            throw std::invalid_argument("PathCover body clearance leaves no workspace");
        }
        make_map();
        occupancy_inflation::InflationParams parameters;
        parameters.resolution = options_.resolution;
        parameters.radius_xy = options_.radius;
        parameters.radius_z = options_.radius_z;
        parameters.shape = occupancy_inflation::InflationShape::kEllipsoid;
        parameters.input_voxel_volumes = true;
        parameters.incremental = false; // NanoVoxMap can clear previously occupied cells.
        parameters.use_cuda = options_.use_cuda;
        parameters.num_threads = options_.num_threads;
        parameters.cuda_min_grid_voxels = options_.cuda_min_grid_voxels;
        parameters.output_bounds_enable = true;
        parameters.output_min = options_.lower;
        parameters.output_max = options_.upper;
        std::string error;
        if (!inflater_.configure(parameters, &error)) {
            throw std::invalid_argument(error);
        }
        if (options_.use_cuda && !inflater_.cudaEnabled()) {
            throw std::runtime_error("Native PathCover CUDA backend is unavailable");
        }
        std::vector<signed char> empty;
        dimensions_ = VoxelGrid::initializeGrid<double, 3>(
            grid_origin_, options_.upper, options_.search_resolution, empty);
        // Virtual workspace boundaries are physical surfaces too. Mark them once;
        // Projection records only previously free cells, so cloud resets preserve these.
        for (int z = 0; z < dimensions_.z(); ++z) {
            for (int y = 0; y < dimensions_.y(); ++y) {
                for (int x = 0; x < dimensions_.x(); ++x) {
                    const Vec3 center = grid_origin_
                        + (Vec3(x, y, z).array() + 0.5).matrix() * options_.search_resolution;
                    if ((center.array() < search_lower.array()).any()
                        || (center.array() > search_upper.array()).any()) {
                        empty[x + dimensions_.x() * (y + dimensions_.y() * z)] = 100;
                    }
                }
            }
        }
        grid_ = std::make_shared<JPS::VoxelMapUtil>();
        grid_->setMap(grid_origin_, dimensions_, empty, options_.search_resolution);
        dmp_ = std::make_unique<DMPlanner<3>>(false);
        dmp_->setXYHeuristic(options_.dmp_xy_heuristic);
        dmp_->setPotentialRadius(Vec3::Constant(options_.potential_radius));
        // Weight 0 is Dijkstra, 1 is admissible A*, and >1 explicitly trades
        // shortest-path optimality for fewer expansions on the same legal graph.
        dmp_->setEps(options_.dmp_heuristic_weight);
#ifdef OCCUPANCY_INFLATION_WITH_CUDA
        if (options_.use_cuda) {
            distance_cuda_ = std::make_unique<pathcover_cuda::DistanceMap>();
            boxes_cuda_ = std::make_unique<pathcover_cuda::BoxCompressor>();
            for (const auto& entry : dmp_->potentialMask(options_.search_resolution)) {
                distance_mask_.push_back({entry.first.x(), entry.first.y(), entry.first.z(), entry.second});
            }
        }
#endif
    }

    void reset() {
        make_map();
        inflater_.reset();
        VoxelGrid::resetOccupancy(grid_->map_, occupied_indices_);
        geometry_box_centers_.clear();
        geometry_box_halfwidths_.clear();
        map_dirty_ = true;
        last_update_ = MappingStats{};
    }

    MappingStats update_cloud(const Points &points, const Vec3 &sensor_origin) {
        const auto begin = Clock::now();
        std::vector<Vec3> endpoints;
        endpoints.reserve(points.rows());
        for (Eigen::Index i = 0; i < points.rows(); ++i) {
            if (points.row(i).allFinite()) endpoints.emplace_back(points.row(i));
        }
        map_->insertPointCloud(sensor_origin, endpoints);
        last_update_.mapping_ms = milliseconds(begin);
        last_update_.occupied_voxels = map_->numOccupied();
        last_update_.gpu_mapping = map_->activeBackend() == NanoVoxMap::Backend::kCuda;
        const auto inflate_begin = Clock::now();
        occupancy_inflation::InflationStats stats;
        inflater_.inflateOccupancySnapshot(*map_, &stats);
        if (!stats.ok) throw std::runtime_error("PathCover occupancy inflation failed");
        last_update_.inflation_ms = milliseconds(inflate_begin);
        last_update_.gpu_inflation = stats.gpu_dilation;
        last_update_.inflated_voxels = stats.inflated_voxels;
        const auto grid_begin = Clock::now();
        last_update_.gpu_box_compression = false;
        if (stats.changed) {
            // Exact union of inflated fine cells. Integer runs are merged without
            // filling gaps; no point-cloud serialization or second inflation.
            const auto& fine_grid = inflater_.outputGrid();
            std::vector<pathcover_boxes::Box> boxes;
#ifdef OCCUPANCY_INFLATION_WITH_CUDA
            if (options_.use_cuda && !fine_grid.empty()
                && fine_grid.wordCount() * occupancy_inflation::kWordBits >= options_.cuda_min_grid_voxels) {
                boxes = boxes_cuda_->compress(fine_grid);
                last_update_.gpu_box_compression = true;
            } else
#endif
            {
                boxes = pathcover_boxes::compress(fine_grid);
            }
            VoxelGrid::resetOccupancy(grid_->map_, occupied_indices_);
            pathcover_grid::coarsenOccupiedBoxes(
                boxes, fine_grid.window(), grid_resolution_ratio_, grid_origin_index_, dimensions_,
                grid_->map_, occupied_indices_);
            geometry_box_centers_.clear();
            geometry_box_halfwidths_.clear();
            geometry_box_centers_.reserve(boxes.size());
            geometry_box_halfwidths_.reserve(boxes.size());
            for (const auto& box : boxes) {
                geometry_box_centers_.push_back(
                    (box.lower.cast<double>() + box.upper.cast<double>()) * (0.5 * options_.resolution));
                geometry_box_halfwidths_.push_back(
                    (box.upper - box.lower).cast<double>() * (0.5 * options_.resolution));
            }
            map_dirty_ = true;
        }
        last_update_.geometry_boxes = geometry_box_centers_.size();
        last_update_.grid_ms = milliseconds(grid_begin);
        return last_update_;
    }

    MappingStats last_update() const { return last_update_; }

    Points inflated_cloud() const {
        // Keep the historical packed-float diagnostic representation. Planning
        // consumes the logical bit-grid and does not serialize this point cloud.
        Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor> packed(inflater_.occupiedCount(), 3);
        inflater_.serialize(reinterpret_cast<uint8_t *>(packed.data()), 3 * sizeof(float), false);
        return packed.cast<double>();
    }

    Points obstacle_centers() const {
        // Diagnostics expose the original fine cells, independent of the exact
        // box partition used by RISP. This copy is outside the planning path.
        const auto centers = pathcover_endpoint::FineGeometry(
            inflater_.outputGrid(), options_.resolution, center_lower_, center_upper_).occupiedCenters();
        Points output(centers.size(), 3);
        for (std::size_t i = 0; i < centers.size(); ++i) output.row(i) = centers[i];
        return output;
    }

    Points search_obstacle_centers() const {
        // Preserve the historical sorted diagnostic order without sorting or
        // materializing coarse centers during every planning update.
        auto indices = occupied_indices_;
        std::sort(indices.begin(), indices.end());
        Points output(indices.size(), 3);
        for (std::size_t i = 0; i < indices.size(); ++i) {
            const int index = indices[i];
            const int x = index % dimensions_.x();
            const int y = (index / dimensions_.x()) % dimensions_.y();
            const int z = index / (dimensions_.x() * dimensions_.y());
            output.row(i) = grid_origin_
                + (Vec3(x, y, z).array() + 0.5).matrix() * options_.search_resolution;
        }
        return output;
    }

    std::pair<uint64_t, uint64_t> map_fingerprint() const {
        const auto value = map_->fingerprint();
        return {value.voxels, value.occupied};
    }

    std::vector<int8_t> potential_map() const {
        const auto map = dmp_->getMapUtil();
        return map ? map->map_ : std::vector<int8_t>{};
    }

    py::object failed_problem() const {
        if (!failed_problem_) return py::none();
        py::dict value;
        value["initial_pva"] = py::cast(failed_problem_->initial);
        value["terminal_pva"] = py::cast(failed_problem_->terminal);
        value["corridors"] = py::cast(failed_problem_->corridors);
        value["optimizer_status"] = failed_problem_->status;
        value["stage"] = failed_problem_->stage;
        value["trial_durations"] = py::cast(failed_problem_->trial_durations);
        value["constraint_metrics"] = py::cast(failed_problem_->constraint_metrics);
        value["constraint_metrics_order"] = py::make_tuple(
            "sfc_max_m", "speed_max_m_s", "thrust_min_n", "thrust_max_n", "tilt_max_rad", "rate_max_rad_s");
        value["constraint_finite"] = failed_problem_->constraint_finite;
        value["time_initialization"] = timeInitializationDict(failed_problem_->time_initialization);
        py::dict bounds;
        bounds["sfc_max_m"] = 0.0;
        bounds["speed_max_m_s"] = options_.max_velocity;
        bounds["thrust_min_n"] = options_.thrust_min;
        bounds["thrust_max_n"] = options_.thrust_max;
        bounds["tilt_max_rad"] = options_.tilt_max;
        bounds["rate_max_rad_s"] = options_.body_rate_max;
        value["constraint_bounds"] = std::move(bounds);
        value["reference_body_rate_margin_rad_s"] = options_.reference_body_rate_margin;
        value["optimizer_body_rate_target_rad_s"] = options_.body_rate_max - options_.reference_body_rate_margin;
        value["body_rate_physical_limit_rad_s"] = options_.body_rate_max;
        py::dict allowed_error;
        allowed_error["sfc_m"] = kSfcFeasibilityToleranceM;
        allowed_error["speed_relative"] = 1e-3;
        allowed_error["speed_m_s"] = 1e-3 * options_.max_velocity;
        allowed_error["thrust_n"] = 1e-3;
        allowed_error["tilt_rad"] = 1e-3;
        allowed_error["body_rate_rad_s"] = 1e-3;
        value["soft_constraint_allowed_error"] = std::move(allowed_error);
        const double initial_speed = failed_problem_->initial.col(1).norm();
        const double peak_speed = failed_problem_->constraint_metrics(1);
        py::dict speed;
        speed["initial_speed_m_s"] = initial_speed;
        speed["nominal_limit_m_s"] = options_.nominal_velocity;
        speed["physical_limit_m_s"] = options_.max_velocity;
        speed["acceptance_limit_m_s"] = speedAcceptanceLimit(options_.max_velocity);
        speed["peak_speed_m_s"] = peak_speed;
        speed["nominal_excess_m_s"] = std::max(peak_speed - options_.nominal_velocity, 0.0);
        value["speed_feasibility"] = std::move(speed);
        if (!failed_problem_->coefficients.empty()) {
            py::dict candidate;
            candidate["coefficients"] = py::cast(failed_problem_->coefficients);
            candidate["durations"] = py::cast(failed_problem_->durations);
            candidate["corridor_indices"] = py::cast(failed_problem_->corridor_indices);
            candidate["objective"] = failed_problem_->cost;
            value["candidate"] = std::move(candidate);
        }
        return value;
    }

    PlanResult plan(const Vec3 &position, const Vec3 &velocity,
                    const Vec3 &acceleration, const Vec3 &goal, const Points &initial_motion_seed) {
        failed_problem_.reset();
        if (!position.allFinite() || !velocity.allFinite() || !acceleration.allFinite() || !goal.allFinite()) {
            throw std::invalid_argument("PathCover state and goal must be finite");
        }
        if ((initial_motion_seed.rows() != 0 && initial_motion_seed.rows() != 3)
            || !initial_motion_seed.allFinite()
            || (initial_motion_seed.rows() && (initial_motion_seed.row(0).transpose() - position).norm() > 1e-9)) {
            throw std::invalid_argument("PathCover initial motion seed must have three finite points starting at the current position");
        }
        std::vector<Vec3> seed_region;
        for (Eigen::Index i = 0; i < initial_motion_seed.rows(); ++i) {
            const Vec3 point = initial_motion_seed.row(i);
            if ((point.array() < center_lower_.array()).any() || (point.array() > center_upper_.array()).any()) {
                throw PathCoverPlanningError("PathCover initial motion seed violates workspace body clearance");
            }
            seed_region.push_back(point);
        }
        if ((position.array() < center_lower_.array()).any()
            || (position.array() > center_upper_.array()).any()
            || (goal.array() < center_lower_.array()).any()
            || (goal.array() > center_upper_.array()).any()) {
            throw PathCoverPlanningError("PathCover state or goal violates workspace body clearance");
        }
        PlanResult result;
        result.initial_speed = velocity.norm();
        result.speed_acceptance_limit = speedAcceptanceLimit(options_.max_velocity);
        const auto search_begin = Clock::now();
        const pathcover_endpoint::FineGeometry fine_geometry(
            inflater_.outputGrid(), options_.resolution, center_lower_, center_upper_);
        if (!fine_geometry.pointFree(position) || !fine_geometry.pointFree(goal)) {
            throw PathCoverPlanningError("PathCover start or goal lies in inflated occupancy");
        }
        std::vector<Vec3> path;
        if ((goal - position).norm() < 1e-8) {
            path.push_back(position);
        } else {
            const pathcover_endpoint::CoarseGrid coarse{
                grid_->map_, dimensions_, grid_origin_, options_.search_resolution};
            const auto start_connections = pathcover_endpoint::connect(
                position, fine_geometry, coarse, options_.endpoint_connection_radius);
            const auto goal_connections = pathcover_endpoint::connect(
                goal, fine_geometry, coarse, options_.endpoint_connection_radius);
            result.endpoint_connection_visited_cells = start_connections.visited_cells + goal_connections.visited_cells;
            result.start_endpoint_anchors = start_connections.anchors.size();
            result.goal_endpoint_anchors = goal_connections.anchors.size();
            result.endpoint_connection_ms = milliseconds(search_begin);
            if (start_connections.anchors.empty() || goal_connections.anchors.empty()) {
                throw PathCoverPlanningError("PathCover endpoint has no fine-grid connection to the coarse graph within endpoint_connection_radius");
            }
            const auto distance_map_begin = Clock::now();
            if (map_dirty_) {
#ifdef OCCUPANCY_INFLATION_WITH_CUDA
                if (distance_cuda_) {
                    auto field = distance_cuda_->compute(grid_->map_,
                        {dimensions_.x(), dimensions_.y(), dimensions_.z()}, distance_mask_);
                    dmp_->setPotentialMap(grid_, std::move(field.potential), std::move(field.legal_moves));
                } else
#endif
                dmp_->setMap(grid_, position);
            }
            result.gpu_distance_map = options_.use_cuda;
            result.gpu_legal_moves = options_.use_cuda && dmp_->hasLegalMoveMask();
            map_dirty_ = false;
            result.distance_map_ms = milliseconds(distance_map_begin);
            const auto dmp_begin = Clock::now();
            const bool ordinary_endpoints = start_connections.endpoint_in_coarse_free_cell
                && goal_connections.endpoint_in_coarse_free_cell;
            bool found;
            if (ordinary_endpoints) {
                // Preserve the original single-source/single-goal path and cost.
                found = dmp_->computeGlobalRawPath(position, goal);
            } else {
                DMPlanner<3>::EndpointAnchors starts, goals;
                for (const auto& anchor : start_connections.anchors) starts.emplace_back(anchor.coarse_index, anchor.length);
                for (const auto& anchor : goal_connections.anchors) goals.emplace_back(anchor.coarse_index, anchor.length);
                found = dmp_->computeGlobalRawPath(starts, goals);
            }
            if (!found) {
                throw PathCoverPlanningError("PathCover global DMP did not find a path");
            }
            result.dmp_ms = milliseconds(dmp_begin);
            result.dmp_cost = dmp_->getPathCost();
            result.dmp_xy_heuristic = dmp_->hasXYHeuristic();
            result.dmp_xy_cache_hit = dmp_->xyHeuristicCacheHit();
            result.dmp_xy_preprocessing_ms = dmp_->xyHeuristicMilliseconds();
            auto append_point = [&](const Vec3& point) {
                if (path.empty() || (point - path.back()).norm() > 1e-9) path.push_back(point);
            };
            if (ordinary_endpoints) append_point(position);
            else {
                const auto& connection = start_connections.anchors.at(dmp_->getStartAnchorIndex());
                for (const auto& point : connection.points) append_point(point);
            }
            for (const auto& point : dmp_->getRawPath()) append_point(point);
            if (ordinary_endpoints) append_point(goal);
            else {
                const auto& connection = goal_connections.anchors.at(dmp_->getGoalAnchorIndex());
                for (auto it = connection.points.rbegin(); it != connection.points.rend(); ++it) append_point(*it);
            }
        }
        result.search_ms = milliseconds(search_begin);
        result.path.resize(path.size(), 3);
        for (std::size_t i = 0; i < path.size(); ++i) result.path.row(i) = path[i];

        const auto corridor_begin = Clock::now();
        Eigen::Matrix<double, Eigen::Dynamic, 3> bounds_A(6, 3);
        Eigen::VectorXd bounds_b(6);
        bounds_A.topRows(3) = -Eigen::Matrix3d::Identity();
        bounds_A.bottomRows(3) = Eigen::Matrix3d::Identity();
        bounds_b.head(3) = -center_lower_;
        bounds_b.tail(3) = center_upper_;
        std::vector<Eigen::Matrix<double, Eigen::Dynamic, 3>> As;
        std::vector<Eigen::VectorXd> bs;
        std::vector<Vec3> seeds;
        std::vector<std::vector<int>> decay;
        PathCover::setRandomSeed(options_.random_seed);
        sdlp::permutation_rng.seed(options_.random_seed);
        try {
            PathCover::pathCover<double, 3>(geometry_box_centers_, path, bounds_A, bounds_b,
                As, bs, seeds, decay, options_.horizon, options_.alpha, geometry_box_halfwidths_, seed_region);
        } catch (const PathCover::SeedRegionCollision &error) {
            throw PathCoverPlanningError(error.what());
        }
        result.local_goal = seeds.back();
        for (std::size_t i = 0; i < As.size(); ++i) {
            HPoly h(As[i].rows(), 4);
            h.leftCols<3>() = As[i];
            h.col(3) = -bs[i];
            h.array().colwise() /= As[i].rowwise().norm().array();
            result.corridors.push_back(std::move(h));
        }
        result.corridor_ms = milliseconds(corridor_begin);

        const auto trajectory_begin = Clock::now();
        if ((result.local_goal - position).norm() < 1e-8 && velocity.norm() < 1e-8
            && acceleration.norm() < 1e-8) {
            Coefficients hold = Coefficients::Zero();
            hold.col(5) = position;
            result.coefficients.push_back(hold);
            result.durations = Eigen::VectorXd::Constant(1, 0.1);
            result.corridor_indices = Eigen::VectorXi::Zero(1);
            result.applied_reference_margins = Eigen::VectorXd::Zero(result.corridors.size());
            result.reference_margin_caps = Eigen::VectorXd::Zero(result.corridors.size());
            for (const auto &corridor : result.corridors) {
                result.applied_reference_face_margins.push_back(Eigen::VectorXd::Zero(corridor.rows()));
            }
            result.constraint_metrics <<
                (result.corridors.front().leftCols<3>() * position + result.corridors.front().col(3)).maxCoeff(),
                0.0, options_.mass * options_.gravity, options_.mass * options_.gravity, 0.0, 0.0;
        } else {
            sdlp::permutation_rng.seed(options_.random_seed);
            trajectory_server::GcopterSolver<3> solver;
            Eigen::Matrix3d initial, terminal = Eigen::Matrix3d::Zero();
            initial.col(0) = position;
            initial.col(1) = velocity;
            initial.col(2) = acceleration;
            terminal.col(0) = result.local_goal;
            Eigen::VectorXd bounds(5), physics(6);
            bounds << options_.nominal_velocity, options_.body_rate_max - options_.reference_body_rate_margin, options_.tilt_max,
                options_.thrust_min, options_.thrust_max;
            physics << options_.mass, options_.gravity, 0.0, 0.0, 0.0, 1e-3;
            if (!solver.setup(options_.time_weight, initial, terminal, result.corridors,
                              options_.safety_margin, 2.0, options_.pieces_per_polytope,
                              0.01, options_.integral_resolution, bounds,
                              options_.penalty_weights, physics)) {
                failed_problem_ = OptimizationFailure{initial, terminal, result.corridors, -1, "setup"};
                throw PathCoverPlanningError("PathCover GCOPTER corridor decomposition failed");
            }
            solver.setPositionPenaltySmoothing(options_.position_penalty_smoothing);
            result.applied_reference_margins = solver.getAppliedMargins();
            result.reference_margin_caps = solver.getMarginCaps();
            result.applied_reference_face_margins = solver.getAppliedFaceMargins();
            Trajectory<5> trajectory;
            result.cost = solver.optimize(trajectory, options_.relative_tolerance, options_.max_iterations);
            result.optimizer_status = solver.getStatus();
            result.time_initialization = solver.getTimeInitialization();
            const auto &quality = solver.getConstraintMetrics();
            result.constraint_metrics = quality.values();
            if (!std::isfinite(result.cost) || trajectory.getPieceNum() == 0) {
                failed_problem_ = OptimizationFailure{
                    initial, terminal, result.corridors, result.optimizer_status, "optimize", solver.getTrialDurations(),
                    result.constraint_metrics, quality.finite};
                failed_problem_->time_initialization = result.time_initialization;
                throw PathCoverPlanningError(std::string("PathCover GCOPTER optimization failed: ")
                    + lbfgs::lbfgs_strerror(result.optimizer_status));
            }
            // These diagnostics reuse the final objective evaluation. Finite
            // soft-constraint residuals do not reject an optimized reference.
            const auto &m = result.constraint_metrics;
            if (!quality.finite || !m.allFinite()) {
                failed_problem_ = OptimizationFailure{
                    initial, terminal, result.corridors, result.optimizer_status, "nonfinite", solver.getTrialDurations(),
                    result.constraint_metrics, quality.finite};
                failed_problem_->time_initialization = result.time_initialization;
                // Copy the existing rejected polynomial only on failure. This
                // enables exact offline replay without another optimization.
                failed_problem_->durations = trajectory.getDurations();
                failed_problem_->corridor_indices = solver.getPiecePolytopeIndices();
                failed_problem_->cost = result.cost;
                for (int i = 0; i < trajectory.getPieceNum(); ++i) {
                    failed_problem_->coefficients.push_back(trajectory[i].getCoeffMat());
                }
                throw PathCoverPlanningError("PathCover GCOPTER constraint evaluation is nonfinite");
            }
            result.durations = trajectory.getDurations();
            result.corridor_indices = solver.getPiecePolytopeIndices();
            for (int i = 0; i < trajectory.getPieceNum(); ++i) {
                result.coefficients.push_back(trajectory[i].getCoeffMat());
            }
        }
        const auto &m = result.constraint_metrics;
        result.optimizer_quadrature_feasible = m(0) <= kSfcFeasibilityToleranceM
            && m(1) <= result.speed_acceptance_limit
            && m(2) >= options_.thrust_min - 1e-3 && m(3) <= options_.thrust_max + 1e-3
            && m(4) <= options_.tilt_max + 1e-3 && m(5) <= options_.body_rate_max + 1e-3;
        result.trajectory_ms = milliseconds(trajectory_begin);
        return result;
    }
};

PYBIND11_MODULE(_pathcover_native, m) {
    // The public Python base is importable without loading this extension.
    // Registering a native subtype preserves specific catches in A0 and A1/A2.
    auto planning_error_base = py::module_::import("isaaclab_ipc.pathcover_native")
        .attr("PathCoverPlanningError");
    py::register_exception<PathCoverPlanningError>(
        m, "PathCoverPlanningError", planning_error_base.ptr());
    m.def("bounded_body_yaw_profile", [](py::array_t<double, py::array::c_style | py::array::forcecast> coefficients,
                                          py::array_t<double, py::array::c_style | py::array::forcecast> durations,
                                          double initial_yaw, double body_r_max, double sample_dt,
                                          double low_speed, double gravity) {
        if (coefficients.ndim() != 3 || coefficients.shape(1) != 3 || coefficients.shape(2) != 6
            || durations.ndim() != 1 || coefficients.shape(0) != durations.shape(0)) {
            throw std::invalid_argument("MINCO heading input requires [piece,3,6] coefficients and [piece] durations");
        }
        pathcover_yaw::Profile profile;
        {
            py::gil_scoped_release release;
            try {
                const pathcover_yaw::PolynomialGeometry polynomial(
                    coefficients.data(), durations.data(), durations.size(), gravity);
                pathcover_yaw::Settings settings;
                settings.body_r_max = body_r_max;
                settings.base_dt = sample_dt;
                settings.low_speed = low_speed;
                profile = pathcover_yaw::Builder(polynomial, settings).run(
                    polynomial.boundaries.back(), initial_yaw, polynomial.boundaries);
            } catch (const pathcover_yaw::ProfileError& error) {
                throw PathCoverPlanningError(error.what());
            }
        }
        py::array_t<double> output({static_cast<py::ssize_t>(profile.times.size()), py::ssize_t(3)});
        auto values = output.mutable_unchecked<2>();
        for (std::size_t i = 0; i < profile.times.size(); ++i) {
            values(i, 0) = profile.times[i];
            values(i, 1) = profile.values[i];
            values(i, 2) = profile.rates[i];
        }
        return output;
    }, py::arg("coefficients"), py::arg("durations"), py::arg("initial_yaw"),
       py::arg("body_r_max"), py::arg("sample_dt"), py::arg("low_speed"), py::arg("gravity"));
    m.attr("upstream_revision") = "8723a05d1993bf0199a453b47dc6ee167d493c23";
    m.attr("cuda_built") = occupancy_inflation::cuda::built();
    py::class_<MappingStats>(m, "MappingStats")
        .def_readonly("mapping_ms", &MappingStats::mapping_ms)
        .def_readonly("inflation_ms", &MappingStats::inflation_ms)
        .def_readonly("grid_ms", &MappingStats::grid_ms)
        .def_readonly("occupied_voxels", &MappingStats::occupied_voxels)
        .def_readonly("inflated_voxels", &MappingStats::inflated_voxels)
        .def_readonly("geometry_boxes", &MappingStats::geometry_boxes)
        .def_readonly("gpu_mapping", &MappingStats::gpu_mapping)
        .def_readonly("gpu_inflation", &MappingStats::gpu_inflation)
        .def_readonly("gpu_box_compression", &MappingStats::gpu_box_compression);
    py::class_<PlanResult>(m, "PlanResult")
        .def_readonly("path", &PlanResult::path)
        .def_readonly("local_goal", &PlanResult::local_goal)
        .def_readonly("corridors", &PlanResult::corridors)
        .def_readonly("coefficients", &PlanResult::coefficients)
        .def_readonly("durations", &PlanResult::durations)
        .def_readonly("applied_reference_margins", &PlanResult::applied_reference_margins)
        .def_readonly("reference_margin_caps", &PlanResult::reference_margin_caps)
        .def_readonly("applied_reference_face_margins", &PlanResult::applied_reference_face_margins)
        .def_readonly("corridor_indices", &PlanResult::corridor_indices)
        .def_readonly("constraint_metrics", &PlanResult::constraint_metrics)
        .def_readonly("optimizer_quadrature_feasible", &PlanResult::optimizer_quadrature_feasible)
        .def_readonly("initial_speed", &PlanResult::initial_speed)
        .def_readonly("speed_acceptance_limit", &PlanResult::speed_acceptance_limit)
        .def_property_readonly("time_initialization", [](const PlanResult &value) {
            return timeInitializationDict(value.time_initialization);
        })
        .def_readonly("cost", &PlanResult::cost)
        .def_readonly("optimizer_status", &PlanResult::optimizer_status)
        .def_readonly("gpu_distance_map", &PlanResult::gpu_distance_map)
        .def_readonly("gpu_legal_moves", &PlanResult::gpu_legal_moves)
        .def_readonly("search_ms", &PlanResult::search_ms)
        .def_readonly("jps_ms", &PlanResult::jps_ms)
        .def_readonly("distance_map_ms", &PlanResult::distance_map_ms)
        .def_readonly("dmp_ms", &PlanResult::dmp_ms)
        .def_readonly("dmp_cost", &PlanResult::dmp_cost)
        .def_readonly("dmp_xy_heuristic", &PlanResult::dmp_xy_heuristic)
        .def_readonly("dmp_xy_cache_hit", &PlanResult::dmp_xy_cache_hit)
        .def_readonly("dmp_xy_preprocessing_ms", &PlanResult::dmp_xy_preprocessing_ms)
        .def_readonly("endpoint_connection_ms", &PlanResult::endpoint_connection_ms)
        .def_readonly("endpoint_connection_visited_cells", &PlanResult::endpoint_connection_visited_cells)
        .def_readonly("start_endpoint_anchors", &PlanResult::start_endpoint_anchors)
        .def_readonly("goal_endpoint_anchors", &PlanResult::goal_endpoint_anchors)
        .def_readonly("corridor_ms", &PlanResult::corridor_ms)
        .def_readonly("trajectory_ms", &PlanResult::trajectory_ms);
    py::class_<NativeFrontend>(m, "NativeFrontend")
        .def(py::init<const py::dict &>())
        .def("reset", &NativeFrontend::reset, py::call_guard<py::gil_scoped_release>())
        .def("update_cloud", &NativeFrontend::update_cloud, py::call_guard<py::gil_scoped_release>())
        .def("plan", &NativeFrontend::plan, py::call_guard<py::gil_scoped_release>())
        .def("inflated_cloud", &NativeFrontend::inflated_cloud, py::call_guard<py::gil_scoped_release>())
        .def("obstacle_centers", &NativeFrontend::obstacle_centers, py::call_guard<py::gil_scoped_release>())
        .def("search_obstacle_centers", &NativeFrontend::search_obstacle_centers, py::call_guard<py::gil_scoped_release>())
        .def("map_fingerprint", &NativeFrontend::map_fingerprint, py::call_guard<py::gil_scoped_release>())
        .def("potential_map", &NativeFrontend::potential_map, py::call_guard<py::gil_scoped_release>())
        .def("failed_problem", &NativeFrontend::failed_problem)
        .def_property_readonly("last_update", &NativeFrontend::last_update);
}
