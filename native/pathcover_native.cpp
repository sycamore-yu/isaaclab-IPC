// Native PathCover front end. Upstream provenance: third_party/pathcover_native/SOURCE.md.
#include <pybind11/eigen.h>
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
#include "pathcover_grid.h"
#ifdef OCCUPANCY_INFLATION_WITH_CUDA
#include "pathcover_distance_map.h"
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
    double thrust_min, thrust_max, body_rate_max, tilt_max;
    double potential_radius, dmp_heuristic_weight, time_weight, safety_margin;
    double relative_tolerance, alpha, position_penalty_smoothing;
    int horizon, pieces_per_polytope, integral_resolution, max_iterations, num_threads;
    std::size_t cuda_min_grid_voxels;
    unsigned int random_seed;
    bool use_cuda;
    Eigen::VectorXd penalty_weights;
    explicit Options(const py::dict &values) {
        lower = values["bounds_lower"].cast<Vec3>();
        upper = values["bounds_upper"].cast<Vec3>();
        resolution = values["resolution"].cast<double>();
        search_resolution = values["search_resolution"].cast<double>();
        radius = values["robot_radius"].cast<double>();
        radius_z = values["robot_radius_z"].cast<double>();
        max_velocity = values["max_velocity"].cast<double>();
        nominal_velocity = values["nominal_velocity"].cast<double>();
        mass = values["mass"].cast<double>();
        gravity = values["gravity"].cast<double>();
        thrust_min = values["thrust_min"].cast<double>();
        thrust_max = values["thrust_max"].cast<double>();
        body_rate_max = values["body_rate_max"].cast<double>();
        tilt_max = values["tilt_max"].cast<double>();
        potential_radius = values["potential_radius"].cast<double>();
        dmp_heuristic_weight = values["dmp_heuristic_weight"].cast<double>();
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
    std::size_t occupied_voxels = 0, inflated_voxels = 0;
    bool gpu_mapping = false, gpu_inflation = false;
};

struct PlanResult {
    Points path;
    Vec3 local_goal;
    std::vector<HPoly> corridors;
    std::vector<Coefficients> coefficients;
    Eigen::VectorXd durations;
    Eigen::VectorXd applied_reference_margins;
    Eigen::VectorXi corridor_indices;
    Eigen::Matrix<double, 6, 1> constraint_metrics = Eigen::Matrix<double, 6, 1>::Zero();
    double cost = 0.0, search_ms = 0.0, corridor_ms = 0.0, trajectory_ms = 0.0;
    double jps_ms = 0.0, distance_map_ms = 0.0, dmp_ms = 0.0, dmp_cost = 0.0;
    double initial_speed = 0.0, speed_acceptance_limit = 0.0;
    int optimizer_status = 0;
    bool gpu_distance_map = false;
    bool optimizer_quadrature_feasible = false;
    TimeInitialization time_initialization;
};

struct CloudReader {
    const std::vector<Vec3> &points;
    std::size_t size() const { return points.size(); }
    bool point(std::size_t i, float &x, float &y, float &z) const {
        x = points[i].x(); y = points[i].y(); z = points[i].z();
        return true;
    }
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
    std::vector<std::array<int, 4>> distance_mask_;
#endif
    Eigen::Vector3i dimensions_;
    Vec3 center_lower_, center_upper_, grid_origin_;
    std::vector<int> occupied_indices_;
    std::vector<Vec3> inflated_points_;
    std::vector<Vec3> obstacle_centers_;
    MappingStats last_update_;
    bool map_dirty_ = true;
    std::optional<OptimizationFailure> failed_problem_;

    void make_map() {
        map_ = std::make_unique<NanoVoxMap::OccupancyMapd>(
            options_.resolution, 0.7, 0.4, 0.1192, 0.971,
            options_.use_cuda ? NanoVoxMap::Backend::kCuda : NanoVoxMap::Backend::kCpu,
            false);
    }

public:
    explicit NativeFrontend(const py::dict &values) : options_(values) {
        if (options_.resolution <= 0 || options_.search_resolution < options_.resolution
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
        // Mapping, search and geometry share a lattice anchored at world zero.
        // Round the body's usable workspace inward to complete free grid cells.
        grid_origin_ = (options_.lower.array() / options_.search_resolution).floor().matrix()
            * options_.search_resolution;
        center_lower_ = (((options_.lower + clearance).array() / options_.search_resolution)
            .ceil()).matrix() * options_.search_resolution;
        center_upper_ = (((options_.upper - clearance).array() / options_.search_resolution)
            .floor()).matrix() * options_.search_resolution;
        if ((center_upper_.array() <= center_lower_.array()).any()) {
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
        // setOccupancy records only previously free cells, so cloud resets preserve these.
        for (int z = 0; z < dimensions_.z(); ++z) {
            for (int y = 0; y < dimensions_.y(); ++y) {
                for (int x = 0; x < dimensions_.x(); ++x) {
                    const Vec3 center = grid_origin_
                        + (Vec3(x, y, z).array() + 0.5).matrix() * options_.search_resolution;
                    if ((center.array() < center_lower_.array()).any()
                        || (center.array() > center_upper_.array()).any()) {
                        empty[x + dimensions_.x() * (y + dimensions_.y() * z)] = 100;
                    }
                }
            }
        }
        grid_ = std::make_shared<JPS::VoxelMapUtil>();
        grid_->setMap(grid_origin_, dimensions_, empty, options_.search_resolution);
        dmp_ = std::make_unique<DMPlanner<3>>(false);
        dmp_->setPotentialRadius(Vec3::Constant(options_.potential_radius));
        // Weight 0 is Dijkstra, 1 is admissible A*, and >1 explicitly trades
        // shortest-path optimality for fewer expansions on the same legal graph.
        dmp_->setEps(options_.dmp_heuristic_weight);
#ifdef OCCUPANCY_INFLATION_WITH_CUDA
        if (options_.use_cuda) {
            distance_cuda_ = std::make_unique<pathcover_cuda::DistanceMap>();
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
        inflated_points_.clear();
        obstacle_centers_.clear();
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
        auto occupied = map_->getOccupiedVoxels();
        last_update_.mapping_ms = milliseconds(begin);
        last_update_.occupied_voxels = occupied.size();
        last_update_.gpu_mapping = map_->activeBackend() == NanoVoxMap::Backend::kCuda;
        const auto inflate_begin = Clock::now();
        occupancy_inflation::InflationStats stats;
        inflater_.inflate(CloudReader{occupied}, &stats);
        if (!stats.ok) throw std::runtime_error("PathCover occupancy inflation failed");
        last_update_.inflation_ms = milliseconds(inflate_begin);
        last_update_.gpu_inflation = stats.gpu_dilation;
        last_update_.inflated_voxels = stats.inflated_voxels;
        const auto grid_begin = Clock::now();
        if (stats.changed) {
            // Upstream serialization is packed float XYZ, without ROS or PCL.
            Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor> packed(inflater_.occupiedCount(), 3);
            inflater_.serialize(reinterpret_cast<uint8_t *>(packed.data()), 3 * sizeof(float), false);
            inflated_points_.resize(packed.rows());
            for (Eigen::Index i = 0; i < packed.rows(); ++i) {
                inflated_points_[i] = packed.row(i).cast<double>();
            }
            VoxelGrid::resetOccupancy(grid_->map_, occupied_indices_);
            if (options_.search_resolution == options_.resolution) {
                VoxelGrid::setOccupancy<double, 3>(inflated_points_, grid_origin_,
                    options_.resolution, grid_->map_, dimensions_, occupied_indices_);
            } else {
                // Preserve exact fine-voxel AABB coverage; merge repeated coarse runs.
                pathcover_grid::coarsenOccupiedVolumes(
                    inflated_points_, grid_origin_, options_.resolution,
                    options_.search_resolution, dimensions_, grid_->map_, occupied_indices_);
            }
            std::sort(occupied_indices_.begin(), occupied_indices_.end());
            obstacle_centers_.clear();
            obstacle_centers_.reserve(occupied_indices_.size());
            for (int index : occupied_indices_) {
                const int x = index % dimensions_.x();
                const int y = (index / dimensions_.x()) % dimensions_.y();
                const int z = index / (dimensions_.x() * dimensions_.y());
                obstacle_centers_.push_back(grid_origin_
                    + (Vec3(x, y, z).array() + 0.5).matrix() * options_.search_resolution);
            }
            map_dirty_ = true;
        }
        last_update_.grid_ms = milliseconds(grid_begin);
        return last_update_;
    }

    MappingStats last_update() const { return last_update_; }

    Points inflated_cloud() const {
        Points output(inflated_points_.size(), 3);
        for (std::size_t i = 0; i < inflated_points_.size(); ++i) output.row(i) = inflated_points_[i];
        return output;
    }

    Points obstacle_centers() const {
        Points output(obstacle_centers_.size(), 3);
        for (std::size_t i = 0; i < obstacle_centers_.size(); ++i) output.row(i) = obstacle_centers_[i];
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
        if (grid_->isOccupied(grid_->floatToInt(position))
            || grid_->isOccupied(grid_->floatToInt(goal))) {
            throw PathCoverPlanningError("PathCover start or goal lies in inflated occupancy");
        }
        PlanResult result;
        result.initial_speed = velocity.norm();
        result.speed_acceptance_limit = speedAcceptanceLimit(options_.max_velocity);
        const auto search_begin = Clock::now();
        std::vector<Vec3> path;
        if ((goal - position).norm() < 1e-8) {
            path.push_back(position);
        } else {
            const auto distance_map_begin = Clock::now();
            if (map_dirty_) {
#ifdef OCCUPANCY_INFLATION_WITH_CUDA
                if (distance_cuda_) {
                    auto potential = distance_cuda_->compute(grid_->map_,
                        {dimensions_.x(), dimensions_.y(), dimensions_.z()}, distance_mask_);
                    dmp_->setPotentialMap(grid_, std::move(potential));
                } else
#endif
                dmp_->setMap(grid_, position);
            }
            result.gpu_distance_map = options_.use_cuda;
            map_dirty_ = false;
            result.distance_map_ms = milliseconds(distance_map_begin);
            const auto dmp_begin = Clock::now();
            if (!dmp_->computeGlobalPath(position, goal)) {
                throw PathCoverPlanningError("PathCover global DMP did not find a path");
            }
            result.dmp_ms = milliseconds(dmp_begin);
            result.dmp_cost = dmp_->getPathCost();
            path.push_back(position);
            for (const auto &p : dmp_->getRawPath()) {
                if ((p - path.back()).norm() > 1e-9) path.push_back(p);
            }
            if ((goal - path.back()).norm() > 1e-9) path.push_back(goal);
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
        try {
            PathCover::pathCover<double, 3>(obstacle_centers_, path, bounds_A, bounds_b,
                As, bs, seeds, decay, options_.horizon, options_.alpha, 0.5 * options_.search_resolution, seed_region);
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
            bounds << options_.nominal_velocity, options_.body_rate_max, options_.tilt_max,
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
    m.attr("upstream_revision") = "8723a05d1993bf0199a453b47dc6ee167d493c23";
    m.attr("cuda_built") = occupancy_inflation::cuda::built();
    py::class_<MappingStats>(m, "MappingStats")
        .def_readonly("mapping_ms", &MappingStats::mapping_ms)
        .def_readonly("inflation_ms", &MappingStats::inflation_ms)
        .def_readonly("grid_ms", &MappingStats::grid_ms)
        .def_readonly("occupied_voxels", &MappingStats::occupied_voxels)
        .def_readonly("inflated_voxels", &MappingStats::inflated_voxels)
        .def_readonly("gpu_mapping", &MappingStats::gpu_mapping)
        .def_readonly("gpu_inflation", &MappingStats::gpu_inflation);
    py::class_<PlanResult>(m, "PlanResult")
        .def_readonly("path", &PlanResult::path)
        .def_readonly("local_goal", &PlanResult::local_goal)
        .def_readonly("corridors", &PlanResult::corridors)
        .def_readonly("coefficients", &PlanResult::coefficients)
        .def_readonly("durations", &PlanResult::durations)
        .def_readonly("applied_reference_margins", &PlanResult::applied_reference_margins)
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
        .def_readonly("search_ms", &PlanResult::search_ms)
        .def_readonly("jps_ms", &PlanResult::jps_ms)
        .def_readonly("distance_map_ms", &PlanResult::distance_map_ms)
        .def_readonly("dmp_ms", &PlanResult::dmp_ms)
        .def_readonly("dmp_cost", &PlanResult::dmp_cost)
        .def_readonly("corridor_ms", &PlanResult::corridor_ms)
        .def_readonly("trajectory_ms", &PlanResult::trajectory_ms);
    py::class_<NativeFrontend>(m, "NativeFrontend")
        .def(py::init<const py::dict &>())
        .def("reset", &NativeFrontend::reset, py::call_guard<py::gil_scoped_release>())
        .def("update_cloud", &NativeFrontend::update_cloud, py::call_guard<py::gil_scoped_release>())
        .def("plan", &NativeFrontend::plan, py::call_guard<py::gil_scoped_release>())
        .def("inflated_cloud", &NativeFrontend::inflated_cloud, py::call_guard<py::gil_scoped_release>())
        .def("obstacle_centers", &NativeFrontend::obstacle_centers, py::call_guard<py::gil_scoped_release>())
        .def("map_fingerprint", &NativeFrontend::map_fingerprint, py::call_guard<py::gil_scoped_release>())
        .def("potential_map", &NativeFrontend::potential_map, py::call_guard<py::gil_scoped_release>())
        .def("failed_problem", &NativeFrontend::failed_problem)
        .def_property_readonly("last_update", &NativeFrontend::last_update);
}
