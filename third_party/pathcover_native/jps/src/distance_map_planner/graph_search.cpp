#include <cmath>
#include <array>
#include <cstdint>
#include <algorithm>
#include <distance_map_planner/graph_search.h>

using namespace DMP;

namespace {
constexpr unsigned neighbourIndex(int x, int y, int z) {
  return static_cast<unsigned>((x + 1) * 9 + (y + 1) * 3 + z + 1);
}
constexpr uint32_t requiredFreeMask(int dx, int dy, int dz) {
  uint32_t mask = 0;
  for (int sx = 0; sx < 2; ++sx)
    for (int sy = 0; sy < 2; ++sy)
      for (int sz = 0; sz < 2; ++sz) {
        const int x = sx * dx, y = sy * dy, z = sz * dz;
        if (x || y || z) mask |= uint32_t{1} << neighbourIndex(x, y, z);
      }
  return mask;
}
constexpr std::array<uint32_t, 27> requiredFreeMasks() {
  std::array<uint32_t, 27> result{};
  for (int x = -1; x <= 1; ++x)
    for (int y = -1; y <= 1; ++y)
      for (int z = -1; z <= 1; ++z)
        result[neighbourIndex(x, y, z)] = requiredFreeMask(x, y, z);
  return result;
}
constexpr auto kRequiredFreeMasks = requiredFreeMasks();
}


GraphSearch::GraphSearch(const int8_t *cMap, int xDim, int yDim, double eps,
                         double cweight, bool verbose)
    : cMap_(cMap), xDim_(xDim), yDim_(yDim), eps_(eps), cweight_(cweight),
      verbose_(verbose) {
  hm_.resize(xDim_ * yDim_);
  seen_.resize(xDim_ * yDim_, false);

  for (int x = -1; x <= 1; x++) {
    for (int y = -1; y <= 1; y++) {
      if (x == 0 && y == 0)
        continue;
      ns_.push_back(std::vector<int>{x, y});
    }
  }
}

GraphSearch::GraphSearch(const int8_t *cMap, int xDim, int yDim, int zDim,
                         double eps, double cweight, bool verbose)
    : cMap_(cMap), xDim_(xDim), yDim_(yDim), zDim_(zDim), eps_(eps),
      cweight_(cweight), verbose_(verbose) {
  hm_.resize(xDim_ * yDim_ * zDim_);
  seen_.resize(xDim_ * yDim_ * zDim_, false);

  // Set 3D neighbors
  for (int x = -1; x <= 1; x++) {
    for (int y = -1; y <= 1; y++) {
      for (int z = -1; z <= 1; z++) {
        if (x == 0 && y == 0 && z == 0)
          continue;
        ns_.push_back(std::vector<int>{x, y, z});
      }
    }
  }
}

inline int GraphSearch::coordToId(int x, int y) const { return x + y * xDim_; }

inline int GraphSearch::coordToId(int x, int y, int z) const {
  return x + y * xDim_ + z * xDim_ * yDim_;
}

inline bool GraphSearch::isFree(int x, int y) const {
  return x >= 0 && x < xDim_ && y >= 0 && y < yDim_ &&
         cMap_[coordToId(x, y)] < val_occ_;
}

inline bool GraphSearch::isFree(int x, int y, int z) const {
  return x >= 0 && x < xDim_ && y >= 0 && y < yDim_ && z >= 0 && z < zDim_ &&
         cMap_[coordToId(x, y, z)] < val_occ_;
}

inline double GraphSearch::getHeur(int x, int y) const {
  const int dx = std::abs(x - xGoal_), dy = std::abs(y - yGoal_);
  const int lo = std::min(dx, dy), hi = std::max(dx, dy);
  return eps_ * (hi + (std::sqrt(2.0) - 1.0) * lo);
}

inline double GraphSearch::getHeur(int x, int y, int z) const {
  // Exact obstacle-free 26-neighbour grid cost, consistent with every DMP edge.
  const int dx = std::abs(x - xGoal_), dy = std::abs(y - yGoal_), dz = std::abs(z - zGoal_);
  const int lo = std::min({dx, dy, dz}), hi = std::max({dx, dy, dz});
  const int mid = dx + dy + dz - lo - hi;
  return eps_ * (hi + (std::sqrt(2.0) - 1.0) * mid + (std::sqrt(3.0) - std::sqrt(2.0)) * lo);
}

double GraphSearch::plan(int xStart, int yStart, int xGoal, int yGoal,
                       std::vector<bool> in_region) {
  use_2d_ = true;
  pq_.clear();
  path_.clear();
  hm_.resize(xDim_ * yDim_);
  seen_.resize(xDim_ * yDim_, false);
  in_region_ = in_region;
  if (in_region.empty()) {
    global_ = true;
    if (verbose_)
      printf("global planning!\n");
  } else {
    global_ = false;
    if (verbose_)
      printf("local planning!\n");
  }

  // Set goal
  int goal_id = coordToId(xGoal, yGoal);
  xGoal_ = xGoal;
  yGoal_ = yGoal;

  // Set start node
  int start_id = coordToId(xStart, yStart);
  StatePtr currNode_ptr =
      std::make_shared<State>(State(start_id, xStart, yStart));
  currNode_ptr->g = cMap_[start_id];
  currNode_ptr->h = getHeur(xStart, yStart);

  return plan(currNode_ptr, start_id, goal_id);
}

double GraphSearch::plan(int xStart, int yStart, int zStart, int xGoal,
                         int yGoal, int zGoal, std::vector<bool> in_region) {
  use_2d_ = false;
  pq_.clear();
  path_.clear();
  hm_.resize(xDim_ * yDim_ * zDim_);
  seen_.resize(xDim_ * yDim_ * zDim_, false);
  in_region_ = in_region;
  if (in_region.empty()) {
    global_ = true;
    if (verbose_)
      printf("global planning!\n");
  } else {
    global_ = false;
    if (verbose_)
      printf("local planning!\n");
  }

  // Set goal
  int goal_id = coordToId(xGoal, yGoal, zGoal);
  xGoal_ = xGoal;
  yGoal_ = yGoal;
  zGoal_ = zGoal;

  // Set start node
  int start_id = coordToId(xStart, yStart, zStart);
  StatePtr currNode_ptr =
      std::make_shared<State>(State(start_id, xStart, yStart, zStart));
  currNode_ptr->g = cMap_[start_id];
  currNode_ptr->h = getHeur(xStart, yStart, zStart);

  return plan(currNode_ptr, start_id, goal_id);
}

double GraphSearch::plan(StatePtr &currNode_ptr, int start_id, int goal_id) {
  // Insert start node
  currNode_ptr->heapkey = pq_.push(currNode_ptr);
  currNode_ptr->opened = true;
  hm_[currNode_ptr->id] = currNode_ptr;
  seen_[currNode_ptr->id] = true;

  int expand_iteration = 0;
  std::vector<int> succ_ids;
  std::vector<double> succ_costs;
  succ_ids.reserve(26);
  succ_costs.reserve(26);
  while (true) {
    expand_iteration++;
    // get element with smallest cost
    currNode_ptr = pq_.top();
    pq_.pop();
    currNode_ptr->closed = true; // Add to closed list

    if (currNode_ptr->id == goal_id) {
      if (verbose_)
        printf("Goal Reached!!!!!!\n\n");
      break;
    }

    // printf("expand: %d, %d\n", currNode_ptr->x, currNode_ptr->y);
    succ_ids.clear();
    succ_costs.clear();
    // Get successors
    getSucc(currNode_ptr, succ_ids, succ_costs);

    // Process successors
    for (int s = 0; s < (int)succ_ids.size(); s++) {
      // see if we can improve the value of succstate
      StatePtr &child_ptr = hm_[succ_ids[s]];
      double tentative_gval = currNode_ptr->g + succ_costs[s];

      if (tentative_gval + 1e-12 < child_ptr->g) {
        child_ptr->parentId = currNode_ptr->id; // Assign new parent
        child_ptr->g = tentative_gval;          // Update gval

        // double fval = child_ptr->g + child_ptr->h;

        // if currently in OPEN, update
        if (child_ptr->opened && !child_ptr->closed)
          pq_.increase(child_ptr->heapkey); // update heap
        // if currently in CLOSED
        else if (child_ptr->opened && child_ptr->closed) {
          // Reopen an improved node instead of silently leaving a stale closed path.
          child_ptr->closed = false;
          child_ptr->heapkey = pq_.push(child_ptr);
        } else // new node, add to heap
        {
          // printf("add to open set: %d, %d\n", child_ptr->x, child_ptr->y);
          child_ptr->heapkey = pq_.push(child_ptr);
          child_ptr->opened = true;
        }
      } //
    }   // Process successors

    if (pq_.empty()) {
      if (verbose_)
        printf("Priority queue is empty!!!!!!\n\n");
      return std::numeric_limits<double>::infinity();
    }
  }

  if (verbose_) {
    printf("goal g: %f, h: %f!\n", currNode_ptr->g, currNode_ptr->h);
    printf("Expand [%d] nodes!\n", expand_iteration);
  }

  path_ = recoverPath(currNode_ptr, start_id);

  return currNode_ptr->g;
}

std::vector<StatePtr> GraphSearch::recoverPath(StatePtr node, int start_id) {
  std::vector<StatePtr> path;
  path.push_back(node);
  while (node && node->id != start_id) {
    node = hm_[node->parentId];
    // printf("waypoint g: %f, h: %f!\n", node->g, node->h);
    path.push_back(node);
  }

  return path;
}

void GraphSearch::getSucc(const StatePtr &curr, std::vector<int> &succ_ids,
                          std::vector<double> &succ_costs) {
  // A diagonal touches every nonempty axis-subset cell at an edge or corner.
  // Closed occupied voxel AABBs require all these cells to be free. Probe each
  // physical neighbour once; search-region membership applies only to the target.
  uint32_t free_mask = 0;
  for (const auto &d : ns_) {
    const int dz = use_2d_ ? 0 : d[2];
    const bool free = use_2d_ ? isFree(curr->x + d[0], curr->y + d[1])
                            : isFree(curr->x + d[0], curr->y + d[1], curr->z + dz);
    if (free) free_mask |= uint32_t{1} << neighbourIndex(d[0], d[1], dz);
  }
  if (use_2d_) {
    for (const auto &d : ns_) {
      int new_x = curr->x + d[0];
      int new_y = curr->y + d[1];
      const auto index = neighbourIndex(d[0], d[1], 0);
      const uint32_t required = kRequiredFreeMasks[index];
      if ((free_mask & required) != required) continue;

      int new_id = coordToId(new_x, new_y);
      if (!global_ && !in_region_[new_id])
        continue;

      if (!seen_[new_id]) {
        seen_[new_id] = true;
        hm_[new_id] = std::make_shared<State>(new_id, new_x, new_y);
        hm_[new_id]->h = getHeur(new_x, new_y);
      }

      succ_ids.push_back(new_id);
      succ_costs.push_back(std::sqrt(d[0] * d[0] + d[1] * d[1]) +
                           cweight_ * (cMap_[new_id]));
    }
  } else {
    for (const auto &d : ns_) {
      int new_x = curr->x + d[0];
      int new_y = curr->y + d[1];
      int new_z = curr->z + d[2];
      const auto index = neighbourIndex(d[0], d[1], d[2]);
      const uint32_t required = kRequiredFreeMasks[index];
      if ((free_mask & required) != required) continue;

      int new_id = coordToId(new_x, new_y, new_z);
      if (!global_ && !in_region_[new_id])
        continue;

      if (!seen_[new_id]) {
        seen_[new_id] = true;
        hm_[new_id] = std::make_shared<State>(new_id, new_x, new_y, new_z);
        hm_[new_id]->h = getHeur(new_x, new_y, new_z);
      }

      succ_ids.push_back(new_id);
      succ_costs.push_back(std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]) +
                           cweight_ * cMap_[new_id]);
    }
  }
}

std::vector<StatePtr> GraphSearch::getPath() const { return path_; }

std::vector<StatePtr> GraphSearch::getOpenSet() const {
  std::vector<StatePtr> ss;
  for (const auto &it : hm_) {
    if (it && it->opened && !it->closed)
      ss.push_back(it);
  }
  return ss;
}

std::vector<StatePtr> GraphSearch::getCloseSet() const {
  std::vector<StatePtr> ss;
  for (const auto &it : hm_) {
    if (it && it->closed)
      ss.push_back(it);
  }
  return ss;
}

std::vector<StatePtr> GraphSearch::getAllSet() const {
  std::vector<StatePtr> ss;
  for (const auto &it : hm_) {
    if (it)
      ss.push_back(it);
  }
  return ss;
}
