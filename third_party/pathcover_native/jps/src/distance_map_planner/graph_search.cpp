#include <cmath>
#include <array>
#include <cstdint>
#include <algorithm>
#include <stdexcept>
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
    : cMap_(cMap), xDim_(xDim), yDim_(yDim), zDim_(1), eps_(eps), cweight_(cweight),
      verbose_(verbose), use_2d_(true) {
  initializeNeighbours();
}

GraphSearch::GraphSearch(const int8_t *cMap, int xDim, int yDim, int zDim,
                         double eps, double cweight, bool verbose, const uint32_t* legal_moves)
    : cMap_(cMap), legal_moves_(legal_moves), xDim_(xDim), yDim_(yDim), zDim_(zDim), eps_(eps),
      cweight_(cweight), verbose_(verbose), use_2d_(false) {
  initializeNeighbours();
}

void GraphSearch::initializeNeighbours() {
  // Preserve the original x/y/z emission order and the exact sqrt edge costs.
  // Map-sized state arrays are initialized once by resetSearch(), not here.
  for (int x = -1; x <= 1; x++) {
    for (int y = -1; y <= 1; y++) {
      for (int z = use_2d_ ? 0 : -1; z <= (use_2d_ ? 0 : 1); z++) {
        if (x == 0 && y == 0 && z == 0)
          continue;
        const unsigned index = neighbourIndex(x, y, z);
        ns_[neighbour_count_++] = {
            x, y, z, x + y * xDim_ + z * xDim_ * yDim_,
            uint32_t{1} << index, kRequiredFreeMasks[index],
            std::sqrt(x * x + y * y + z * z)};
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
  if (!goal_anchors_.empty()) return getAnchorHeur(x, y, 0);
  const int dx = std::abs(x - xGoal_), dy = std::abs(y - yGoal_);
  const int lo = std::min(dx, dy), hi = std::max(dx, dy);
  return eps_ * (hi + (std::sqrt(2.0) - 1.0) * lo);
}

inline double GraphSearch::getHeur(int x, int y, int z) const {
  if (!goal_anchors_.empty()) return getAnchorHeur(x, y, z);
  // The unweighted 26-neighbour metric is consistent with every legal edge.
  // eps_ scales it for weighted A*; the scaled heuristic need not be consistent.
  const int dx = std::abs(x - xGoal_), dy = std::abs(y - yGoal_), dz = std::abs(z - zGoal_);
  const int lo = std::min({dx, dy, dz}), hi = std::max({dx, dy, dz});
  const int mid = dx + dy + dz - lo - hi;
  return eps_ * (hi + (std::sqrt(2.0) - 1.0) * mid + (std::sqrt(3.0) - std::sqrt(2.0)) * lo);
}

double GraphSearch::getAnchorHeur(int x, int y, int z) const {
  double lower_bound = std::numeric_limits<double>::infinity();
  for (const auto& goal : goal_anchors_) {
    const int dx = std::abs(x-goal.x), dy = std::abs(y-goal.y);
    double distance;
    if (use_2d_) {
      const int lo = std::min(dx,dy), hi = std::max(dx,dy);
      distance = hi + (std::sqrt(2.0)-1.0)*lo;
    } else {
      const int dz = std::abs(z-goal.z);
      const int lo = std::min({dx,dy,dz}), hi = std::max({dx,dy,dz});
      const int mid = dx+dy+dz-lo-hi;
      distance = hi + (std::sqrt(2.0)-1.0)*mid + (std::sqrt(3.0)-std::sqrt(2.0))*lo;
    }
    lower_bound = std::min(lower_bound, distance+goal.connector_cost);
  }
  return eps_*lower_bound;
}

void GraphSearch::resetSearch(std::vector<bool> in_region) {
  pq_.clear();
  path_.clear();
  const int cells = xDim_*yDim_*(use_2d_ ? 1 : zDim_);
  hm_.assign(cells, nullptr);
  seen_.assign(cells, false);
  virtual_goal_id_ = selected_start_anchor_ = selected_goal_anchor_ = -1;
  goal_anchors_.clear();
  start_anchor_indices_.clear();
  goal_anchor_indices_.clear();
  in_region_ = std::move(in_region);
  if (in_region_.empty()) {
    global_ = true;
    if (verbose_)
      printf("global planning!\n");
  } else {
    global_ = false;
    if (verbose_)
      printf("local planning!\n");
  }
}

double GraphSearch::plan(int xStart, int yStart, int xGoal, int yGoal,
                       std::vector<bool> in_region) {
  use_2d_ = true;
  resetSearch(std::move(in_region));

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
  resetSearch(std::move(in_region));

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

double GraphSearch::plan(const std::vector<GridAnchor>& starts,
                         const std::vector<GridAnchor>& goals) {
  resetSearch({});
  if (starts.empty() || goals.empty()) return std::numeric_limits<double>::infinity();
  auto valid = [&](const GridAnchor& anchor) {
    const bool free = use_2d_ ? anchor.z == 0 && isFree(anchor.x,anchor.y)
                             : isFree(anchor.x,anchor.y,anchor.z);
    return free && std::isfinite(anchor.connector_cost) && anchor.connector_cost >= 0;
  };
  for (const auto& anchor : starts)
    if (!valid(anchor)) throw std::invalid_argument("Invalid DMP start anchor");
  for (const auto& anchor : goals)
    if (!valid(anchor)) throw std::invalid_argument("Invalid DMP goal anchor");

  goal_anchors_ = goals;
  for (int i=0;i<static_cast<int>(goals.size());++i) {
    const auto& anchor=goals[i];
    const int id=coordToId(anchor.x,anchor.y,anchor.z);
    const auto found=goal_anchor_indices_.find(id);
    if (found==goal_anchor_indices_.end() || anchor.connector_cost < goals[found->second].connector_cost)
      goal_anchor_indices_[id]=i;
  }
  virtual_goal_id_=static_cast<int>(hm_.size());
  auto virtual_goal=std::make_shared<State>(virtual_goal_id_,0,0,0);
  virtual_goal->h=0.0;
  hm_.push_back(virtual_goal);
  seen_.push_back(true);
  for (int i=0;i<static_cast<int>(starts.size());++i) {
    const auto& anchor=starts[i];
    const int id=coordToId(anchor.x,anchor.y,anchor.z);
    const double g=anchor.connector_cost+cMap_[id];
    auto& node=hm_[id];
    if (!node) {
      node=std::make_shared<State>(id,anchor.x,anchor.y,anchor.z);
      node->h=getAnchorHeur(anchor.x,anchor.y,anchor.z);
      seen_[id]=true;
    }
    if (g < node->g) {
      node->g=g;
      start_anchor_indices_[id]=i;
      if (node->opened) pq_.increase(node->heapkey);
      else { node->heapkey=pq_.push(node); node->opened=true; }
    }
  }
  return search(-1,virtual_goal_id_);
}

double GraphSearch::plan(StatePtr &currNode_ptr, int start_id, int goal_id) {
  // Insert start node
  currNode_ptr->heapkey = pq_.push(currNode_ptr);
  currNode_ptr->opened = true;
  hm_[currNode_ptr->id] = currNode_ptr;
  seen_[currNode_ptr->id] = true;

  return search(start_id,goal_id);
}

double GraphSearch::search(int start_id, int goal_id) {
  StatePtr currNode_ptr;
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

    int goal_anchor=-1;
    if (virtual_goal_id_ >= 0) {
      const auto found=goal_anchor_indices_.find(currNode_ptr->id);
      if (found!=goal_anchor_indices_.end()) {
        goal_anchor=found->second;
        succ_ids.push_back(virtual_goal_id_);
        succ_costs.push_back(goal_anchors_[goal_anchor].connector_cost);
      }
    }

    // Process successors
    for (int s = 0; s < (int)succ_ids.size(); s++) {
      // see if we can improve the value of succstate
      StatePtr &child_ptr = hm_[succ_ids[s]];
      // Weighted A* uses a closed list: keep closed g/parent immutable.
      if (child_ptr->closed) continue;
      double tentative_gval = currNode_ptr->g + succ_costs[s];

      if (tentative_gval + 1e-12 < child_ptr->g) {
        child_ptr->parentId = currNode_ptr->id; // Assign new parent
        child_ptr->g = tentative_gval;          // Update gval
        if (child_ptr->id == virtual_goal_id_) selected_goal_anchor_=goal_anchor;

        // double fval = child_ptr->g + child_ptr->h;

        // if currently in OPEN, update
        if (child_ptr->opened && !child_ptr->closed)
          pq_.increase(child_ptr->heapkey); // update heap
        else // new node, add to heap
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

  if (virtual_goal_id_ >= 0) {
    // The virtual node is an endpoint connector, not a voxel waypoint.
    path_=recoverPath(hm_[currNode_ptr->parentId],-1);
    selected_start_anchor_=start_anchor_indices_.at(path_.back()->id);
  } else {
    path_ = recoverPath(currNode_ptr, start_id);
  }

  return currNode_ptr->g;
}

std::vector<StatePtr> GraphSearch::recoverPath(StatePtr node, int start_id) {
  std::vector<StatePtr> path;
  path.push_back(node);
  while (node && node->id != start_id && node->parentId >= 0) {
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
  if (!legal_moves_) {
    for (int i = 0; i < neighbour_count_; ++i) {
      const auto& d = ns_[i];
      const int x = curr->x + d.dx, y = curr->y + d.dy, z = curr->z + d.dz;
      const bool free = x >= 0 && x < xDim_ && y >= 0 && y < yDim_
          && (use_2d_ || (z >= 0 && z < zDim_))
          && cMap_[curr->id + d.index_offset] < val_occ_;
      if (free) free_mask |= d.bit;
    }
  }
  if (use_2d_) {
    for (int i = 0; i < neighbour_count_; ++i) {
      const auto& d = ns_[i];
      int new_x = curr->x + d.dx;
      int new_y = curr->y + d.dy;
      if ((free_mask & d.required_free) != d.required_free) continue;

      int new_id = curr->id + d.index_offset;
      if (!global_ && !in_region_[new_id])
        continue;
      // CLOSED is a search state, never an obstacle in the physical free mask.
      if (seen_[new_id] && hm_[new_id]->closed) continue;

      if (!seen_[new_id]) {
        seen_[new_id] = true;
        hm_[new_id] = std::make_shared<State>(new_id, new_x, new_y);
        hm_[new_id]->h = getHeur(new_x, new_y);
      }

      succ_ids.push_back(new_id);
      succ_costs.push_back(d.step + cweight_ * (cMap_[new_id]));
    }
  } else {
    const uint32_t legal = legal_moves_ ? legal_moves_[curr->id] : 0;
    for (int i = 0; i < neighbour_count_; ++i) {
      const auto& d = ns_[i];
      int new_x = curr->x + d.dx;
      int new_y = curr->y + d.dy;
      int new_z = curr->z + d.dz;
      if (legal_moves_ ? !(legal & d.bit) : (free_mask & d.required_free) != d.required_free) continue;

      int new_id = curr->id + d.index_offset;
      if (!global_ && !in_region_[new_id])
        continue;
      if (seen_[new_id] && hm_[new_id]->closed) continue;

      if (!seen_[new_id]) {
        seen_[new_id] = true;
        hm_[new_id] = std::make_shared<State>(new_id, new_x, new_y, new_z);
        hm_[new_id]->h = getHeur(new_x, new_y, new_z);
      }

      succ_ids.push_back(new_id);
      succ_costs.push_back(d.step + cweight_ * cMap_[new_id]);
    }
  }
}

std::vector<StatePtr> GraphSearch::getPath() const { return path_; }

std::vector<StatePtr> GraphSearch::getOpenSet() const {
  std::vector<StatePtr> ss;
  for (const auto &it : hm_) {
    if (it && it->id != virtual_goal_id_ && it->opened && !it->closed)
      ss.push_back(it);
  }
  return ss;
}

std::vector<StatePtr> GraphSearch::getCloseSet() const {
  std::vector<StatePtr> ss;
  for (const auto &it : hm_) {
    if (it && it->id != virtual_goal_id_ && it->closed)
      ss.push_back(it);
  }
  return ss;
}

std::vector<StatePtr> GraphSearch::getAllSet() const {
  std::vector<StatePtr> ss;
  for (const auto &it : hm_) {
    if (it && it->id != virtual_goal_id_)
      ss.push_back(it);
  }
  return ss;
}
