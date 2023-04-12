#include <cmath>
#include <functional>
#include <limits>
#include <queue>
#include <random>
#include <unordered_set>
#include <utility>
#include <vector>

#include "simulator.h"
namespace FlexFlow {
#define PRINT_EDGE(e, n)                                                       \
  do {                                                                         \
    std::cout << "(" << e / n << ", " << e % n << ")";                         \
  } while (0);

#define INSERT_OR_ADD(_map, _key, _val)                                        \
  do {                                                                         \
    if ((_map).find(_key) == (_map).end()) {                                   \
      (_map)[(_key)] = _val;                                                   \
    } else {                                                                   \
      (_map)[(_key)] += _val;                                                  \
    }                                                                          \
  } while (0);

static std::random_device rd;
static std::mt19937 gen = std::mt19937(rd());
static std::uniform_real_distribution<float> unif(0, 1);

// for summing connections...
template <typename T>
static std::vector<T> operator+(std::vector<T> const &a,
                                std::vector<T> const &b) {
  assert(a.size() == b.size());

  std::vector<T> result;
  result.reserve(a.size());

  std::transform(a.begin(),
                 a.end(),
                 b.begin(),
                 std::back_inserter(result),
                 std::plus<T>());
  return result;
}

WeightedShortestPathRoutingStrategy::WeightedShortestPathRoutingStrategy(
    ConnectionMatrix const &c,
    std::map<size_t, CommDevice *> const &devmap,
    int total_devs)
    : conn(c), devmap(devmap), total_devs(total_devs) {}

EcmpRoutes WeightedShortestPathRoutingStrategy::get_routes(int src_node,
                                                           int dst_node) {
  int key = src_node * total_devs + dst_node;

  if (conn[key] > 0) {
    return std::make_pair(std::vector<float>({1}),
                          std::vector<Route>({Route({devmap.at(key)})}));
  }

  // one-shortest path routing
  std::vector<uint64_t> dist(total_devs, std::numeric_limits<uint64_t>::max());
  std::vector<int> prev(total_devs, -1);
  std::vector<bool> visited(total_devs, false);

  std::priority_queue<std::pair<uint64_t, uint64_t>,
                      std::vector<std::pair<uint64_t, uint64_t>>,
                      std::greater<std::pair<uint64_t, uint64_t>>>
      pq;
  pq.push(std::make_pair(dist[src_node], src_node));
  dist[src_node] = 0;

  while (!pq.empty()) {
    int min_node = pq.top().second;
    pq.pop();
    visited[min_node] = true;

    if (min_node == dst_node) {
      break;
    }

    for (int i = 0; i < total_devs; i++) {
      if (visited[i] || conn[min_node * total_devs + i] == 0) {
        continue;
      }
      float new_dist = dist[min_node] + 1; // numeric_limits<uint64_t>::max() /
                                           // get_bandwidth_bps(min_node, i);
      if (new_dist < dist[i] || (new_dist == dist[i] && unif(gen) < 0.5)) {
        dist[i] = new_dist;
        prev[i] = min_node;
        pq.push(std::make_pair(new_dist, i));
      }
    }
  }

  Route result = Route();
  int curr = dst_node;
  while (prev[curr] != -1) {
    result.insert(result.begin(), devmap.at(prev[curr] * total_devs + curr));
    curr = prev[curr];
  }
  assert(result.size() || src_node == dst_node);
  return std::make_pair(std::vector<float>{1}, std::vector<Route>{result});
}

void WeightedShortestPathRoutingStrategy::hop_count(int src_node,
                                                    int dst_node,
                                                    int &hop,
                                                    int &narrowest) {
  int key = src_node * total_devs + dst_node;

  if (conn[key] > 0) {
    hop = 0;
    narrowest = conn[key];
    return;
  }
  // one-shortest path routing
  std::vector<uint64_t> dist(total_devs, std::numeric_limits<uint64_t>::max());
  std::vector<int> prev(total_devs, -1);
  std::vector<bool> visited(total_devs, false);

  std::priority_queue<std::pair<uint64_t, uint64_t>,
                      std::vector<std::pair<uint64_t, uint64_t>>,
                      std::greater<std::pair<uint64_t, uint64_t>>>
      pq;
  pq.push(std::make_pair(dist[src_node], src_node));
  dist[src_node] = 0;
  while (!pq.empty()) {
    int min_node = pq.top().second;
    pq.pop();
    visited[min_node] = true;
    if (min_node == dst_node) {
      break;
    }
    for (int i = 0; i < total_devs; i++) {
      if (visited[i] || conn[min_node * total_devs + i] == 0) {
        continue;
      }
      float new_dist = dist[min_node] + 1; // numeric_limits<uint64_t>::max() /
                                           // get_bandwidth_bps(min_node, i);
      if (new_dist < dist[i]) {
        dist[i] = new_dist;
        prev[i] = min_node;
        pq.push(std::make_pair(new_dist, i));
      }
    }
  }
  hop = 0;
  narrowest = std::numeric_limits<int>::max();
  int curr = dst_node;
  while (prev[curr] != -1) {
    if (narrowest > conn[prev[curr] * total_devs + curr]) {
      narrowest = conn[prev[curr] * total_devs + curr];
    }
    hop++;
    curr = prev[curr];
  }
  assert(hop > 0 || src_node == dst_node);
}

std::vector<EcmpRoutes>
    WeightedShortestPathRoutingStrategy::get_routes_from_src(int src_node) {
  std::vector<uint64_t> dist(total_devs, std::numeric_limits<uint64_t>::max());
  std::vector<int> prev(total_devs, -1);
  std::vector<bool> visited(total_devs, false);

  std::priority_queue<std::pair<uint64_t, uint64_t>,
                      std::vector<std::pair<uint64_t, uint64_t>>,
                      std::greater<std::pair<uint64_t, uint64_t>>>
      pq;
  pq.push(std::make_pair(dist[src_node], src_node));
  dist[src_node] = 0;
  while (!pq.empty()) {
    int min_node = pq.top().second;
    pq.pop();
    visited[min_node] = true;

    for (int i = 0; i < total_devs; i++) {
      if (visited[i] || conn[min_node * total_devs + i] == 0) {
        continue;
      }
      float new_dist = dist[min_node] + 1; // numeric_limits<uint64_t>::max() /
                                           // get_bandwidth_bps(min_node, i);
      if (new_dist < dist[i]) {
        dist[i] = new_dist;
        prev[i] = min_node;
        pq.push(std::make_pair(new_dist, i));
      }
    }
  }
  std::vector<EcmpRoutes> final_result;
  for (int i = 0; i < total_devs; i++) {
    if (i == src_node) {
      final_result.emplace_back(
          std::make_pair(std::vector<float>{}, std::vector<Route>{}));
      continue;
    }
    Route result = Route();
    int curr = i;
    while (prev[curr] != -1) {
      result.insert(result.begin(), devmap.at(prev[curr] * total_devs + curr));
      curr = prev[curr];
    }
    assert(result.size() > 0);
    final_result.emplace_back(
        std::make_pair(std::vector<float>{1}, std::vector<Route>{result}));
  }
  return final_result;
}

std::vector<std::pair<int, int>>
    WeightedShortestPathRoutingStrategy::hop_count(int src_node) {
  std::vector<uint64_t> dist(total_devs, std::numeric_limits<uint64_t>::max());
  std::vector<int> prev(total_devs, -1);
  std::vector<bool> visited(total_devs, false);

  std::priority_queue<std::pair<uint64_t, uint64_t>,
                      std::vector<std::pair<uint64_t, uint64_t>>,
                      std::greater<std::pair<uint64_t, uint64_t>>>
      pq;
  pq.push(std::make_pair(dist[src_node], src_node));
  dist[src_node] = 0;
  while (!pq.empty()) {
    int min_node = pq.top().second;
    pq.pop();
    visited[min_node] = true;

    for (int i = 0; i < total_devs; i++) {
      if (visited[i] || conn[min_node * total_devs + i] == 0) {
        continue;
      }
      float new_dist = dist[min_node] + 1; // numeric_limits<uint64_t>::max() /
                                           // get_bandwidth_bps(min_node, i);
      if (new_dist < dist[i]) {
        dist[i] = new_dist;
        prev[i] = min_node;
        pq.push(std::make_pair(new_dist, i));
      }
    }
  }

  std::vector<std::pair<int, int>> result;
  for (int i = 0; i < total_devs; i++) {
    if (i == src_node) {
      result.emplace_back(std::make_pair(-1, 0));
      continue;
    }
    int hop = -1;
    int narrowest = 0;
    int curr = i;
    while (prev[curr] != -1) {
      if (!narrowest || (narrowest > conn[prev[curr] * total_devs + curr])) {
        narrowest = conn[prev[curr] * total_devs + curr];
      }
      hop++;
      curr = prev[curr];
    }
    result.emplace_back(std::make_pair(hop, narrowest));
  }
  return result;
}

ShortestPathNetworkRoutingStrategy::ShortestPathNetworkRoutingStrategy(
    ConnectionMatrix const &c,
    std::map<size_t, CommDevice *> const &devmap,
    int total_devs)
    : conn(c), devmap(devmap), total_devs(total_devs) {}

EcmpRoutes ShortestPathNetworkRoutingStrategy::get_routes(int src_node,
                                                          int dst_node) {
  int key = src_node * total_devs + dst_node;
  // std::cerr << "routing " << src_node << ", " << dst_node << std::endl;

  if (conn[key] > 0) {
    return std::make_pair(std::vector<float>({1}),
                          std::vector<Route>({Route({devmap.at(key)})}));
  }

  // one-shortest path routing
  std::vector<uint64_t> dist(total_devs, std::numeric_limits<uint64_t>::max());
  std::vector<int> prev(total_devs, -1);
  std::vector<bool> visited(total_devs, false);

  std::queue<uint64_t> q;
  q.push(src_node);
  dist[src_node] = 0;

  // BFS
  while (!q.empty()) {
    int min_node = q.front();
    q.pop();
    visited[min_node] = true;

    if (min_node == dst_node) {
      break;
    }

    for (int i = 0; i < total_devs; i++) {
      if (visited[i] || conn[min_node * total_devs + i] == 0) {
        continue;
      }
      float new_dist = dist[min_node] + 1;
      if (new_dist < dist[i] || (new_dist == dist[i] && unif(gen) < 0.5)) {
        dist[i] = new_dist;
        prev[i] = min_node;
        q.push(i);
      }
    }
  }

  Route result = Route();
  int curr = dst_node;
  while (prev[curr] != -1) {
    result.insert(result.begin(), devmap.at(prev[curr] * total_devs + curr));
    curr = prev[curr];
  }
  assert(result.size() || src_node == dst_node);
  return std::make_pair(std::vector<float>{1}, std::vector<Route>{result});
}

std::vector<EcmpRoutes>
    ShortestPathNetworkRoutingStrategy::get_routes_from_src(int src_node) {
  std::vector<uint64_t> dist(total_devs, std::numeric_limits<uint64_t>::max());
  std::vector<int> prev(total_devs, -1);
  std::vector<bool> visited(total_devs, false);

  std::queue<uint64_t> q;
  q.push(src_node);
  dist[src_node] = 0;

  // BFS
  while (!q.empty()) {
    int min_node = q.front();
    q.pop();
    visited[min_node] = true;

    for (int i = 0; i < total_devs; i++) {
      if (visited[i] || conn[min_node * total_devs + i] == 0) {
        continue;
      }
      float new_dist = dist[min_node] + 1;
      if (new_dist < dist[i] || (new_dist == dist[i] && unif(gen) < 0.5)) {
        dist[i] = new_dist;
        prev[i] = min_node;
        q.push(i);
      }
    }
  }

  std::vector<EcmpRoutes> final_result;
  for (int i = 0; i < total_devs; i++) {
    if (i == src_node) {
      final_result.emplace_back(
          std::make_pair(std::vector<float>{}, std::vector<Route>{}));
      continue;
    }
    Route result = Route();
    int curr = i;
    while (prev[curr] != -1) {
      result.insert(result.begin(), devmap.at(prev[curr] * total_devs + curr));
      curr = prev[curr];
    }
    // assert(result.size() > 0);
    final_result.emplace_back(
        std::make_pair(std::vector<float>{1}, std::vector<Route>{result}));
  }
  return final_result;
}

void ShortestPathNetworkRoutingStrategy::hop_count(int src_node,
                                                   int dst_node,
                                                   int &hop,
                                                   int &narrowest) {
  int key = src_node * total_devs + dst_node;

  if (conn[key] > 0) {
    hop = 0;
    narrowest = conn[key];
    return;
  }
  // one-shortest path routing
  std::vector<uint64_t> dist(total_devs, std::numeric_limits<uint64_t>::max());
  std::vector<int> prev(total_devs, -1);
  std::vector<bool> visited(total_devs, false);

  std::queue<uint64_t> q;
  q.push(src_node);
  dist[src_node] = 0;

  // BFS
  while (!q.empty()) {
    int min_node = q.front();
    q.pop();
    visited[min_node] = true;

    if (min_node == dst_node) {
      break;
    }

    for (int i = 0; i < total_devs; i++) {
      if (visited[i] || conn[min_node * total_devs + i] == 0) {
        continue;
      }
      float new_dist = dist[min_node] + 1;
      if (new_dist < dist[i] || (new_dist == dist[i] && unif(gen) < 0.5)) {
        dist[i] = new_dist;
        prev[i] = min_node;
        q.push(i);
      }
    }
  }
  hop = 0;
  narrowest = std::numeric_limits<int>::max();
  int curr = dst_node;
  while (prev[curr] != -1) {
    if (narrowest > conn[prev[curr] * total_devs + curr]) {
      narrowest = conn[prev[curr] * total_devs + curr];
    }
    hop++;
    curr = prev[curr];
  }
  assert(hop > 0 || src_node == dst_node);
}

std::vector<std::pair<int, int>>
    ShortestPathNetworkRoutingStrategy::hop_count(int src_node) {
  std::vector<uint64_t> dist(total_devs, std::numeric_limits<uint64_t>::max());
  std::vector<int> prev(total_devs, -1);
  std::vector<bool> visited(total_devs, false);

  std::queue<uint64_t> q;
  q.push(src_node);
  dist[src_node] = 0;

  // BFS
  while (!q.empty()) {
    int min_node = q.front();
    q.pop();
    visited[min_node] = true;

    for (int i = 0; i < total_devs; i++) {
      if (visited[i] || conn[min_node * total_devs + i] == 0) {
        continue;
      }
      float new_dist = dist[min_node] + 1;
      if (new_dist < dist[i] || (new_dist == dist[i] && unif(gen) < 0.5)) {
        dist[i] = new_dist;
        prev[i] = min_node;
        q.push(i);
      }
    }
  }

  std::vector<std::pair<int, int>> result;
  for (int i = 0; i < total_devs; i++) {
    if (i == src_node) {
      result.emplace_back(std::make_pair(-1, 0));
      continue;
    }
    int hop = -1;
    int narrowest = 0;
    int curr = i;
    while (prev[curr] != -1) {
      if (!narrowest || (narrowest > conn[prev[curr] * total_devs + curr])) {
        narrowest = conn[prev[curr] * total_devs + curr];
      }
      hop++;
      curr = prev[curr];
    }
    result.emplace_back(std::make_pair(hop, narrowest));
  }
  return result;
}

FlatDegConstraintNetworkTopologyGenerator::
    FlatDegConstraintNetworkTopologyGenerator(int num_nodes, int degree)
    : num_nodes(num_nodes), degree(degree) {}

ConnectionMatrix
    FlatDegConstraintNetworkTopologyGenerator::generate_topology() const {
  ConnectionMatrix conn = std::vector<int>(num_nodes * num_nodes, 0);

  int allocated = 0;
  int curr_node = 0;
  std::unordered_set<int> visited_node;
  visited_node.insert(0);

  std::uniform_int_distribution<> distrib(0, num_nodes - 1);

  while ((long)visited_node.size() != num_nodes) {
    distrib(gen);
    int next_step = distrib(gen);
    if (next_step == curr_node) {
      continue;
    }
    if (visited_node.find(next_step) == visited_node.end()) {
      if (conn[get_id(curr_node, next_step)] == degree) {
        continue;
      }
      conn[get_id(curr_node, next_step)]++;
      conn[get_id(next_step, curr_node)]++;
      visited_node.insert(next_step);
      curr_node = next_step;
      allocated += 2;
    }
  }

  assert(allocated == (num_nodes - 1) * 2);

  std::vector<std::pair<int, int>> node_with_avail_if;
  for (int i = 0; i < num_nodes; i++) {
    int if_inuse = get_if_in_use(i, conn);
    if (if_inuse < degree) {
      node_with_avail_if.emplace_back(i, degree - if_inuse);
    }
  }

  distrib = std::uniform_int_distribution<>(0, node_with_avail_if.size() - 1);
  int a = 0, b = 0;

  while (node_with_avail_if.size() > 1) {
    a = distrib(gen);
    while ((b = distrib(gen)) == a) {
      ;
    }

    assert(
        conn[get_id(node_with_avail_if[a].first, node_with_avail_if[b].first)] <
        degree);
    conn[get_id(node_with_avail_if[a].first, node_with_avail_if[b].first)]++;
    conn[get_id(node_with_avail_if[b].first, node_with_avail_if[a].first)]++;
    allocated += 2;

    bool changed = false;
    if (--node_with_avail_if[a].second == 0) {
      if (a < b) {
        b--;
      }
      node_with_avail_if.erase(node_with_avail_if.begin() + a);
      changed = true;
    }
    if (--node_with_avail_if[b].second == 0) {
      node_with_avail_if.erase(node_with_avail_if.begin() + b);
      changed = true;
    }
    if (changed) {
      distrib =
          std::uniform_int_distribution<>(0, node_with_avail_if.size() - 1);
    }
  }

#ifdef DEBUG_PRINT
  std::cout << "Topology generated: " << std::endl;
  NetworkTopologyGenerator::print_conn_matrix(conn, num_nodes, 0);
#endif
  return conn;
}

int FlatDegConstraintNetworkTopologyGenerator::get_id(int i, int j) const {
  return i * num_nodes + j;
}

int FlatDegConstraintNetworkTopologyGenerator::get_if_in_use(
    int node, ConnectionMatrix const &conn) const {
  int result = 0;
  for (int i = 0; i < num_nodes; i++) {
    result += conn[get_id(node, i)];
  }
  return result;
}

BigSwitchNetworkTopologyGenerator::BigSwitchNetworkTopologyGenerator(
    int num_nodes)
    : num_nodes(num_nodes) {}

ConnectionMatrix BigSwitchNetworkTopologyGenerator::generate_topology() const {
  ConnectionMatrix conn =
      std::vector<int>((num_nodes + 1) * (num_nodes + 1), 0);
  for (int i = 0; i < num_nodes; i++) {
    conn[i * (num_nodes + 1) + num_nodes] = 1;
    conn[num_nodes * (num_nodes + 1) + i] = 1;
  }
  return conn;
}

}; // namespace FlexFlow
