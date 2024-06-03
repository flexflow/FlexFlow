
#include "compiler/machine_mapping.h"
#include "pcg/parallel_computation_graph.h"
#include "utils/exception.h"
#include "utils/graph/serialparallel.h"
#include "utils/deduplicated_priority_queue.h"
#include <algorithm>

namespace FlexFlow {

// Computes estimated execution cost for a single node
float node_estimate_cost(Node node,
                         SubParallelComputationGraphView const &g,
                         CostEstimator const &estimator,
                         MachineMapping const &device_mapping) {
  std::unordered_set<UpwardOpenMultiDiEdge> incoming_edges =
      get_incoming_edges(g, node);
  std::vector<ParallelTensorShape> inputs =
      transform(as_vector(get_incoming_edges(g, node)),
                [&](UpwardOpenMultiDiEdge const &input_edge) {
                  return g.at(input_edge).get_shape();
                });
  float cost = estimator.estimate_cost(
      g.at(node).attrs, inputs, device_mapping.machine_views.at(node));
  return cost;
}

struct TimedNode { // Node and associated finishing time
  Node node;
  req<float> endtime;
};
FF_VISITABLE_STRUCT(TimedNode, node, endtime);

struct TimeComparison {
  bool operator()(TimedNode const &lhs, TimedNode const &rhs) const {
    return (lhs.endtime < rhs.endtime);
  }
};

float parallel_estimate_cost(
    SubParallelComputationGraphView const &g,
    CostEstimator const &estimator,
    MachineMapping const &device_mapping,
    std::unordered_map<InputMultiDiEdge, MachineView> const
        &frontier_machine_views) {
  float current_time = 0;
  std::unordered_set<Node> frontier; // nodes whose dependencies (previous nodes) have been met, and
                // are waiting to be processed.
  DeduplicatedPriorityQueue<TimedNode, std::vector<TimedNode>, TimeComparison>
      processing; // nodes currently being processed.
  std::unordered_set<Node>
      processed; // set of nodes that have already been processed
  std::unordered_map<device_id_t, bool>
      occupied; // keeps track of the devices that are currently occupied
  // Filling the frontier
  for (auto const &[edge, _] : frontier_machine_views) {
      auto node = get_dst_node(edge);
      frontier.insert(node);
    }

  while (!frontier.empty() || !processing.empty()) {
    // Processing new nodes
    std::unordered_set<Node> copy(frontier);
    for (Node const &node : copy) {
      std::vector<device_id_t> devices =
          device_mapping.machine_views.at(node).device_ids();
      if (std::all_of(devices.begin(), devices.end(), [&occupied](device_id_t d) {
            return occupied[d] == false;
          })) {
        float cost = node_estimate_cost(node, g, estimator, device_mapping);
        processing.push({node, current_time + cost});
        for (device_id_t d : devices) {
          occupied[d] = true;
        }
        frontier.erase(node);
      }
    }
    // Finish processing one node
    TimedNode finished = processing.top();
    processing.pop();
    std::vector<device_id_t> devices =
        device_mapping.machine_views.at(finished.node).device_ids();
    for (device_id_t d : devices) { // free devices
      occupied[d] = false;
    }
    processed.insert(finished.node);
    current_time = finished.endtime;

    // Adding candidates to the frontier
    for (Node const &successor :
         get_successors(g, finished.node)) { // All nodes depending on finished
      std::unordered_set<Node> predecessors = get_predecessors(g, successor);
      if (std::all_of(
              predecessors.begin(), predecessors.end(), [&processed](Node p) {
                return processed.find(p) != processed.end();
              })) {
        frontier.insert(successor);
      }
    }
  }
  return current_time;
}
} // namespace FlexFlow
