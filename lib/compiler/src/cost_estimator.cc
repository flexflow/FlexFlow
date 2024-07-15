#include "compiler/cost_estimator.h"
#include "compiler/cost_estimate.h"
#include "compiler/machine_mapping.h"
#include "pcg/parallel_computation_graph.h"
#include "utils/deduplicated_priority_queue.h"
#include "utils/exception.h"
#include "utils/graph/serialparallel.h"
#include <algorithm>

namespace FlexFlow {

// Computes estimated execution cost for a single node
float node_estimate_cost(Node const &node,
                         SubParallelComputationGraphView const &g,
                         CostEstimator const &estimator,
                         MachineMapping const &device_mapping) {
  std::unordered_set<UpwardOpenMultiDiEdge> incoming_edges =
      get_incoming_edges(g, node);

  std::vector<ParallelTensorShape> inputs = transform(
      as_vector(incoming_edges), [&](UpwardOpenMultiDiEdge const &input_edge) {
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

bool predecessors_have_been_processed(
    std::unordered_set<Node> const &predecessors,
    std::unordered_set<TimedNode> processed) {
  std::unordered_set<Node> simple_processed =
      transform(processed, [](TimedNode const &tn) { return tn.node; });

  return all_of(predecessors, [&simple_processed](Node p) {
    return simple_processed.find(p) != simple_processed.end();
  });
}

std::vector<device_id_t> get_devices(Node const &node,
                                     MachineMapping const &device_mapping) {
  return device_mapping.machine_views.at(node).device_ids();
}

float parallel_estimate_cost(
    SubParallelComputationGraphView const &g,
    CostEstimator const &estimator,
    MachineMapping const &device_mapping,
    std::unordered_map<InputMultiDiEdge, MachineView> const
        &frontier_machine_views) {
  float current_time = 0;
  std::unordered_set<Node>
      frontier; // nodes whose dependencies (previous nodes) have been met, and
                // are waiting to be processed.
  DeduplicatedPriorityQueue<TimedNode, std::vector<TimedNode>, TimeComparison>
      processing; // nodes currently being processed.
  std::unordered_set<TimedNode>
      processed; // set of nodes that have already been processed
  std::unordered_map<device_id_t, bool>
      occupied; // keeps track of the devices that are currently occupied

  // Filling the frontier
  for (auto const &[edge, _] : frontier_machine_views) {
    Node node = get_dst_node(edge);
    frontier.insert(node);
  }

  auto start_node_processing = [&](Node const &node,
                                   std::vector<device_id_t> const &devices) {
    float cost = node_estimate_cost(node, g, estimator, device_mapping);
    processing.push({node, current_time + cost});
    for (device_id_t d : devices) {
      occupied[d] = true;
    }
    frontier.erase(node);
  };

  auto finish_node_processing = [&](TimedNode const &finished) {
    std::vector<device_id_t> devices =
        get_devices(finished.node, device_mapping);
    for (device_id_t d : devices) { // free devices
      occupied[d] = false;
    }
    processed.insert(finished);
    current_time = finished.endtime;
  };

  while (!frontier.empty() || !processing.empty()) {
    // Processing new nodes
    std::unordered_set<Node> frontier_copy(frontier);
    for (Node const &node : frontier_copy) {
      std::vector<device_id_t> devices = get_devices(node, device_mapping);
      if (all_of(devices,
                 [&occupied](device_id_t d) { return occupied[d] == false; })) {
        start_node_processing(node, devices);
      }
    }

    // Finish processing all nodes
    while (!processing.empty()) {
      TimedNode finished = processing.top();
      processing.pop();
      finish_node_processing(finished);

      // Adding candidates to the frontier
      for (Node const &successor : get_successors(g, finished.node)) {
        std::unordered_set<Node> predecessors = get_predecessors(g, successor);

        if (predecessors_have_been_processed(predecessors, processed)) {

          frontier.insert(successor);
        }
      }
    }
  }
  return current_time;
}
} // namespace FlexFlow
