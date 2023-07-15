#include "search_helper.h"

namespace FlexFlow {
namespace PCG {

SearchHelper::SearchHelper() {
  this->logger = std::unique_ptr<RecursiveLogger>(new RecursiveLogger("DP"));
}

template <typename T>
T SearchHelper::execute_sequence_split(std::unique_ptr<Graph> const &pre_graph,
                                       std::unique_ptr<Graph> const &post_graph,
                                       NodeAssignment const &source,
                                       NodeAssignment const &sink,
                                       MachineResource const &resources,
                                       SequenceSplit const &bn) const {
  return sequence_cost<T>(
      this->graph_cost<T>(pre_graph.get(), source, bn, resources, true),
      this->graph_cost<T>(post_graph.get(), bn, sink, resources, false));
}

template <typename T>
T SearchHelper::find_optimal_sequence_graph_time(
    Graph const *g,
    Node const &bn_node,
    NodeAssignment const &source,
    NodeAssignment const &sink,
    MachineResource const &resources) const {
  std::unique_ptr<Graph> pre_graph;
  std::unique_ptr<Graph> post_graph;
  std::tie(pre_graph, post_graph) = g->split_at_node(bn_node);

  T optimal = this->infinity<T>();

  std::vector<MachineView> valid_views =
      this->get_valid_machine_views(bn_node.op_params, resources);
  // A Corner Case:
  // If bn_node is a parallel_op and an input to sink_node,
  // Add sink_node's view to the list, since sink_node's view
  // may not be a valid view for resources, but UniFlow support
  // this case since parallel_op does not trigger computation
  if (is_parallel_op(bn_node.op_params)) {
    bool found = false;
    auto const &inList = g->inEdges.find(sink.node)->second;
    for (auto const &e : inList) {
      if (e.srcOp == bn_node) {
        found = true;
        break;
      }
    }
    if (found) {
      for (int j = 0; j < bn_node.ptr->numOutputs; j++) {
        if (!bn_node.ptr->outputs[j]->is_valid_machine_view(sink.view)) {
          found = false;
        }
      }
    }
    if (found) {
      valid_views.push_back(sink.view);
    }
  }

  if (valid_views.empty()) {
    return optimal;
  }

  float optimal_cost = std::numeric_limits<float>::infinity();
  MachineView best_view;

  for (MachineView const &bn_view : valid_views) {
    float cost = this->execute_sequence_split<float>(
        pre_graph, post_graph, source, sink, resources, {bn_node, bn_view});

    if (cost < optimal_cost) {
      best_view = bn_view;
      optimal_cost = cost;
    }
  }

  if (optimal_cost != std::numeric_limits<float>::infinity()) {
    optimal = this->execute_sequence_split<T>(
        pre_graph, post_graph, source, sink, resources, {bn_node, best_view});
  }

  check_matches_graph<T>(g, optimal, sink.node);

  return optimal;
}

template <typename T>
T SearchHelper::execute_nonsequence_split(
    std::unique_ptr<Graph> const &first_graph,
    std::unique_ptr<Graph> const &second_graph,
    NodeAssignment const &source,
    NodeAssignment const &sink,
    MachineResource const &resources,
    NonsequenceSplit const &split) const {
  Graph const *first = first_graph.get();
  Graph const *second = second_graph.get();
  if (split.flip_graphs) {
    std::swap(first, second);
  }
  switch (split.type) {
    case SplitType::SEQUENTIAL:
      this->logger->debug() << "Exploring sequential nonsequence split";
      return sequence_cost<T>(
          this->graph_cost<T>(first, source, sink, resources, false),
          this->graph_cost<T>(second, source, sink, resources, false));
    case SplitType::VERTICAL: {
      this->logger->debug() << "Exploring vertical nonsequence split ("
                            << split.param << ", " << split.flip_graphs << ")";
      MachineResource firstRes = resources, secondRes = resources;
      firstRes.num_nodes = split.param;
      secondRes.num_nodes = resources.num_nodes - split.param;
      secondRes.start_gpu_id =
          resources.start_gpu_id + resources.all_gpus_per_node * split.param;

      return parallel_cost<T>(
          this->graph_cost<T>(first, source, sink, firstRes, false),
          this->graph_cost<T>(second, source, sink, secondRes, false));
    }
    case SplitType::HORIZONTAL: {
      this->logger->debug() << "Exploring horizontal nonsequence split ("
                            << split.param << ", " << split.flip_graphs << ")";
      MachineResource firstRes = resources, secondRes = resources;
      firstRes.available_gpus_per_node = split.param;
      secondRes.available_gpus_per_node =
          resources.available_gpus_per_node - split.param;
      secondRes.start_gpu_id = resources.start_gpu_id + split.param;

      return parallel_cost<T>(
          this->graph_cost<T>(first, source, sink, firstRes, false),
          this->graph_cost<T>(second, source, sink, secondRes, false));
    }
    default:
      assert(false);
  }
}

template <typename T>
T SearchHelper::find_optimal_nonsequence_graph_time(
    Graph const *g,
    NodeAssignment const &source,
    NodeAssignment const &sink,
    MachineResource const &resources) const {
  std::unique_ptr<Graph> first_graph;
  std::unique_ptr<Graph> second_graph;
  std::tie(first_graph, second_graph) =
      g->split_horizontal(source.node, sink.node);

  std::vector<NonsequenceSplit> potential_splits;

  for (int i = 1; i < resources.num_nodes; i++) {
    potential_splits.push_back(NonsequenceSplit::vertical(i, false));
    potential_splits.push_back(NonsequenceSplit::vertical(i, true));
  }
  for (int i = 1; i < resources.available_gpus_per_node; i++) {
    potential_splits.push_back(NonsequenceSplit::horizontal(i, false));
    potential_splits.push_back(NonsequenceSplit::horizontal(i, true));
  }

  NonsequenceSplit best_split = NonsequenceSplit::sequential();
  float best_cost = this->execute_nonsequence_split<float>(
      first_graph, second_graph, source, sink, resources, best_split);
  for (NonsequenceSplit const &split : potential_splits) {
    float cost = this->execute_nonsequence_split<float>(
        first_graph, second_graph, source, sink, resources, split);
    this->logger->debug() << "Found cost: " << cost;

    if (cost < best_cost) {
      best_cost = cost;
      best_split = split;
    }
  }

  switch (best_split.type) {
    case SplitType::SEQUENTIAL:
      this->logger->debug() << "Best split: SEQUENTIAL";
      break;
    case SplitType::VERTICAL:
      this->logger->debug() << "Best split: VERTICAL(" << best_split.param
                            << ", " << best_split.flip_graphs << ")";
      break;
    case SplitType::HORIZONTAL:
      this->logger->debug() << "Best split: HORIZONTAL(" << best_split.param
                            << ", " << best_split.flip_graphs << ")";
      break;
  }
  T optimal = this->execute_nonsequence_split<T>(
      first_graph, second_graph, source, sink, resources, best_split);

  check_matches_graph<T>(g, optimal, sink.node);

  return optimal;
}

std::vector<MachineView> SearchHelper::get_valid_machine_views(
    Node const &node, MachineResource const &resource, bool log) const {
  this->logger->info() << "Getting valid machine views for "
                       << node.to_string();
  return this->get_valid_machine_views(node.ptr, resource, log);
}

std::vector<MachineView> SearchHelper::get_valid_machine_views(
    Op const *op, MachineResource const &resource, bool log) const {
  std::vector<MachineView> const *cached_op_views = NULL;
  std::vector<MachineView> valid_views;

  auto const &iter = cached_operator_valid_views.find(op->op_guid);
  if (iter != cached_operator_valid_views.end()) {
    cached_op_views = iter->second.get();
  } else {
    auto to_cache = std::unique_ptr<std::vector<MachineView>>(
        new std::vector<MachineView>());
    if (log) {
      this->logger->info() << "Considering a total of "
                           << this->model->all_valid_views.size()
                           << " potential valid views";
    }
    for (size_t i = 0; i < this->model->all_valid_views.size(); i++) {
      bool valid = true;
      for (int j = 0; j < op->numOutputs; j++) {
        if (!op->outputs[j]->is_valid_machine_view(
                this->model->all_valid_views[i])) {
          valid = false;
          {
            MachineView const &view = this->model->all_valid_views[i];
            std::ostringstream oss;
            oss << "[" << view.ndims << "](";
            for (int i = 0; i < view.ndims; i++) {
              oss << view.dim[i] << "/" << view.stride[i];
              if (i != view.ndims - 1) {
                oss << " ";
              }
            }
            oss << ")";
            if (log) {
              this->logger->info() << "Rejecting machine view: " << oss.str();
            }
          }
          break;
        }
      }
      if (valid) {
        {
          MachineView const &view = this->model->all_valid_views[i];
          std::ostringstream oss;
          oss << "[" << view.ndims << "](";
          for (int i = 0; i < view.ndims; i++) {
            oss << view.dim[i] << "/" << view.stride[i];
            if (i != view.ndims - 1) {
              oss << " ";
            }
          }
          oss << ")";
          if (log) {
            this->logger->info() << "Accepting machine view: " << oss.str();
          }
        }
        to_cache->push_back(this->model->all_valid_views[i]);
      }
    }
    cached_operator_valid_views[op->op_guid] = std::move(to_cache);
    cached_op_views = cached_operator_valid_views.at(op->op_guid).get();
  }
  if (log) {
    this->logger->info() << "Found " << cached_op_views->size()
                         << " cached op views";
  }
  for (size_t i = 0; i < cached_op_views->size(); i++) {
    MachineView view = (*cached_op_views)[i];
    if (view.device_type == MachineView::GPU) {
      view.start_device_id = resource.start_gpu_id;
    } else if (view.device_type == MachineView::CPU) {
      view.start_device_id = resource.start_cpu_id;
    } else {
      assert(false);
    }
    if (resource.is_valid_machine_view(view)) {
      valid_views.push_back(view);
    }
  }
  return valid_views;
}

template <>
bool SearchHelper::is_invalid<float>(float const &cost) const {
  return cost == std::numeric_limits<float>::infinity();
}

template <>
bool SearchHelper::is_invalid<GraphCostResult>(
    GraphCostResult const &cost) const {
  return cost.cost == std::numeric_limits<float>::infinity();
}

/**
 * @brief Asserts that the results of graph optimization are valid for the graph
 *
 * @param g the graph to check against
 * @param r the results to check
 * @param sink the sink node of the graph g
 * @param include_sink whether or not to include the sink node
 */
template <>
void SearchHelper::check_matches_graph<GraphCostResult>(
    Graph const *g, GraphCostResult const &r, Node const &sink) const {
  using FlexFlow::PCG::Utils::nodes;

  if (this->is_invalid(r)) {
    return;
  }

  std::unordered_set<Node> g_nodes = nodes(*g);
  g_nodes.erase(sink);

  std::unordered_set<Node> r_nodes;
  for (auto const &kv : r.views) {
    r_nodes.insert(kv.first);
  }

  assert(g_nodes == r_nodes);
}

template <>
void SearchHelper::check_matches_graph<float>(Graph const *g,
                                              float const &r,
                                              Node const &sink) const {}

template <>
std::pair<bool, float>
    SearchHelper::try_get_cost_from_cache<float>(size_t hash) const {
  if (this->cached_graph_costs.find(hash) == this->cached_graph_costs.end()) {
    return {false, std::numeric_limits<float>::infinity()};
  } else {
    return {true, this->cached_graph_costs.at(hash)};
  }
}

template <>
std::pair<bool, GraphCostResult>
    SearchHelper::try_get_cost_from_cache<GraphCostResult>(size_t hash) const {
  return {false, GraphCostResult::invalid()};
}

template <>
void SearchHelper::try_cache_result<float>(size_t hash,
                                           float const &value) const {
  this->logger->debug() << "cached_graph_costs[" << hash << "] = " << value;
  this->cached_graph_costs[hash] = value;
}

template <>
void SearchHelper::try_cache_result<GraphCostResult>(
    size_t hash, GraphCostResult const &value) const {
  this->logger->debug() << "cached_graph_costs[" << hash << "=" << value.cost
                        << "]";
  this->cached_graph_costs[hash] = value.cost;
}

template <>
float SearchHelper::infinity<float>() const {
  return std::numeric_limits<float>::infinity();
}

template <>
GraphCostResult SearchHelper::infinity<GraphCostResult>() const {
  return {std::numeric_limits<float>::infinity(), {}};
}

template <>
float SearchHelper::empty<float>() const {
  return 0.0f;
}

template <>
GraphCostResult SearchHelper::empty<GraphCostResult>() const {
  return {0.0f, {}};
}

template <typename T>
T SearchHelper::estimate_xfer_cost(Graph const *graph,
                                   NodeAssignment const &source,
                                   NodeAssignment const &sink) const {
  T result = this->empty<T>();

  if (source.node != Node::INVALID_NODE) {
    auto const &inList = graph->inEdges.find(sink.node)->second;
    float op_cost = 0.0f;
    for (auto const &it2 : inList) {
      assert(it2.srcOp == source.node);
      assert(sink.node.ptr->inputs[it2.dstIdx]->is_valid_machine_view(
          source.view));

      float estimated_xfer_cost = this->model->simulator->estimate_xfer_cost(
          sink.node.ptr, it2.dstIdx, source.view, sink.view);
      // printf("Estimated xfer cost from %s to %s: %fms\n",
      // source.node.ptr->name, sink.node.ptr->name, estimated_xfer_cost);
      op_cost += estimated_xfer_cost;
    }
    this->add_operator_cost<T>(source, op_cost, &result);
  } else {
    Node real_source = graph->find_source_node();
    assert(real_source.ptr->op_type == OP_INPUT);
    this->add_operator_cost({real_source, MachineView::NO_VIEW}, 0.0f, &result);
  }

  return result;
}

template <>
void SearchHelper::add_operator_cost<float>(NodeAssignment const &node,
                                            float node_cost,
                                            float *cost) const {
  *cost += node_cost;
}

template <>
void SearchHelper::add_operator_cost<GraphCostResult>(
    NodeAssignment const &node, float node_cost, GraphCostResult *cost) const {
  cost->cost += node_cost;
  cost->views[node.node] = node.view;
}

template <>
float SearchHelper::get_cost<float>(float const &f) const {
  return f;
}

template <>
float SearchHelper::get_cost<GraphCostResult>(
    GraphCostResult const &gcr) const {
  return gcr.cost;
}

template <typename T>
T SearchHelper::graph_cost(Graph const *graph,
                           NodeAssignment const &source,
                           NodeAssignment const &sink,
                           MachineResource const &resources,
                           bool include_sink_compute_time) const {
  TAG_ENTER(this->logger);
  this->logger->debug() << "sink(" << sink.node.guid << ") "
                        << "sink.view(" << sink.view.ndims << " "
                        << sink.view.start_device_id << " " << sink.view.dim[0]
                        << ") "
                        << "source(" << source.node.guid << ") "
                        << "source.view(" << source.view.ndims << " "
                        << source.view.start_device_id << " "
                        << source.view.dim[0] << ") "
                        << "resources(" << resources.num_nodes << " "
                        << resources.start_gpu_id << " "
                        << resources.available_gpus_per_node << ")";
  if (this->model->config.profiling) {
    graph->print_dot();
  }

  assert(graph->inEdges.find(sink.node) != graph->inEdges.end());
  if (source.node != Node::INVALID_NODE) {
    assert(graph->outEdges.find(source.node) != graph->outEdges.end());
  }

  size_t hash = dp_state_hash(
      graph, sink.node, sink.view, source.node, source.view, resources);
  this->logger->spew() << "hash = " << hash;

  T result;

  std::pair<bool, T> from_cache = this->try_get_cost_from_cache<T>(hash);
  if (from_cache.first) {
    // cached_graph_costs does not include sink_compute_time
    result = from_cache.second;
  } else {
    if (graph->inEdges.size() <= 2) {
      result = this->estimate_xfer_cost<T>(graph, source, sink);
      this->logger->debug()
          << "Estimated xfer cost is " << this->get_cost(result);
    } else {
      Node bn_node = graph->find_bottleneck_node(sink.node, source.node);
      if (bn_node != Node::INVALID_NODE) {
        // We found a bottleneck node
        this->logger->debug() << "Found bn_node = " << bn_node.guid;

        result = this->find_optimal_sequence_graph_time<T>(
            graph,
            bn_node,
            {source.node, source.view},
            {sink.node, sink.view},
            resources);
      } else {
        // sink node must have multiple branches
        // otherwise we should not be here
        assert(graph->inEdges.find(sink.node)->second.size() > 1);

        result = this->find_optimal_nonsequence_graph_time<T>(
            graph,
            {source.node, source.view},
            {sink.node, sink.view},
            resources);
      }
    }

    this->try_cache_result<T>(hash, result);
  }

  check_matches_graph<T>(graph, result, sink.node);

  if (include_sink_compute_time) {
    CostMetrics metrics =
        this->model->simulator->measure_operator_cost(sink.node.ptr, sink.view);
    this->logger->debug() << "Sink node cost: "
                          << "forward(" << metrics.forward_time << ") "
                          << "backward(" << metrics.backward_time << ") "
                          << "sync(" << metrics.sync_time << ")";
    this->add_operator_cost<T>(sink,
                               metrics.forward_time + metrics.backward_time +
                                   metrics.sync_time,
                               &result);
  }

  return result;
}

} // namespace PCG
} // namespace FlexFlow
