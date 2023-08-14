/* Copyright 2023 CMU, Facebook, LANL, MIT, NVIDIA, and Stanford (alphabetical)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef _FLEXFLOW_GRAPH_H_
#define _FLEXFLOW_GRAPH_H_
#include "flexflow/basic_graph.h"
#include "flexflow/graph_structures.h"
#include "flexflow/memory_optimization.h"
#include "flexflow/model.h"
#include "flexflow/utils/dot/dot_file.h"
#include "flexflow/utils/recursive_logger.h"
#include "legion/legion_utilities.h"
#include <unordered_set>

extern LegionRuntime::Logger::Category log_dp;

namespace FlexFlow::PCG {

struct Edge {
  Edge(void);
  Edge(Node const &_srcOp, Node const &_dstOp, int _srcIdx, int _dstIdx);
  bool operator==(Edge const &rhs) const;
  Node srcOp, dstOp;
  int srcIdx, dstIdx;

  void replace_node(Node const &currentOp, Node const &replaceWith);
};

struct EdgeCompare {
  bool operator()(Edge const &a, Edge const &b) const {
    if (!(a.srcOp == b.srcOp)) {
      return a.srcOp < b.srcOp;
    }
    if (!(a.dstOp == b.dstOp)) {
      return a.dstOp < b.dstOp;
    }
    if (a.srcIdx != b.srcIdx) {
      return a.srcIdx < b.srcIdx;
    }
    if (a.dstIdx != b.dstIdx) {
      return a.dstIdx < b.dstIdx;
    }
    return false;
  };
};
}; // namespace FlexFlow::PCG

namespace std {
template <>
struct hash<FlexFlow::PCG::Edge> {
  size_t operator()(FlexFlow::PCG::Edge const &e) const {
    size_t res = 17;
    res = res * 31 + hash<size_t>()((size_t)e.srcOp.guid);
    res = res * 31 + hash<size_t>()((size_t)e.dstOp.guid);
    res = res * 31 + hash<int>()(e.srcIdx);
    res = res * 31 + hash<int>()(e.dstIdx);
    return res;
  }
};

template <>
struct hash<FlexFlow::PCG::Node> {
  size_t operator()(FlexFlow::PCG::Node const &n) const {
    return n.guid;
  }
};
}; // namespace std

namespace FlexFlow::PCG {

struct NodeCompare {
  bool operator()(Node const &a, Node const &b) const {
    if (a.guid != b.guid) {
      return a.guid < b.guid;
    }
    return a.ptr < b.ptr;
  };
};

struct GraphOptimalViewSerialized {
#ifdef LEGION_MAX_RETURN_SIZE
  static const size_t buffer_size = 4 * LEGION_MAX_RETURN_SIZE - 8;
#else
  static const size_t buffer_size = 1024 * 1024 - 8;
#endif
  size_t total_bytes;
  char data[buffer_size];
};

struct NodeAssignment {
  Node node;
  MachineView view;
};

struct GraphCostResult {
  float cost;
  std::unordered_map<Node, MachineView> views;

  static GraphCostResult invalid();

  bool operator<(GraphCostResult const &other) const;

  friend std::ostream &operator<<(std::ostream &, GraphCostResult const &);
};

/**
 * @brief Holds the cost information of a PCG.
 */
struct GraphCostResultWithMemory {
  float cost;           ///< Run time cost
  MemoryUsage mem_cost; ///< Memory usage
  ///< Corresponding machine views (device placement views)
  std::unordered_map<Node, MachineView> views;

  /**
   * @brief Get the multi-objective cost that combines the run time and memory
   * cost.
   *
   * @return float Numerical value to represent the overall cost
   */
  float get_multi_obj_cost() const;

  static GraphCostResultWithMemory invalid();

  bool operator<(GraphCostResultWithMemory const &other) const;

  friend std::ostream &operator<<(std::ostream &,
                                  GraphCostResultWithMemory const &);
};

template <typename T>
T sequence_cost(T const &first, T const &second);

template <typename T>
T parallel_cost(T const &first, T const &second);

size_t dp_state_hash(Graph const *graph,
                     Node const &sink_node,
                     MachineView const &sink_view,
                     Node const &source_node,
                     MachineView const &source_view,
                     MachineResource const &resource);

enum class SplitType { SEQUENTIAL, VERTICAL, HORIZONTAL };

struct NonsequenceSplit {
  SplitType type;
  int param;
  bool flip_graphs;

  static NonsequenceSplit sequential();
  static NonsequenceSplit vertical(int param, bool flip_graphs);
  static NonsequenceSplit horizontal(int param, bool flip_graphs);
};

using SequenceSplit = NodeAssignment;

class SearchHelper {
public:
  SearchHelper(FFModel *model);

  template <typename T>
  T graph_cost(Graph const *graph,
               NodeAssignment const &source,
               NodeAssignment const &sink,
               MachineResource const &resources,
               bool include_sink_compute_time) const;
  template <typename T>
  T find_optimal_sequence_graph_time(Graph const *g,
                                     Node const &bottleneck_node,
                                     NodeAssignment const &source,
                                     NodeAssignment const &sink,
                                     MachineResource const &resources) const;
  /**
   * @brief Starting point to get parallel split time cost.
   *
   * @tparam T float or GraphCostResult (or GraphCostResultWithMemory in memory
   * optimization)
   */
  template <typename T>
  T find_optimal_nonsequence_graph_time(Graph const *g,
                                        NodeAssignment const &source,
                                        NodeAssignment const &sink,
                                        MachineResource const &resources) const;
  /* void find_optimal_nonsequence_graph_views(Graph const *g, */
  /*                                           NodeAssignment const &source, */
  /*                                           NodeAssignment const &sink, */
  /*                                           MachineResource const &resources,
   */
  /*                                           float optimal_cost, */
  /*                                           std::unordered_map<Node,
   * MachineView>& optimal_views) const; */
  std::vector<MachineView>
      get_valid_machine_views(Node const &node,
                              MachineResource const &resource,
                              bool log = false) const;
  std::vector<MachineView> get_valid_machine_views(
      Op const *op, MachineResource const &resource, bool log = false) const;

  template <typename T>
  std::pair<bool, T> try_get_cost_from_cache(size_t hash) const;

  template <typename T>
  void try_cache_result(size_t hash, T const &value) const;

  template <typename T>
  T infinity() const;

  template <typename T>
  T empty() const;

  template <typename T>
  bool is_invalid(T const &) const;

  template <typename T>
  T estimate_xfer_cost(Graph const *g,
                       NodeAssignment const &source,
                       NodeAssignment const &sink) const;

  template <typename T>
  void add_operator_cost(NodeAssignment const &, float, T *) const;

  template <typename T>
  void add_sink_node_costs(NodeAssignment const &sink,
                           CostMetrics metrics,
                           T *result) const;

  /**
   * @brief Add run time cost and memory cost of the operator to the graph cost.
   * This is a temp workaround and should be refactored eventually.
   */
  void add_operator_cost_with_memory(NodeAssignment const &node,
                                     float node_run_time_cost,
                                     MemoryUsage node_mem_cost,
                                     GraphCostResultWithMemory *cost) const;

  template <typename T>
  float get_cost(T const &) const;

  template <typename T>
  void check_matches_graph(Graph const *, T const &, Node const &) const;

public:
  mutable std::unique_ptr<RecursiveLogger> logger;

  void clear_cache();

private:
  template <typename T>
  T execute_nonsequence_split(std::unique_ptr<Graph> const &first_graph,
                              std::unique_ptr<Graph> const &second_graph,
                              NodeAssignment const &source,
                              NodeAssignment const &sink,
                              MachineResource const &resources,
                              NonsequenceSplit const &split) const;

  template <typename T>
  T execute_sequence_split(std::unique_ptr<Graph> const &first_graph,
                           std::unique_ptr<Graph> const &second_graph,
                           NodeAssignment const &source,
                           NodeAssignment const &sink,
                           MachineResource const &resources,
                           SequenceSplit const &split) const;

private:
  FFModel *model;

  mutable std::unordered_map<size_t, float> cached_graph_costs;
  mutable std::unordered_map<size_t,
                             std::unique_ptr<const std::vector<MachineView>>>
      cached_operator_valid_views;
};

struct SimplificationSettings {
  bool simplify_parallel_ops = false;
  bool fuse_parallel_ops = false;
  bool remove_trailing_parallel_ops = false;
  bool remove_noops = false;
};

class Graph {
public:
  Graph(FFModel *model);
  void add_edge(Node const &srcOp, Node const &dstOp, int srcIdx, int dstIdx);
  void add_node(Node const &);
  void add_edge(Edge const &e);
  void remove_node(Node const &, bool purge_edges = false);
  void remove_edge(Edge const &e, bool remove_node_if_unused = true);
  bool has_edge(Node const &srcOp,
                Node const &dstOp,
                int srcIdx,
                int dstIdx) const;
  bool has_edge(Edge const &e) const;
  void replace_subgraph(std::unordered_set<Node> const &currentNodes,
                        Graph const &replaceWith);
  Graph subgraph(std::unordered_set<Node> const &nodes) const;
  void contract_out_node(Node const &);
  float optimal_cost() const;
  float optimal_cost_with_memory(float run_time_cost_factor) const;
  std::unordered_map<Node, MachineView> optimal_views() const;
  void remove_input_nodes();
  void duplicate_input_node(Node const &);
  void duplicate_input_nodes();
  Node clone_node(Node const &);
  std::pair<Node, std::unordered_set<Node>>
      deduplicate_input_node(Node const &);
  std::unordered_map<Node, Node> deduplicate_input_nodes();
  Node declone_node(Node const &);

  size_t hash(void) const;
  void print(void) const;
  void print_dot() const;
  void print_dot(std::ostream &) const;

  bool check_correctness(void);
  bool has_loop(void);
  bool map_operators_to_layers(std::vector<Op *> &layers) const;
  static GraphOptimalViewSerialized
      graph_optimize_task(Legion::Task const *task,
                          std::vector<Legion::PhysicalRegion> const &regions,
                          Legion::Context ctx,
                          Legion::Runtime *runtime);
  static GraphOptimalViewSerialized
      graph_optimize_wrapper(FFModel * model);
  Node find_bottleneck_node(Node const &sink_node,
                            Node const &source_node) const;
  void print_strategy_computation_graph(
      std::unordered_map<Node, MachineView> const &strategy) const;
  void export_strategy_computation_graph(
      std::unordered_map<Node, MachineView> const &strategy,
      std::string const &out_filename) const;
  void export_strategy_computation_graph(
      std::unordered_map<Node, MachineView> const &strategy,
      DotFile<Node> &dot) const;

  std::pair<std::unique_ptr<Graph>, std::unique_ptr<Graph>>
      split_at_node(Node const &bottleneck) const;
  std::pair<std::unique_ptr<Graph>, std::unique_ptr<Graph>>
      split_horizontal(Node const &source_node, Node const &sink_node) const;

  Graph reduced() const;

  Node find_sink_node() const;
  Node find_source_node() const;
  void reshape_output_tensor(ParallelTensorShape const &shape);
  std::unique_ptr<Graph>
      with_output_tensor_reshaped_to(ParallelTensorShape const &shape) const;

  void simplify(SimplificationSettings const &);
  void simplify_parallel_ops();

  static Graph singleton(FFModel *, Node const &);
  bool empty() const;

  template <typename T>
  T generic_optimal_cost() const;

public:
  FFModel *model;
  SearchHelper *search;
  std::unordered_map<Node, std::unordered_set<Edge>> inEdges, outEdges;

private:
  void remove_inverse_parallel_ops();
  void replace_subgraph_with_nonempty(
      std::unordered_set<Node> const &currentNodes, Graph const &replaceWith);
};

struct GraphOptimizeResult {
  tl::optional<Graph> graph;
  float cost;
  std::unordered_map<Node, MachineView> views;

  friend std::ostream &operator<<(std::ostream &, GraphOptimizeResult const &);
};

/**
 * @brief Hold the optimization results with memory information.
 */
struct GraphOptimizeResultWithMemory {
  tl::optional<Graph> graph; ///< Optimized PCG
  float cost;                ///< Run time cost
  MemoryUsage mem_cost;      ///< Memory usage
  ///< Corresponding machine views (device placement views)
  std::unordered_map<Node, MachineView> views;

  friend std::ostream &operator<<(std::ostream &,
                                  GraphOptimizeResultWithMemory const &);
};

namespace Utils {
template <>
struct GraphStructure<FlexFlow::PCG::Graph> {
  using G = FlexFlow::PCG::Graph;
  using graph_type = FlexFlow::PCG::Graph;
  using vertex_type = FlexFlow::PCG::Node;
  using edge_type = FlexFlow::PCG::Edge;

  std::unordered_set<vertex_type> get_nodes(G const &g) const {
    std::unordered_set<vertex_type> nodes;
    for (auto const &kv : g.inEdges) {
      nodes.insert(kv.first);
    }
    for (auto const &kv : g.outEdges) {
      nodes.insert(kv.first);
    }

    return nodes;
  }

  std::unordered_set<edge_type> get_incoming_edges(G const &g,
                                                   vertex_type const &n) const {
    if (g.inEdges.find(n) == g.inEdges.end()) {
      return {};
    } else {
      return {g.inEdges.at(n).begin(), g.inEdges.at(n).end()};
    }
  }

  std::unordered_set<edge_type> get_outgoing_edges(G const &g,
                                                   vertex_type const &n) const {
    if (g.outEdges.find(n) == g.outEdges.end()) {
      return {};
    } else {
      return {g.outEdges.at(n).begin(), g.outEdges.at(n).end()};
    }
  }

  vertex_type get_src(G const &g, edge_type const &e) const {
    return e.srcOp;
  }

  vertex_type get_dst(G const &g, edge_type const &e) const {
    return e.dstOp;
  }

  void set_src(G const &g, edge_type &e, vertex_type const &n) const {
    e.srcOp = n;
  }

  void set_dst(G const &g, edge_type &e, vertex_type const &n) const {
    e.dstOp = n;
  }
};

template <>
struct invalid_node<Graph, GraphStructure<Graph>> {
  using G = Graph;
  using Structure = GraphStructure<Graph>;
  using vertex_type = typename Structure::vertex_type;

  vertex_type operator()() const {
    return vertex_type::INVALID_NODE;
  }
};

template <>
struct invalid_node<BasicGraph<Node>, GraphStructure<BasicGraph<Node>>> {
  Node operator()() const {
    return Node::INVALID_NODE;
  }
};
}; // namespace Utils
}; // namespace FlexFlow::PCG
#endif
