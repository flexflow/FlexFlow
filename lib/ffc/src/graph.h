/* Copyright 2021 CMU, Facebook, LANL, MIT, and Stanford (alphabetical)
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
#include "basic_graph.h"
/* #include "node.h" */
#include "edge.h"
#include "graph_structures.h"
#include "utils/dot_file.h"
#include "pcg/machine_view.h"
#include "utils/recursive_logger.h"
#include <unordered_set>
#include <memory>
#include "op-meta/op-meta.h"
#include "utils/graph.h"
#include "utils/bidict.h"

//extern LegionRuntime::Logger::Category log_dp;

namespace FlexFlow {
namespace ffc {

class SearchHelper;

struct GraphOptimalViewSerialized {
#ifdef LEGION_MAX_RETURN_SIZE
  static const size_t buffer_size = LEGION_MAX_RETURN_SIZE - 8;
#else
  static const size_t buffer_size = 1024 * 1024 - 8;
#endif
  size_t total_bytes;
  char data[buffer_size];
};

class Graph {
public:
  Graph();
  void add_edge(opmeta::OperatorParameters const &srcOp, opmeta::OperatorParameters const &dstOp, int srcIdx, int dstIdx);
  void add_node(opmeta::OperatorParameters const &);
  void add_edge(Edge const &e);
  void remove_node(opmeta::OperatorParameters const &, bool purge_edges = false);
  void remove_edge(Edge const &e, bool remove_node_if_unused = true);
  bool has_edge(opmeta::OperatorParameters const &srcOp,
                opmeta::OperatorParameters const &dstOp,
                int srcIdx,
                int dstIdx) const;
  bool has_edge(Edge const &e) const;
  void replace_subgraph(std::unordered_set<opmeta::OperatorParameters> const &currentNodes,
                        Graph const &replaceWith);
  Graph subgraph(std::unordered_set<opmeta::OperatorParameters> const &nodes) const;
  void contract_out_node(opmeta::OperatorParameters const &);
  float optimal_cost() const;
  std::unordered_map<opmeta::OperatorParameters, MachineView> optimal_views() const;
  void remove_input_nodes();
  void duplicate_input_node(opmeta::OperatorParameters const &);
  void duplicate_input_nodes();
  opmeta::OperatorParameters clone_node(opmeta::OperatorParameters const &);
  std::pair<opmeta::OperatorParameters, std::unordered_set<opmeta::OperatorParameters>>
      deduplicate_input_node(opmeta::OperatorParameters const &);
  std::unordered_map<opmeta::OperatorParameters, opmeta::OperatorParameters> deduplicate_input_nodes();
  opmeta::OperatorParameters declone_node(opmeta::OperatorParameters const &);

  size_t hash(void) const;
  void print(void) const;
  void print_dot() const;
  void print_dot(std::ostream &) const;

  bool check_correctness(void);
  bool has_loop(void);
  //bool map_operators_to_layers(std::vector<OpMeta *> &layers) const;
  //static GraphOptimalViewSerialized
  //    graph_optimize_task(Legion::Task const *task,
  //                        std::vector<Legion::PhysicalRegion> const &regions,
  //                        Legion::Context ctx,
  //                        Legion::Runtime *runtime);
  opmeta::OperatorParameters find_bottleneck_node(opmeta::OperatorParameters const &sink_node,
                            opmeta::OperatorParameters const &source_node) const;
  void print_strategy_computation_graph(
      std::unordered_map<opmeta::OperatorParameters, MachineView> const &strategy) const;
  void export_strategy_computation_graph(
      std::unordered_map<opmeta::OperatorParameters, MachineView> const &strategy,
      std::string const &out_filename) const;
  void export_strategy_computation_graph(
      std::unordered_map<opmeta::OperatorParameters, MachineView> const &strategy,
      DotFile<opmeta::OperatorParameters> &dot) const;

  std::pair<std::unique_ptr<Graph>, std::unique_ptr<Graph>>
      split_at_node(opmeta::OperatorParameters const &bottleneck) const;
  std::pair<std::unique_ptr<Graph>, std::unique_ptr<Graph>>
      split_horizontal(opmeta::OperatorParameters const &source_node, opmeta::OperatorParameters const &sink_node) const;

  Graph reduced() const;

  opmeta::OperatorParameters find_sink_node() const;
  opmeta::OperatorParameters find_source_node() const;
  void reshape_output_tensor(ParallelTensorShape const &shape);
  std::unique_ptr<Graph>
      with_output_tensor_reshaped_to(ParallelTensorShape const &shape) const;


  static Graph singleton(opmeta::OperatorParameters const &);
  bool empty() const;

  template <typename T>
  T generic_optimal_cost() const;

private:
  void remove_inverse_parallel_ops();
  void replace_subgraph_with_nonempty(
      std::unordered_set<opmeta::OperatorParameters> const &currentNodes, Graph const &replaceWith);
private:
  utils::AdjacencyMultiDiGraph g;
  utils::bidict<utils::Node, opmeta::OperatorParameters> nodeMap;
};

struct GraphOptimizeResult {
  tl::optional<Graph> graph;
  float cost;
  std::unordered_map<utils::Node, MachineView> views;

  friend std::ostream &operator<<(std::ostream &, GraphOptimizeResult const &);
};

/* namespace Utils { */
/* template <> */
/* struct GraphStructure<FlexFlow::PCG::Graph> { */
/*   using G = FlexFlow::PCG::Graph; */
/*   using graph_type = FlexFlow::PCG::Graph; */
/*   using vertex_type = FlexFlow::PCG::Node; */
/*   using edge_type = FlexFlow::PCG::Edge; */

/*   std::unordered_set<vertex_type> get_nodes(G const &g) const { */
/*     std::unordered_set<vertex_type> nodes; */
/*     for (auto const &kv : g.inEdges) { */
/*       nodes.insert(kv.first); */
/*     } */
/*     for (auto const &kv : g.outEdges) { */
/*       nodes.insert(kv.first); */
/*     } */

/*     return nodes; */
/*   } */

/*   std::unordered_set<edge_type> get_incoming_edges(G const &g, */
/*                                                    vertex_type const &n) const { */
/*     if (g.inEdges.find(n) == g.inEdges.end()) { */
/*       return {}; */
/*     } else { */
/*       return {g.inEdges.at(n).begin(), g.inEdges.at(n).end()}; */
/*     } */
/*   } */

/*   std::unordered_set<edge_type> get_outgoing_edges(G const &g, */
/*                                                    vertex_type const &n) const { */
/*     if (g.outEdges.find(n) == g.outEdges.end()) { */
/*       return {}; */
/*     } else { */
/*       return {g.outEdges.at(n).begin(), g.outEdges.at(n).end()}; */
/*     } */
/*   } */

/*   vertex_type get_src(G const &g, edge_type const &e) const { */
/*     return e.srcOp; */
/*   } */

/*   vertex_type get_dst(G const &g, edge_type const &e) const { */
/*     return e.dstOp; */
/*   } */

/*   void set_src(G const &g, edge_type &e, vertex_type const &n) const { */
/*     e.srcOp = n; */
/*   } */

/*   void set_dst(G const &g, edge_type &e, vertex_type const &n) const { */
/*     e.dstOp = n; */
/*   } */
/* }; */

size_t dp_state_hash(Graph const *graph,
                     Node const &sink_node,
                     MachineView const &sink_view,
                     Node const &source_node,
                     MachineView const &source_view,
                     MachineResource const &resource);

// template <>
// struct invalid_node<Graph, GraphStructure<Graph>> {
//   using G = Graph;
//   using Structure = GraphStructure<Graph>;
//   using vertex_type = typename Structure::vertex_type;
// 
//   vertex_type operator()() const {
//     return vertex_type::INVALID_NODE;
//   }
// };
// 
// template <>
// struct invalid_node<BasicGraph<Node>, GraphStructure<BasicGraph<Node>>> {
//   Node operator()() const {
//     return Node::INVALID_NODE;
//   }
// };

/* } // namespace Utils */
} // namespace ffc
} // namespace FlexFlow

#endif
