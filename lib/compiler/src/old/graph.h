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
#include "graph_structures.h"
#include "op-attrs/op-attrs.h"
#include "pcg/machine_view.h"
#include "utils/bidict.h"
#include "utils/dot_file.h"
#include "utils/graph.h"
#include "utils/graph/serialparallel.h"
#include "utils/recursive_logger.h"
#include <memory>
#include <unordered_set>

// extern LegionRuntime::Logger::Category log_dp;

/* namespace FlexFlow { */
/* namespace ffc { */

/* class SearchHelper; */

/* struct GraphOptimalViewSerialized { */
/* #ifdef LEGION_MAX_RETURN_SIZE */
/*   static const size_t buffer_size = LEGION_MAX_RETURN_SIZE - 8; */
/* #else */
/*   static const size_t buffer_size = 1024 * 1024 - 8; */
/* #endif */
/*   size_t total_bytes; */
/*   char data[buffer_size]; */
/* }; */

/* class Graph { */
/* public: */
/*   Graph() = default; */
/*   Graph(std::string const &logger_name); */
/*   Graph(std::shared_ptr<spdlog::logger> const &logger); */

/*   void add_edge(utils::Node const &srcOp, utils::Node const &dstOp, int
 * srcIdx, int dstIdx); */
/*   utils::Node add_node(opmeta::OperatorParameters const &); */
/*   void add_edge(utils::MultiDiEdge const &e); */
/*   void remove_node(utils::Node const &, bool purge_edges = false); */
/*   void remove_edge(utils::MultiDiEdge const &e, bool remove_node_if_unused =
 * true); */
/*   bool has_edge(utils::MultiDiEdge const &e) const; */
/*   void replace_subgraph(std::unordered_set<opmeta::OperatorParameters> const
 * &currentNodes, */
/*                         Graph const &replaceWith); */
/*   Graph subgraph(std::unordered_set<utils::Node> const &nodes) const; */
/*   void contract_out_node(opmeta::OperatorParameters const &); */
/*   float optimal_cost() const; */
/*   std::unordered_map<opmeta::OperatorParameters, MachineView> optimal_views()
 * const; */
/*   void remove_input_nodes(); */
/*   void duplicate_input_node(opmeta::OperatorParameters const &); */
/*   void duplicate_input_nodes(); */
/*   opmeta::OperatorParameters clone_node(opmeta::OperatorParameters const &);
 */
/*   std::pair<opmeta::OperatorParameters,
 * std::unordered_set<opmeta::OperatorParameters>> */
/*       deduplicate_input_node(opmeta::OperatorParameters const &); */
/*   std::unordered_map<opmeta::OperatorParameters, opmeta::OperatorParameters>
 * deduplicate_input_nodes(); */
/*   opmeta::OperatorParameters declone_node(opmeta::OperatorParameters const
 * &); */

/*   size_t hash(void) const; */
/*   void print(void) const; */
/*   void print_dot() const; */
/*   void print_dot(std::ostream &) const; */

/*   bool check_correctness(void); */
/*   bool has_loop(void); */
/*   //bool map_operators_to_layers(std::vector<OpMeta *> &layers) const; */
/*   //static GraphOptimalViewSerialized */
/*   //    graph_optimize_task(Legion::Task const *task, */
/*   //                        std::vector<Legion::PhysicalRegion> const
 * &regions, */
/*   //                        Legion::Context ctx, */
/*   //                        Legion::Runtime *runtime); */
/*   /1* opmeta::OperatorParameters
 * find_bottleneck_node(opmeta::OperatorParameters const &sink_node, *1/ */
/*   /1*                           opmeta::OperatorParameters const
 * &source_node) const; *1/ */
/*   void print_strategy_computation_graph( */
/*       std::unordered_map<opmeta::OperatorParameters, MachineView> const
 * &strategy) const; */
/*   void export_strategy_computation_graph( */
/*       std::unordered_map<opmeta::OperatorParameters, MachineView> const
 * &strategy, */
/*       std::string const &out_filename) const; */
/*   void export_strategy_computation_graph( */
/*       std::unordered_map<opmeta::OperatorParameters, MachineView> const
 * &strategy, */
/*       DotFile<opmeta::OperatorParameters> &dot) const; */

/*   /1* std::pair<std::unique_ptr<Graph>, std::unique_ptr<Graph>> *1/ */
/*   /1*     split_at_node(opmeta::OperatorParameters const &bottleneck) const;
 * *1/ */
/*   /1* std::pair<std::unique_ptr<Graph>, std::unique_ptr<Graph>> *1/ */
/*   /1*     split_horizontal(opmeta::OperatorParameters const &source_node,
 * opmeta::OperatorParameters const &sink_node) const; *1/ */

/*   Graph reduced() const; */

/*   opmeta::OperatorParameters find_sink_node() const; */
/*   opmeta::OperatorParameters find_source_node() const; */
/*   void reshape_output_tensor(opmeta::ParallelTensorShape const &shape); */
/*   std::unique_ptr<Graph> */
/*       with_output_tensor_reshaped_to(opmeta::ParallelTensorShape const
 * &shape) const; */

/*   static Graph singleton(opmeta::OperatorParameters const &); */
/*   bool empty() const; */

/*   template <typename T> */
/*   T generic_optimal_cost() const; */

/* private: */
/*   void remove_inverse_parallel_ops(); */
/*   void replace_subgraph_with_nonempty( */
/*       std::unordered_set<opmeta::OperatorParameters> const &currentNodes,
 * Graph const &replaceWith); */
/* private: */
/*   Graph(utils::AdjacencyMultiDiGraph const &, utils::bidict<utils::Node,
 * opmeta::OperatorParameters> const &, std::shared_ptr<spdlog::logger> const
 * &); */

/*   utils::AdjacencyMultiDiGraph g; */
/*   utils::bidict<utils::Node, opmeta::OperatorParameters> nodeMap; */
/*   std::shared_ptr<spdlog::logger> logger; */
/* }; */

/* struct GraphOptimizeResult { */
/*   tl::optional<Graph> graph; */
/*   float cost; */
/*   std::unordered_map<utils::Node, MachineView> views; */

/*   friend std::ostream &operator<<(std::ostream &, GraphOptimizeResult const
 * &); */
/* }; */

/* /1* namespace Utils { *1/ */
/* /1* template <> *1/ */
/* /1* struct GraphStructure<FlexFlow::PCG::Graph> { *1/ */
/* /1*   using G = FlexFlow::PCG::Graph; *1/ */
/* /1*   using graph_type = FlexFlow::PCG::Graph; *1/ */
/* /1*   using vertex_type = FlexFlow::PCG::Node; *1/ */
/* /1*   using edge_type = FlexFlow::PCG::Edge; *1/ */

/* /1*   std::unordered_set<vertex_type> get_nodes(G const &g) const { *1/ */
/* /1*     std::unordered_set<vertex_type> nodes; *1/ */
/* /1*     for (auto const &kv : g.inEdges) { *1/ */
/* /1*       nodes.insert(kv.first); *1/ */
/* /1*     } *1/ */
/* /1*     for (auto const &kv : g.outEdges) { *1/ */
/* /1*       nodes.insert(kv.first); *1/ */
/* /1*     } *1/ */

/* /1*     return nodes; *1/ */
/* /1*   } *1/ */

/* /1*   std::unordered_set<edge_type> get_incoming_edges(G const &g, *1/ */
/* /1*                                                    vertex_type const &n)
 * const { *1/ */
/* /1*     if (g.inEdges.find(n) == g.inEdges.end()) { *1/ */
/* /1*       return {}; *1/ */
/* /1*     } else { *1/ */
/* /1*       return {g.inEdges.at(n).begin(), g.inEdges.at(n).end()}; *1/ */
/* /1*     } *1/ */
/* /1*   } *1/ */

/* /1*   std::unordered_set<edge_type> get_outgoing_edges(G const &g, *1/ */
/* /1*                                                    vertex_type const &n)
 * const { *1/ */
/* /1*     if (g.outEdges.find(n) == g.outEdges.end()) { *1/ */
/* /1*       return {}; *1/ */
/* /1*     } else { *1/ */
/* /1*       return {g.outEdges.at(n).begin(), g.outEdges.at(n).end()}; *1/ */
/* /1*     } *1/ */
/* /1*   } *1/ */

/* /1*   vertex_type get_src(G const &g, edge_type const &e) const { *1/ */
/* /1*     return e.srcOp; *1/ */
/* /1*   } *1/ */

/* /1*   vertex_type get_dst(G const &g, edge_type const &e) const { *1/ */
/* /1*     return e.dstOp; *1/ */
/* /1*   } *1/ */

/* /1*   void set_src(G const &g, edge_type &e, vertex_type const &n) const {
 * *1/ */
/* /1*     e.srcOp = n; *1/ */
/* /1*   } *1/ */

/* /1*   void set_dst(G const &g, edge_type &e, vertex_type const &n) const {
 * *1/ */
/* /1*     e.dstOp = n; *1/ */
/* /1*   } *1/ */
/* /1* }; *1/ */

/* size_t dp_state_hash(Graph const *graph, */
/*                      opmeta::OperatorParameters const &sink_node, */
/*                      MachineView const &sink_view, */
/*                      opmeta::OperatorParameters const &source_node, */
/*                      MachineView const &source_view, */
/*                      MachineResource const &resource); */

/* // template <> */
/* // struct invalid_node<Graph, GraphStructure<Graph>> { */
/* //   using G = Graph; */
/* //   using Structure = GraphStructure<Graph>; */
/* //   using vertex_type = typename Structure::vertex_type; */
/* // */
/* //   vertex_type operator()() const { */
/* //     return vertex_type::INVALID_NODE; */
/* //   } */
/* // }; */
/* // */
/* // template <> */
/* // struct invalid_node<BasicGraph<Node>, GraphStructure<BasicGraph<Node>>> {
 */
/* //   Node operator()() const { */
/* //     return Node::INVALID_NODE; */
/* //   } */
/* // }; */

/* /1* } // namespace Utils *1/ */
/* } // namespace ffc */
/* } // namespace FlexFlow */

#endif
