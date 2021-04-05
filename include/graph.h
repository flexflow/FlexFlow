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
//#include "ffconst.h"
#include "model.h"
#include <unordered_set>
#include "dot_file.h"
#include "basic_graph.h"
#include "graph_structures.h"

struct Edge {
  Edge(void);
  Edge(const Node& _srcOp,
       const Node& _dstOp,
       int _srcIdx,
       int _dstIdx);
  bool operator==(const Edge &rhs) const;
  Node srcOp, dstOp;
  int srcIdx, dstIdx;
};

struct EdgeCompare {
  bool operator()(const Edge& a, const Edge& b) const {
    if (!(a.srcOp == b.srcOp)) return a.srcOp < b.srcOp;
    if (!(a.dstOp == b.dstOp)) return a.dstOp < b.dstOp;
    if (a.srcIdx != b.srcIdx) return a.srcIdx < b.srcIdx;
    if (a.dstIdx != b.dstIdx) return a.dstIdx < b.dstIdx;
    return false;
  };
};

namespace std {
  template <>
  struct hash<Edge>
  {
    size_t operator()(const Edge& e) const
    {
      size_t res = 17;
      res = res * 31 + hash<size_t>()((size_t)e.srcOp.guid);
      res = res * 31 + hash<size_t>()((size_t)e.dstOp.guid);
      res = res * 31 + hash<int>()(e.srcIdx);
      res = res * 31 + hash<int>()(e.dstIdx);
      return res;
    }
  };

  template <>
  struct hash<Node>
  {
    size_t operator()(const Node& n) const
    {
      return n.guid;
    }
  };
}

struct NodeCompare {
  bool operator()(const Node& a, const Node& b) const {
    if (a.guid != b.guid) return a.guid < b.guid;
    return a.ptr < b.ptr;
  };
};

struct GraphOptimalViewSerialized {
#ifdef LEGION_MAX_RETURN_SIZE
  static const size_t buffer_size = LEGION_MAX_RETURN_SIZE - 8;
#else
  static const size_t buffer_size = 32 * 1024 - 8;
#endif
  size_t total_bytes;
  char data[buffer_size];
};

struct NodeAssignment {
  Node node;
  MachineView view;
};

size_t dp_state_hash(const Graph* graph,
                     const Node& sink_node,
                     const MachineView& sink_view,
                     const Node& source_node,
                     const MachineView& source_view,
                     const MachineResource& resource);

class SearchHelper {
public:
  SearchHelper(FFModel *model);

  float graph_cost(const Graph* graph,
                   const Node& sink_node,
                   const MachineView& sink_view,
                   const Node& source_node,
                   const MachineView& source_view,
                   const MachineResource& resources,
                   bool include_sink_compute_time,
                   bool constructing_optimal_view = false) const;
  void construct_optimal_view(const Graph* graph,
                              const Node& sink_node,
                              const MachineView& sink_view,
                              const Node& source_node,
                              const MachineView& source_view,
                              const MachineResource& resources,
                              bool include_sink_compute_time,
                              float optimal_cost,
                              std::unordered_map<Node, MachineView>& optimal_views) const;
  float find_optimal_sequence_graph_time(Graph const *pre_graph,
                                         Graph const *post_graph,
                                         Node const &bottleneck_node,
                                         NodeAssignment const &source,
                                         NodeAssignment const &sink,
                                         MachineResource const &resources) const;
  void find_optimal_sequence_graph_views(Graph const *first_graph,
                                         Graph const *second_graph,
                                         Node const &bn_node,
                                         NodeAssignment const &source,
                                         NodeAssignment const &sink,
                                         MachineResource const &resources,
                                         float optimal_cost,
                                         std::unordered_map<Node, MachineView>& optimal_views) const;
  float find_optimal_nonsequence_graph_time(Graph const *g,
                                            NodeAssignment const &source,
                                            NodeAssignment const &sink,
                                            MachineResource const &resources) const;
  std::vector<MachineView> get_valid_machine_views(const Op* op, const MachineResource& resource) const;
private:
  FFModel *model;

  mutable std::unordered_map<size_t, float> cached_graph_costs;
  mutable std::unordered_map<size_t, std::unique_ptr<const std::vector<MachineView>>> cached_operator_valid_views;
};

class Graph {
public:
  Graph(FFModel* model);
  void add_edge(const Node& srcOp,
                const Node& dstOp,
                int srcIdx,
                int dstIdx);
  void add_edge(const Edge& e);
  void remove_edge(const Edge& e);
  bool has_edge(const Node& srcOp,
                const Node& dstOp,
                int srcIdx,
                int dstIdx);
  bool has_edge(const Edge& e);
  float total_cost();
  void construct_optimal_view(float optimal_cost,
                              std::unordered_map<Node, MachineView>& optimal_views);
  size_t hash(void) const;
  void print(void) const;
  bool check_correctness(void);
  bool has_loop(void);
  bool map_operators_to_layers(std::vector<Op*>& layers) const;
  static GraphOptimalViewSerialized graph_optimize_task(const Legion::Task *task,
             const std::vector<Legion::PhysicalRegion> &regions,
             Legion::Context ctx, Legion::Runtime *runtime);
  Node find_bottleneck_node(const Node& sink_node, const Node& source_node) const;
  void export_strategy_computation_graph(std::unordered_map<Node, MachineView> const &strategy, std::unique_ptr<std::ostream> out) const;
  void export_strategy_computation_graph(std::unordered_map<Node, MachineView> const &strategy, std::string const &out_filename) const;
  void export_strategy_computation_graph(std::unordered_map<Node, MachineView> const &strategy, DotFile<Node> &dot) const;

  std::pair<std::unique_ptr<Graph>, std::unique_ptr<Graph>> split_at_node(Node const &bottleneck) const;
public:
  FFModel* model;
  SearchHelper* search;
  std::unordered_map<Node, std::unordered_set<Edge> > inEdges, outEdges;
};

namespace flexflow::graph {
  template <>
  struct GraphStructure<::Graph> {
    using G = ::Graph;
    using vertex_type = ::Node;
    using edge_type = ::Edge;

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

    std::unordered_set<edge_type> get_incoming_edges(G const &g, vertex_type const &n) const {
      if (g.inEdges.find(n) == g.inEdges.end()) {
        return {};
      } else {
        return {g.inEdges.at(n).begin(), g.inEdges.at(n).end()};
      }
    }

    std::unordered_set<edge_type> get_outgoing_edges(G const &g, vertex_type const &n) const {
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
  struct invalid_node<::Graph, GraphStructure<::Graph>> {
    using G = ::Graph;
    using Structure = GraphStructure<::Graph>;
    using vertex_type = typename Structure::vertex_type;

    vertex_type operator()() const {
      return vertex_type::INVALID_NODE;
    }
  };

  template <>
  struct invalid_node<::flexflow::graph::BasicGraph<Node>, GraphStructure<::flexflow::graph::BasicGraph<Node>>> {
    Node operator()() const {
      return Node::INVALID_NODE;
    }
  };
}
#endif
