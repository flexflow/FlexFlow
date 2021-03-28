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
#include "dominators.h"

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

class Graph {
public:
  using Node = ::Node;
  using Edge = ::Edge;

  Graph(FFModel* model);
  void add_edge(const Node& srcOp,
                const Node& dstOp,
                int srcIdx,
                int dstIdx);
  void add_edge(const Edge& e);
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
  Node find_bottleneck_node(const Node& sink_node,
                              const Node& source_node,
                              std::unordered_set<Node>& used_nodes) const;
  void export_strategy_computation_graph(std::unordered_map<Node, MachineView> const &strategy, std::unique_ptr<std::ostream> out) const;
  void export_strategy_computation_graph(std::unordered_map<Node, MachineView> const &strategy, std::string const &out_filename) const;
  void export_strategy_computation_graph(std::unordered_map<Node, MachineView> const &strategy, DotFile<Node> &dot) const;
public:
  FFModel* model;
  std::unordered_map<Node, std::unordered_set<Edge> > inEdges, outEdges;
};

namespace flexflow::dominators {
  template <>
  struct GraphStructure<::Graph> {
    std::unordered_set<Node> get_nodes(Graph const &g) const {
      std::unordered_set<Node> nodes;
      for (auto const &kv : g.inEdges) {
        nodes.insert(kv.first);
      }
      for (auto const &kv : g.outEdges) {
        nodes.insert(kv.first);
      }

      return nodes;
    }

    std::unordered_set<Edge> get_incoming_edges(Graph const &g, Node const &n) const {
      if (g.inEdges.find(n) == g.inEdges.end()) {
        return {};
      } else {
        return {g.inEdges.at(n).begin(), g.inEdges.at(n).end()};
      }
    }

    std::unordered_set<Edge> get_outgoing_edges(Graph const &g, Node const &n) const {
      if (g.outEdges.find(n) == g.outEdges.end()) {
        return {};
      } else {
        return {g.outEdges.at(n).begin(), g.outEdges.at(n).end()};
      }
    }

    Node get_src(Graph const &g, Edge const &e) const {
      return e.srcOp;
    }

    Node get_dst(Graph const &g, Edge const &e) const {
      return e.dstOp;
    }
  };
}
#endif
