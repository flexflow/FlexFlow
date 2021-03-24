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
  size_t hash(void) const;
  void print(void) const;
  bool check_correctness(void);
  bool has_loop(void);
  Node find_bottleneck_node(const Node& sink_node,
                              const Node& source_node,
                              std::unordered_set<Node>& used_nodes) const;
public:
  FFModel* model;
  std::unordered_map<Node, std::unordered_set<Edge> > inEdges, outEdges;
};
#endif
