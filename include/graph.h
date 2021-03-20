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
struct Edge {
  Edge(void);
  Edge(Op* _srcOp, Op* _dstOp, int _srcIdx, int _dstIdx);
  Op *srcOp, *dstOp;
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

struct OpCompare {
  bool operator()(const Op* a, const Op* b) const {
    if (a->op_guid != b->op_guid)
      return a->op_guid < b->op_guid;
    return false;
  };
};

class Graph {
public:
  Graph(FFModel* model);
  void add_edge(Op* srcOp, Op* dstOp, int srcIdx, int dstIdx);
  bool has_edge(Op* srcOp, Op* dstOp, int srcIdx, int dstIdx);
  float total_cost();
  size_t hash(void);
  void print(void);
  bool check_correctness(void);
  bool has_loop(void);
public:
  FFModel* model;
  float totalCost;
  std::map<Op*, std::set<Edge, EdgeCompare>, OpCompare> inEdges, outEdges;
};
#endif
