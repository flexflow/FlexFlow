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
#include "graph.h"

Edge::Edge(void)
: srcOp(NULL), dstOp(NULL), srcIdx(-1), dstIdx(-1)
{}

Edge::Edge(Op* _srcOp, Op* _dstOp, int _srcIdx, int _dstIdx)
: srcOp(_srcOp), dstOp(_dstOp), srcIdx(_srcIdx), dstIdx(_dstIdx)
{}

Graph::Graph(FFModel* _model)
: model(_model), totalCost(-1.0f)
{
}

void Graph::add_edge(Op* srcOp, Op* dstOp, int srcIdx, int dstIdx)
{
  if (inEdges.find(dstOp) == inEdges.end()) {
    inEdges[dstOp];
  }
  if (outEdges.find(srcOp) == outEdges.end()) {
    outEdges[srcOp];
  }
  Edge e(srcOp, dstOp, srcIdx, dstIdx);
  inEdges[dstOp].insert(e);
  outEdges[srcOp].insert(e);
}

bool Graph::has_edge(Op* srcOp, Op* dstOp, int srcIdx, int dstIdx)
{
  Edge e(srcOp, dstOp, srcIdx, dstIdx);
  return (inEdges[dstOp].find(e) != inEdges[dstOp].end());
}

size_t Graph::hash(void)
{
  size_t total = 0;
  std::map<Op*, std::set<Edge, EdgeCompare>, OpCompare>::const_iterator it;
  for (it = inEdges.begin(); it != inEdges.end(); it++) {
    size_t my = 17 * 31 + (size_t)(it->first->op_guid);
    std::set<Edge, EdgeCompare> list = it->second;
    std::set<Edge, EdgeCompare>::const_iterator it2;
    for (it2 = list.begin(); it2 != list.end(); it2++) {
      Edge e = *it2;
      my = my * 31 + std::hash<size_t>()((size_t)(e.srcOp));
      my = my * 31 + std::hash<int>()(e.srcIdx);
      my = my * 31 + std::hash<int>()(e.dstIdx);
    }
    total += my;
  }
  return total;
}

void Graph::print(void)
{
  std::map<Op*, std::set<Edge, EdgeCompare>, OpCompare>::const_iterator it;
  for (it = inEdges.begin(); it != inEdges.end(); it++) {
    if (it->first->op_guid == 0) continue;
    printf("	op_guid(%zu) type(%d): ", it->first->op_guid, it->first->op_type);
    std::set<Edge, EdgeCompare> list = it->second;
    std::set<Edge, EdgeCompare>::const_iterator it2;
    for (it2 = list.begin(); it2 != list.end(); it2++) {
      Edge e = *it2;
      printf(" inEdge(op_guid(%zu) idx(%d))", e.srcOp->op_guid, e.srcIdx);
    }
    printf("\n");
    // if (it->first.ptr->type == OP_CONV2D) {
    //   it->first.ptr->inputs[1].print_info("conv weight");
    // }
    // else if (it->first.ptr->type == OP_BROADCAST_ADD) {
    //   it->first.ptr->inputs[1].print_info("conv new bias");
    // }
    // else if (it->first.ptr->type == OP_BATCHNORM) {
    //   it->first.ptr->inputs[1].print_info("gamma");
    //   it->first.ptr->inputs[2].print_info("beta");
    //   it->first.ptr->inputs[3].print_info("mean");
    //   it->first.ptr->inputs[4].print_info("var");
    // }
  }
}

bool Graph::has_loop(void)
{
  std::map<Op*, int, OpCompare> todos;
  std::map<Op*, std::set<Edge, EdgeCompare>, OpCompare>::const_iterator it;
  std::vector<Op*> opList;
  for (it = inEdges.begin(); it != inEdges.end(); it++) {
    std::set<Edge, EdgeCompare> inList = it->second;
    todos[it->first] = (int)inList.size();
    if (todos[it->first] == 0)
      opList.push_back(it->first);
  }
  size_t i = 0;
  while (i < opList.size()) {
    Op* op = opList[i++];
    std::set<Edge, EdgeCompare> outList = outEdges[op];
    std::set<Edge, EdgeCompare>::const_iterator it2;
    for (it2 = outList.begin(); it2 != outList.end(); it2++) {
      todos[it2->dstOp] --;
      if (todos[it2->dstOp] == 0) {
        opList.push_back(it2->dstOp);
      }
    }
  }
  return (opList.size() < inEdges.size());
}

bool Graph::check_correctness(void)
{
  bool okay = true;
  std::map<Op*, std::set<Edge, EdgeCompare>, OpCompare>::const_iterator it;
  for (it = outEdges.begin(); it != outEdges.end(); it++) {
    std::set<Edge, EdgeCompare> list = it->second;
    std::set<Edge, EdgeCompare>::const_iterator it2;
    for (it2 = list.begin(); it2 != list.end(); it2++) {
      Edge e = *it2;
      if (!has_edge(e.srcOp, e.dstOp, e.srcIdx, e.dstIdx)) assert(false);
      if (e.srcOp == NULL) continue;
      Tensor srcTensor = e.srcOp->outputs[e.srcIdx];
      Tensor dstTensor = e.dstOp->inputs[e.dstIdx];
      if (srcTensor->num_dims != dstTensor->num_dims) assert(false);
      for (int i = 0; i < srcTensor->num_dims; i++) {
        assert(srcTensor->dims[i] == dstTensor->dims[i]);
      }
    }
  }
  return okay;
}

float Graph::total_cost(void)
{
  return 0.0f;
}
