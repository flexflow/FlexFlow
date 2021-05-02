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

#include "substitution.h"
#include <chrono>
#include "dot_file.h"
#include "dominators.h"
#include "ops/embedding.h"
#include "ops/linear.h"
#include "ops/conv_2d.h"
#include "ops/pool_2d.h"
#include "ops/attention.h"
#include "ops/flat.h"
#include "ops/element_binary.h"
#include "ops/split.h"
#include "ops/noop.h"
#include "ops/softmax.h"
#include "parallel_ops/combine.h"
#include "parallel_ops/partition.h"
#include "parallel_ops/replicate.h"
#include "parallel_ops/fused_parallel_op.h"
#include "parallel_ops/reduction.h"

using namespace Legion;

const TensorX TensorX::NO_TX = TensorX();

GraphXfer* create_replicate_linear_combine(FFModel* model,
                                           int num_dims,
                                           int num_parts,
                                           ActiMode activation,
                                           bool use_bias);

GraphXfer* create_partition_linear_combine(FFModel* model,
                                           int num_dims,
                                           int num_parts,
                                           ActiMode activation,
                                           bool use_bias);

GraphXfer* create_partition_attention_combine(FFModel* model,
                                              int num_heads,
                                              int num_parts);

GraphXfer* create_replicate_attention_reduce(FFModel* model,
                                             int num_heads,
                                             int num_parts);

GraphXfer* create_partition_add_combine(FFModel* model,
                                        int parallel_dim,
                                        int num_parts);

GraphXfer* create_partition_concat_combine(FFModel* model,
                                           int num_inputs,
                                           int concat_dim,
                                           int parallel_dim,
                                           int num_parts);

GraphXfer* create_partition_softmax_combine(FFModel* model,
                                            int softmax_dim,
                                            int part_dim,
                                            int num_parts);

GraphXfer* eliminate_combine_partition(FFModel* model,
                                       int parallel_dim,
                                       int num_parts);

PMConstraint::PMConstraint(Compare c, PMParameter p, int v)
: comp(c), para(p), value(v) {}

TNConstraint::TNConstraint(Compare c, TNParameter p, DIMParameter d, int v)
: singlePara(true), comp(c), para1(p), dim1(d), value(v) {}

TNConstraint::TNConstraint(Compare c, TNParameter p1, DIMParameter d1,
                           TNParameter p2, DIMParameter d2)
: singlePara(false), comp(c), para1(p1), para2(p2), dim1(d1), dim2(d2) {}

Tensor TensorX::to_tensor(const GraphXfer* xfer) const
{
  if (op != NULL) {
    assert(op->mapOp.ptr != NULL);
    return op->mapOp.ptr->outputs[idx];
  } else {
    const auto& it = xfer->mappedInputs.find(idx);
    assert(it != xfer->mappedInputs.end());
    Node op = it->second.first;
    int outIdx = it->second.second;
    return op.ptr->outputs[outIdx];
  }
}

OpX::OpX(const OperatorType _type,
         int num_inputs,
         int num_outputs,
         const TensorX& input0,
         const TensorX& input1,
         const TensorX& input2,
         const TensorX& input3)
: type(_type), mapOp(Node::INVALID_NODE), matchOpX(NULL)
{
  TensorX all_inputs[MAX_NUM_INPUTS];
  all_inputs[0] = input0;
  all_inputs[1] = input1;
  all_inputs[2] = input2;
  all_inputs[3] = input3;
  for (int i = 0; i < num_inputs; i++) {
    inputs.push_back(all_inputs[i]);
  }
  for (int i = 0; i < num_outputs; i++) {
    TensorX out(this, i);
    outputs.push_back(out);
  }
}

OpX::OpX(const OperatorType _type,
         int num_inputs,
         int num_outputs,
         const TensorX* input_array)
: type(_type), mapOp(Node::INVALID_NODE), matchOpX(NULL)
{
  for (int i = 0; i < num_inputs; i++) {
    inputs.push_back(input_array[i]);
  }
  for (int i = 0; i < num_outputs; i++) {
    TensorX out(this, i);
    outputs.push_back(out);
  }
}

bool OpX::add_pm_constraint(Compare comp, PMParameter para, int value)
{
  PMConstraint pmc(comp, para, value);
  pmConstraints.push_back(pmc);
  return true;
}

bool OpX::add_input_constraint(Compare comp, TNParameter para,
                               DIMParameter dim, int value)
{
  TNConstraint tnc(comp, para, dim, value);
  tnConstraints.push_back(tnc);
  return true;
}

bool OpX::add_input_constraint(Compare comp,
                               TNParameter para1, DIMParameter dim1,
                               TNParameter para2, DIMParameter dim2)
{
  TNConstraint tnc(comp, para1, dim1, para2, dim2);
  tnConstraints.push_back(tnc);
  return true;
}

bool OpX::get_pm_constraint(PMParameter para, int& value) const
{
  for (size_t i = 0; i < pmConstraints.size(); i++)
    if ((pmConstraints[i].comp == COMPARE_EQ)
    && (pmConstraints[i].para == para)) {
      value = pmConstraints[i].value;
      return true;
    }
  return false;
}

GraphXfer::GraphXfer(FFModel* _model)
: model(_model), tensorId(10)
{}

TensorX GraphXfer::new_tensor(void)
{
  TensorX t;
  t.op = NULL;
  t.idx = tensorId++;
  return t;
}

bool GraphXfer::map_output(const TensorX& src, const TensorX& dst)
{
  mappedOutputs[src] = dst;
  return true;
}

bool GraphXfer::can_match(OpX* srcOp, const Node& op, Graph* graph)
{
  if (srcOp->type != op.ptr->op_type) return false;
  // check num input tensors
  if ((int)srcOp->inputs.size() != op.ptr->numInputs) return false;
  // check pmConstraints
  for (size_t i = 0; i < srcOp->pmConstraints.size(); i++) {
    PMConstraint pmc = srcOp->pmConstraints[i];
    int actValue = 0;
    assert(op.ptr->get_int_parameter(pmc.para, &actValue));
    //printf("pmc[%d] para(%d) comp(%d) value(%d) actValue(%d)\n",
    //       i, pmc.para, pmc.comp, pmc.value, actValue);
    switch (pmc.comp) {
      case COMPARE_EQ:
      {
        if (actValue != pmc.value) return false;
        break;
      }
      case COMPARE_NE:
      {
        if (actValue == pmc.value) return false;
        break;
      }
      case COMPARE_LT:
      {
        if (actValue >= pmc.value) return false;
        break;
      }
      case COMPARE_LE:
      {
        if (actValue > pmc.value) return false;
        break;
      }
      case COMPARE_GT:
      {
        if (actValue <= pmc.value) return false;
        break;
      }
      case COMPARE_GE:
      {
        if (actValue < pmc.value) return false;
        break;
      }
      default:
        assert(false);
    }
  }
  // check inputs
  std::map<int, std::pair<Node, int> > newMapInputs;
  for (size_t i = 0; i < srcOp->inputs.size(); i++) {
    TensorX in = srcOp->inputs[i];
    if (in.op == NULL) {
      // input tensor
      std::multimap<int, std::pair<Node, int> >::const_iterator it;
      it = mappedInputs.find(in.idx);
      if (it != mappedInputs.end()) {
        Node mappedOp = it->second.first;
        int mappedIdx = it->second.second;
        if (!(graph->has_edge(mappedOp, op, mappedIdx, i)))
          return false;
      } else {
        std::map<int, std::pair<Node, int> >::const_iterator newit;
        newit = newMapInputs.find(in.idx);
        if (newit != newMapInputs.end()) {
          Node mappedOp = newit->second.first;
          int mappedIdx = newit->second.second;
          if (!(graph->has_edge(mappedOp, op, mappedIdx, i)))
            return false;
        } else {
          const auto& list = graph->inEdges.find(op)->second;
          for (const auto& e : list) {
            if (e.dstIdx == (int)i) {
              newMapInputs.insert(std::make_pair(in.idx,
                                      std::make_pair(e.srcOp, e.srcIdx)));
            }
          }
        }
        // Do nothing when we check the match
        /* mapped in.idx to an op
        std::set<Edge, EdgeCompare> list = graph->inEdges.find(op)->second;
        std::set<Edge, EdgeCompare>::const_iterator it2;
        for (it2 = list.begin(); it2 != list.end(); it2++) {
          Edge e = *it2;
          if (e.dstIdx == i)
            mappedInputs[in.idx] = std::make_pair(e.srcOp, e.srcIdx);
        }*/
      }
    } else {
      // intermediate tensor
      assert(in.op->mapOp != Node::INVALID_NODE);
      if (!(graph->has_edge(in.op->mapOp, op, in.idx, i)))
        return false;
    }
  }
  // check tnConstraints
  for (size_t i = 0; i < srcOp->tnConstraints.size(); i++) {
    TNConstraint tnc = srcOp->tnConstraints[i];
    int actValue = 0, expValue = 0;
    if (tnc.singlePara) {
      assert(op.ptr->get_tensor_parameter(tnc.para1, tnc.dim1, &actValue));
      expValue = tnc.value;
    } else {
      assert(op.ptr->get_tensor_parameter(tnc.para1, tnc.dim1, &actValue));
      assert(op.ptr->get_tensor_parameter(tnc.para2, tnc.dim2, &expValue));
    }
    switch (tnc.comp) {
      case COMPARE_EQ:
      {
        if (actValue != expValue) return false;
        break;
      }
      case COMPARE_NE:
      {
        if (actValue == expValue) return false;
        break;
      }
      case COMPARE_LT:
      {
        if (actValue >= expValue) return false;
        break;
      }
      case COMPARE_LE:
      {
        if (actValue > expValue) return false;
        break;
      }
      case COMPARE_GT:
      {
        if (actValue <= expValue) return false;
        break;
      }
      case COMPARE_GE:
      {
        if (actValue < expValue) return false;
        break;
      }
      default:
        assert(false);
    }
  }
  return true;
}

void GraphXfer::match(OpX* srcOp, const Node& op, Graph* graph)
{
  for (size_t i = 0; i < srcOp->inputs.size(); i++) {
    TensorX in = srcOp->inputs[i];
    if (in.op == NULL) {
      // Update mappedInputs
      const auto& list = graph->inEdges.find(op)->second;
      for (const auto& e : list) {
        if (e.dstIdx == (int)i) {
          mappedInputs.insert(std::make_pair(in.idx,
                                  std::make_pair(e.srcOp, e.srcIdx)));
        }
      }
    }
  }
  // Map srcOp to Op
  srcOp->mapOp = op;
  mappedOps[op] = srcOp;
}

void GraphXfer::unmatch(OpX* srcOp, const Node& op, Graph* graph)
{
  for (size_t i = 0; i < srcOp->inputs.size(); i++) {
    TensorX in = srcOp->inputs[i];
    if (in.op == NULL) {
      // Update mappedInputsa
      std::multimap<int, std::pair<Node, int> >::iterator it;
      it = mappedInputs.find(in.idx);
      mappedInputs.erase(it);
    }
  }
  // Unmap op
  mappedOps.erase(op);
  srcOp->mapOp.guid = 0;
  srcOp->mapOp.ptr = NULL;
}

void GraphXfer::run(int depth, Graph* graph,
                    std::priority_queue<Graph*, std::vector<Graph*>, GraphCompare>& candidates,
                    std::unordered_set<size_t>& hashmap, float threshold, int maxNumOps)
{
  //printf("run: depth(%d) srcOps.size(%zu) graph.size(%zu) candidates(%zu)\n", depth, srcOps.size(), graph->inEdges.size(), candidates.size());
  if (depth >= (int)srcOps.size()) {
    // Create dst operators
    bool pass = true;
    std::vector<OpX*>::const_iterator dstIt;
    for (dstIt = dstOps.begin(); dstIt != dstOps.end(); dstIt++)
      if (pass) {
        OpX* dstOp = *dstIt;
        pass = (pass & create_new_operator(dstOp, dstOp->mapOp));
      }
    if (!pass) return;
    // Check that output tensors with external edges are mapped
    for (const auto& opIt : mappedOps) {
      const auto& list = graph->outEdges[opIt.first];
      for (const auto& e : list)
        if (mappedOps.find(e.dstOp) == mappedOps.end()) {
          // dstOp is external, (srcOp, srcIdx) must be in mappedOutputs
          TensorX srcTen;
          srcTen.op = opIt.second;
          srcTen.idx = e.srcIdx;
          if (mappedOutputs.find(srcTen) == mappedOutputs.end()) {
            pass = false;
            return;
          }
        }
    }
    // Generate a new graph by applying xfer rule
    Graph* newGraph = create_new_graph(graph);
    // Check that the new graph should not have any loop
    if (newGraph->has_loop()) {
      printf("Found a new graph with LOOP!!!!\n");
      newGraph->print();
      delete newGraph;
      return;
    }
    // TODO: remove me for better performance
    assert(newGraph->check_correctness());
    if (newGraph->optimal_cost() < threshold && (int)newGraph->inEdges.size() < maxNumOps) {
      if (hashmap.find(newGraph->hash()) == hashmap.end()) {
        hashmap.insert(newGraph->hash());
        candidates.push(newGraph);
      }
    } else {
      delete newGraph;
    }
  } else {
    OpX* srcOp = srcOps[depth];
    for (const auto& it : graph->inEdges) {
      //printf("can_match(%d)\n", can_match(srcOp, it->first, graph));
      if (can_match(srcOp, it.first, graph)
      && (mappedOps.find(it.first) == mappedOps.end())) {
        Node op = it.first;
        // Check mapOutput
        match(srcOp, op, graph);
        run(depth + 1, graph, candidates, hashmap, threshold, maxNumOps);
        unmatch(srcOp, op, graph);
      }
    }
  }
}

Graph* GraphXfer::create_new_graph(Graph* graph)
{
  Graph* newGraph = new Graph(model);
  // Step 1: map dst ops
  std::vector<OpX*>::const_iterator dstIt;
  // Step 2: add edges to the graph
  for (const auto& opIt : graph->inEdges)
    if (mappedOps.find(opIt.first) == mappedOps.end()) {
      // Unmapped ops
      const auto& list = opIt.second;
      for (const auto& it : list)
        if (mappedOps.find(it.srcOp) != mappedOps.end()) {
          // mapped src -> unmapped dst
          TensorX srcTen;
          srcTen.op = mappedOps[it.srcOp];
          srcTen.idx = it.srcIdx;
          assert(mappedOutputs.find(srcTen) != mappedOutputs.end());
          TensorX dstTen = mappedOutputs[srcTen];
          newGraph->add_edge(dstTen.op->mapOp, it.dstOp, dstTen.idx, it.dstIdx);
        } else {
          // unmapped src -> unmmaped dst
          newGraph->add_edge(it.srcOp, it.dstOp, it.srcIdx, it.dstIdx);
        }
    }
  // Step 3: add edges for mapped ops
  for (dstIt = dstOps.begin(); dstIt != dstOps.end(); dstIt ++) {
    OpX* dstOp = *dstIt;
    for (size_t i = 0; i < dstOp->inputs.size(); i++)
      if (dstOp->inputs[i].op == NULL) {
        // unmapped src -> mapped dst
        std::multimap<int, std::pair<Node, int> >::const_iterator it
            = mappedInputs.find(dstOp->inputs[i].idx);
        assert(it != mappedInputs.end());
        const std::pair<Node, int>& srcEdge = it->second;
        newGraph->add_edge(srcEdge.first, dstOp->mapOp, srcEdge.second, i);
      } else {
        // mapped src -> mapped dst
        OpX* srcOp = dstOp->inputs[i].op;
        int srcIdx = dstOp->inputs[i].idx;
        newGraph->add_edge(srcOp->mapOp, dstOp->mapOp, srcIdx, i);
      }
  }
  // Simplify the graph by eliminating reverse parallel ops
  // and fusing multiple parallel ops
  // old graph: e1->n1->e2->n2->en
  // new graph: e1->new_node->en
  // TODO: temporarily disabled graph simplification
  bool simplify = true;
  while (simplify) {
    simplify = false;
    for (const auto& it : newGraph->inEdges) {
      if (it.first.ptr == NULL) continue;
      if (it.first.ptr->is_parallel_op()) {
        Node n2 = it.first;
        assert(it.second.size() == 1);
        Edge e2 = *it.second.begin();
        Node n1 = e2.srcOp;
        // Check that n1 is a parallel op
        // Check that n1 must have a single out edge
        if (n1.ptr->is_parallel_op() && newGraph->outEdges.find(n1)->second.size() == 1) {
          // merge n1 and n2
          std::vector<ParallelOpInfo> parallel_ops;
          ((ParallelOp*)n1.ptr)->append_parallel_op_info(parallel_ops);
          ((ParallelOp*)n2.ptr)->append_parallel_op_info(parallel_ops);
          Node new_node = model->get_or_create_fused_parallel_node(n1.ptr->inputs[0], parallel_ops);
          const auto& inList = newGraph->inEdges.find(n1)->second;
          assert(inList.size() == 1);
          Edge e1 = *inList.begin();
          // Update graph by adding edges
          newGraph->add_edge(e1.srcOp, new_node, e1.srcIdx, 0);
          newGraph->remove_edge(e1);
          newGraph->remove_edge(e2);
          // make a copy of outList
          if (newGraph->outEdges.find(n2) != newGraph->outEdges.end()) {
            const auto outList = newGraph->outEdges.find(n2)->second;
            for (const auto& e : outList) {
              newGraph->add_edge(new_node, e.dstOp, 0, e.dstIdx);
              newGraph->remove_edge(e);
            }
          }
          simplify = true;
        }
      }
      if (simplify) break;
    }
  }
  // Remove final parallel ops
  std::vector<Node> candidates;
  for (const auto& it : newGraph->outEdges) {
    if (it.second.size() == 0 && it.first.ptr->is_parallel_op()) {
      candidates.push_back(it.first);
    }
  }
  size_t index = 0;
  while (index < candidates.size()) {
    Node parallel_op = candidates[index++];
    const auto& inList = newGraph->inEdges.find(parallel_op)->second;
    assert(inList.size() == 1);
    Edge e = *inList.begin();
    newGraph->remove_edge(e);
    if (newGraph->outEdges.find(e.srcOp)->second.size() == 0 && e.srcOp.ptr->is_parallel_op()) {
      candidates.push_back(e.srcOp);
    }
  }
  // Remove NoOps
  std::vector<Node> noop_nodes;
  for (const auto& it : newGraph->inEdges) {
    if (it.first.ptr == NULL) continue;
    if (it.first.ptr->op_type == OP_NOOP) {
      noop_nodes.push_back(it.first);
    }
  }
  index = 0;
  while (index < noop_nodes.size()) {
    Node noop = noop_nodes[index++];
    const auto& inList = newGraph->inEdges.find(noop)->second;
    assert(inList.size() == 1);
    Edge in_edge = *inList.begin();
    // make a copy of outList
    if (newGraph->outEdges.find(noop) != newGraph->outEdges.end()) {
      const auto outList = newGraph->outEdges.find(noop)->second;
      for (const auto& e : outList) {
        newGraph->add_edge(in_edge.srcOp, e.dstOp, in_edge.srcIdx, e.dstIdx);
        newGraph->remove_edge(e);
      }
    }
    newGraph->remove_edge(in_edge);
  }
  return newGraph;
}

bool GraphXfer::create_new_operator(const OpX* opx, Node& op)
{
  Tensor inputs[MAX_NUM_INPUTS];
  for (size_t i = 0; i < opx->inputs.size(); i++)
    inputs[i] = opx->inputs[i].to_tensor(this);
  // Check that the total degree of inputs[0] does not exceed available resources
  if (opx->inputs.size() > 0) {
    int degree = 1;
    for (int i = 0; i < inputs[0]->num_dims; i++)
      degree *= inputs[0]->dims[i].degree;
    if (degree > model->config.workersPerNode * model->config.numNodes
    && (degree > model->config.cpusPerNode * model->config.numNodes))
      return false;
  }
  switch (opx->type) {
    case OP_NOOP:
    {
      op = model->get_or_create_noop_node(inputs[0]);
      break;
    }
    case OP_CONCAT:
    {
      int axis;
      assert(opx->get_pm_constraint(PM_AXIS, axis));
      op = model->get_or_create_concat_node(opx->inputs.size(), inputs, axis);
      break;
    }
    case OP_EW_ADD:
    {
      op = model->get_or_create_element_binary_node(inputs[0], inputs[1], opx->type);
      break;
    }
    case OP_CONV2D:
    {
      Conv2D* conv = (Conv2D*)opx->matchOpX->mapOp.ptr;
      Conv2DParams params = conv->get_params();
      op = model->get_or_create_conv2d_node(inputs[0], params);
      break;
    }
    case OP_POOL2D:
    {
      Pool2D* pool = (Pool2D*)opx->matchOpX->mapOp.ptr;
      Pool2DParams params = pool->get_params();
      op = model->get_or_create_pool2d_node(inputs[0], params);
      break;
    }
    case OP_FLAT:
    {
      op = model->get_or_create_flat_node(inputs[0]);
      break;
    }
    case OP_LINEAR:
    {
      int activation;
      assert(opx->matchOpX != NULL);
      assert(opx->matchOpX->mapOp.ptr != NULL);
      Linear* linear = (Linear*)opx->matchOpX->mapOp.ptr;
      //assert(opx->get_pm_constraint(PM_OUTPUT_CHANNELS, output_channels));
      assert(opx->get_pm_constraint(PM_ACTI, activation));
      op = model->get_or_create_linear_node(inputs[0], linear->out_channels,
                                            (ActiMode)activation, false);
      break;
    }
    case OP_MULTIHEAD_ATTENTION:
    {
      int num_heads;
      assert(opx->matchOpX != NULL);
      assert(opx->matchOpX->mapOp.ptr != NULL);
      MultiHeadAttention* attn = (MultiHeadAttention*) opx->matchOpX->mapOp.ptr;
      assert(opx->get_pm_constraint(PM_NUM_HEADS, num_heads));
      op = model->get_or_create_multihead_attn_node(inputs[0], inputs[1], inputs[2],
                                                    attn->oProjSize, num_heads,
                                                    attn->qProjSize, attn->vProjSize,
                                                    attn->dropout, attn->bias,
                                                    attn->add_bias_kv, attn->add_zero_attn);
      break;
    }
    case OP_SOFTMAX:
    {
      int softmax_dim;
      assert(opx->get_pm_constraint(PM_SOFTMAX_DIM, softmax_dim));
      op = model->get_or_create_softmax_node(inputs[0], softmax_dim);
      break;
    }
    case OP_REPARTITION:
    {
      int repartition_dim, repartition_degree;
      assert(opx->get_pm_constraint(PM_REPARTITION_DIM, repartition_dim));
      assert(opx->get_pm_constraint(PM_REPARTITION_DEGREE, repartition_degree));
      op = model->get_or_create_repartition_node(inputs[0], repartition_dim,
                                                 repartition_degree);
      break;
    }
    case OP_REPLICATE:
    {
      int replicate_dim, replicate_degree;
      assert(opx->get_pm_constraint(PM_REPLICATE_DIM, replicate_dim));
      assert(opx->get_pm_constraint(PM_REPLICATE_DEGREE, replicate_degree));
      op = model->get_or_create_replicate_node(inputs[0], replicate_dim,
                                               replicate_degree);
      break;
    }
    case OP_REDUCTION:
    {
      int reduction_dim, reduction_degree;
      assert(opx->get_pm_constraint(PM_REDUCTION_DIM, reduction_dim));
      assert(opx->get_pm_constraint(PM_REDUCTION_DEGREE, reduction_degree));
      op = model->get_or_create_reduction_node(inputs[0], reduction_dim,
                                               reduction_degree);
      break;
    }
    case OP_COMBINE:
    {
      int combine_dim, combine_degree;
      assert(opx->get_pm_constraint(PM_COMBINE_DIM, combine_dim));
      assert(opx->get_pm_constraint(PM_COMBINE_DEGREE, combine_degree));
      op = model->get_or_create_combine_node(inputs[0], combine_dim,
                                             combine_degree);
      break;
    }
    default:
    {
      printf("opx->type = %d\n", opx->type);
      assert(false);
    }
  }
  // Check operator validness
  if (op == Node::INVALID_NODE)
    return false;
  // Check tnConstraints
  for (size_t i = 0; i < opx->tnConstraints.size(); i++) {
    TNConstraint tnc = opx->tnConstraints[i];
    int actValue = 0, expValue = 0;
    if (tnc.singlePara) {
      assert(op.ptr->get_tensor_parameter(tnc.para1, tnc.dim1, &actValue));
      expValue = tnc.value;
    } else {
      assert(op.ptr->get_tensor_parameter(tnc.para1, tnc.dim1, &actValue));
      assert(op.ptr->get_tensor_parameter(tnc.para2, tnc.dim2, &expValue));
    }
    switch (tnc.comp) {
      case COMPARE_EQ:
        if (actValue != expValue) return false;
        break;
      case COMPARE_NE:
        if (actValue == expValue) return false;
        break;
      case COMPARE_LT:
        if (actValue >= expValue) return false;
        break;
      case COMPARE_LE:
        if (actValue > expValue) return false;
        break;
      case COMPARE_GT:
        if (actValue <= expValue) return false;
        break;
      case COMPARE_GE:
        if (actValue < expValue) return false;
        break;
      default:
        assert(false);
    }
  }
  return true;
}

OpX* GraphXfer::create_noop(const TensorX& input)
{
  OpX* noop = new OpX(OP_NOOP, 1, 1, input);
  return noop;
}

OpX* GraphXfer::create_concat(const TensorX* inputs,
                              int num_inputs,
                              const OpX* _matchOpX,
                              int concat_dim)
{
  OpX* concat = new OpX(OP_CONCAT, num_inputs, 1/*outputs*/, inputs);
  concat->matchOpX = _matchOpX;
  concat->add_pm_constraint(COMPARE_EQ, PM_AXIS, concat_dim);
  return concat;
}

OpX* GraphXfer::create_element_binary(const TensorX& input1,
                                      const TensorX& input2,
                                      OperatorType op_type)
{
  OpX* eb = new OpX(op_type, 2/*numInputs*/, 1, input1, input2);
  return eb;
}

OpX* GraphXfer::create_linear(const TensorX& input,
                              const OpX* _matchOpX,
                              int num_dims,
                              ActiMode acti_mode,
                              bool use_bias)
{
  OpX* li = new OpX(OP_LINEAR, 1, 1, input);
  li->matchOpX = _matchOpX;
  //li->add_pm_constraint(COMPARE_EQ, PM_OUTPUT_CHANNELS, out_channels);
  li->add_pm_constraint(COMPARE_EQ, PM_ACTI, acti_mode);
  li->add_input_constraint(COMPARE_EQ, INPUT_0, DIM_ND, num_dims);
  return li;
}

OpX* GraphXfer::create_conv2d(const TensorX& input,
                              const OpX* matchOpX) 
{
  OpX* conv = new OpX(OP_CONV2D, 1, 1, input);
  conv->matchOpX = matchOpX;
  return conv;
}

OpX* GraphXfer::create_attention(const TensorX& query,
                                 const TensorX& key,
                                 const TensorX& value,
                                 const OpX* _matchOpX,
                                 int num_heads)
{
  OpX* attn = new OpX(OP_MULTIHEAD_ATTENTION, 3, 1, query, key, value);
  attn->matchOpX = _matchOpX;
  attn->add_pm_constraint(COMPARE_EQ, PM_NUM_HEADS, num_heads);
  attn->add_input_constraint(COMPARE_EQ, INPUT_0, DIM_ND, 4);
  attn->add_input_constraint(COMPARE_EQ, INPUT_1, DIM_ND, 4);
  attn->add_input_constraint(COMPARE_EQ, INPUT_2, DIM_ND, 4);
  return attn;
}

OpX* GraphXfer::create_softmax(const TensorX& input,
                               int softmax_dim)
{
  OpX* softmax = new OpX(OP_SOFTMAX, 1, 1, input);
  softmax->add_pm_constraint(COMPARE_EQ, PM_SOFTMAX_DIM, softmax_dim);
  return softmax;
}

OpX* GraphXfer::create_repartition(const TensorX& input,
                                   int repartition_dim,
                                   int num_parts)
{
  OpX* part = new OpX(OP_REPARTITION, 1, 1, input);
  part->add_pm_constraint(COMPARE_EQ, PM_REPARTITION_DIM, repartition_dim);
  part->add_pm_constraint(COMPARE_EQ, PM_REPARTITION_DEGREE, num_parts);
  return part;
}

OpX* GraphXfer::create_replicate(const TensorX& input,
                                 int replicate_dim,
                                 int num_parts)
{
  OpX* replicate = new OpX(OP_REPLICATE, 1, 1, input);
  replicate->add_pm_constraint(COMPARE_EQ, PM_REPLICATE_DIM, replicate_dim);
  replicate->add_pm_constraint(COMPARE_EQ, PM_REPLICATE_DEGREE, num_parts);
  return replicate;
}

OpX* GraphXfer::create_reduction(const TensorX& input,
                                 int reduction_dim,
                                 int num_parts)
{
  OpX* reduction = new OpX(OP_REDUCTION, 1, 1, input);
  reduction->add_pm_constraint(COMPARE_EQ, PM_REDUCTION_DIM, reduction_dim);
  reduction->add_pm_constraint(COMPARE_EQ, PM_REDUCTION_DEGREE, num_parts);
  return reduction;
}

OpX* GraphXfer::create_combine(const TensorX& input,
                               int combine_dim,
                               int num_parts)
{
  OpX* part = new OpX(OP_COMBINE, 1, 1, input);
  part->add_pm_constraint(COMPARE_EQ, PM_COMBINE_DIM, combine_dim);
  part->add_pm_constraint(COMPARE_EQ, PM_COMBINE_DEGREE, num_parts);
  return part;
}

/* std::vector<Device> MachineView::get_devices() const { */
/*   std::vector<Device> devices; */


/* } */

void Graph::export_strategy_computation_graph(std::unordered_map<Node, MachineView> const &strategy, std::string const &out_filename) const {
  DotFile<Node> dot(out_filename);

  this->export_strategy_computation_graph(strategy, dot);
}

void Graph::export_strategy_computation_graph(std::unordered_map<Node, MachineView> const &strategy, std::unique_ptr<std::ostream> out) const {
  DotFile<Node> dot(std::move(out));

  this->export_strategy_computation_graph(strategy, dot);
}

void Graph::export_strategy_computation_graph(std::unordered_map<Node, MachineView> const &strategy, DotFile<Node> &dot) const {
  using flexflow::graph::GraphStructure;

  GraphStructure<Graph> s;

  for (auto const &node : s.get_nodes(*this)) {
    if (strategy.find(node) == strategy.end()) {
      dot.add_node(node, {{"label", node.to_string()}});
    } else {
      RecordFormatter rf, meta_row, machine_view_row;
      MachineView mv = strategy.at(node);
      std::ostringstream oss;
      switch (node.ptr->op_type) {
        case OP_REPARTITION:
        {
          Repartition *rp = (Repartition*)node.ptr;
          meta_row << std::to_string(rp->repartition_dim) << std::to_string(rp->repartition_degree);
          break;
        }
        case OP_COMBINE:
        {
          Combine *c = (Combine*)node.ptr;
          meta_row << std::to_string(c->combine_dim) << std::to_string(c->combine_degree);
          break;
        }
        case OP_REPLICATE:
        {
          Replicate *r = (Replicate*)node.ptr;
          meta_row << std::to_string(r->replicate_dim) << std::to_string(r->replicate_degree);
          break;
        }
        case OP_REDUCTION:
        {
          Reduction *r = (Reduction*)node.ptr;
          meta_row << std::to_string(r->reduction_dim) << std::to_string(r->reduction_degree);
          break;
        }
        default:
        {
          if (mv.ndims == 0) {
            meta_row << "N/A";
          } else {
            for (int i = 0; i < mv.ndims; i++) {
              meta_row << std::to_string(mv.dim[i]);
            }
          }
        }
      }
      for (int device_id : mv.device_ids()) {
        machine_view_row << std::to_string(device_id);
      }
      rf << node.to_string() << std::to_string(node.guid) << meta_row << machine_view_row;
      dot.add_record_node(node, rf);
    }

    for (auto const &edge : s.get_incoming_edges(*this, node)) {
      dot.add_edge(s.get_src(*this, edge), s.get_dst(*this, edge));
    }
  }

  dot.close();
}

template <typename T>
void create_mapping_xfers(FFModel *model, int degree, std::vector<GraphXfer*> &xfers, tl::optional<std::unordered_set<int>> dims = tl::nullopt)
{
  std::vector<ParallelDimMappingRecord> records; 
  T::construct_output_mappings(records);
  std::unordered_map<int, ParallelDimMappingRecord> output_mappings;

  std::unordered_set<int> all_dims;
  for (ParallelDimMappingRecord const &record : records) {
    assert (record.input_idx == 0);
    assert (record.get_type() == MappingRecordType::INPUT_OUTPUT);
    assert (record.output_idx == 0);
    assert (record.operation.has_value());

    all_dims.insert(record.input_dim);
    output_mappings.insert({record.input_dim, record});
  }

  if (dims.has_value()) {
    all_dims = dims.value();
  } 

  for (int const input_dim : all_dims) {
    int output_dim = output_mappings.at(input_dim).output_dim;
    GraphXfer *subst = new GraphXfer(model); 
    TensorX input = subst->new_tensor();

    OpX* original_op = subst->create_opx<T>(input, NULL/*matchOpX*/);
    subst->srcOps.push_back(original_op);

    OpX *pre;
    switch (output_mappings.at(input_dim).operation.value()) {
      case MappingOperation::PARTITION: 
        pre = subst->create_repartition(input, input_dim, degree);
        break;
      case MappingOperation::REPLICATE:
        pre = subst->create_replicate(input, input_dim, degree);
        break;
    }
    subst->dstOps.push_back(pre);

    OpX* new_op = subst->create_opx<T>(pre->outputs[0], original_op/*matchOpX*/);
    subst->dstOps.push_back(new_op);

    OpX *post;
    switch (output_mappings.at(input_dim).operation.value()) {
      case MappingOperation::PARTITION: 
        post = subst->create_combine(new_op->outputs[0], output_dim, degree);
        break;
      case MappingOperation::REPLICATE:
        post = subst->create_reduction(new_op->outputs[0], output_dim, degree);
        break;
    }
    subst->dstOps.push_back(post);

    subst->map_output(original_op->outputs[0], post->outputs[0]);

    xfers.push_back(subst);
  }
}

void FFModel::graph_optimize(size_t budget,
                             bool only_data_parallel,
                             Graph*& best_graph,
                             std::unordered_map<Node, MachineView>& optimal_views)
{
  // Construct graph structure
  Graph* graph = new Graph(this);
  {
    std::unordered_map<const Op*, Node> op_to_node_map;
    for (size_t l = 0; l < layers.size(); l++) {
      const Op* dstOp = layers[l];
      Node dstNode;
      dstNode.ptr = dstOp;
      dstNode.guid = node_global_guid++;
      op_to_node_map[dstOp] = dstNode;
      for (int j = 0; j < dstOp->numInputs; j++) {
        const Op* srcOp = dstOp->inputs[j]->owner_op;
        assert(op_to_node_map.find(srcOp) != op_to_node_map.end());
        Node srcNode = op_to_node_map[srcOp];
        graph->add_edge(srcNode, dstNode, dstOp->inputs[j]->owner_idx, j);
      }
    }
  }
  // Construct graph substitutions
  std::vector<GraphXfer*> xfers;
  std::vector<int> all_parallel_degrees, single_node_parallel_degrees;
  for (int i = 2; i <= config.workersPerNode; i++)
    if (config.workersPerNode % i == 0) {
      single_node_parallel_degrees.push_back(i);
      all_parallel_degrees.push_back(i);
    }
  for (int i = 2; i <= config.numNodes; i++)
    if (config.numNodes % i == 0) 
      all_parallel_degrees.push_back(i * config.workersPerNode);
  for (const auto& it : single_node_parallel_degrees) {
    xfers.push_back(create_replicate_linear_combine(this, 3, it, AC_MODE_RELU, false));
    xfers.push_back(create_replicate_linear_combine(this, 3, it, AC_MODE_SIGMOID, false));
    xfers.push_back(create_replicate_linear_combine(this, 3, it, AC_MODE_NONE, false));
    if (16 % it == 0) {
      xfers.push_back(create_replicate_attention_reduce(this, 16/*num_heads*/, it));
    }
  }
  for (const int degree : all_parallel_degrees) {
    create_mapping_xfers<Conv2D>(this, degree, xfers);
    create_mapping_xfers<Pool2D>(this, degree, xfers);
    create_mapping_xfers<Flat>(this, degree, xfers);
  }
  for (const auto& it : all_parallel_degrees) {
    if (it != config.numNodes * config.workersPerNode) continue;
    xfers.push_back(create_partition_attention_combine(this, 16/*num_heads*/, it));
    xfers.push_back(create_partition_linear_combine(this, 3/*num_dims*/, it, AC_MODE_RELU, false));
    xfers.push_back(create_partition_linear_combine(this, 3/*num_dims*/, it, AC_MODE_SIGMOID, false));
    xfers.push_back(create_partition_linear_combine(this, 3/*num_dims*/, it, AC_MODE_NONE, false));
    xfers.push_back(create_partition_linear_combine(this, 4/*num_dims*/, it, AC_MODE_RELU, false));
    xfers.push_back(create_partition_linear_combine(this, 4/*num_dims*/, it, AC_MODE_SIGMOID, false));
    xfers.push_back(create_partition_linear_combine(this, 4/*num_dims*/, it, AC_MODE_NONE, false));
    xfers.push_back(create_partition_add_combine(this, 2/*parallel_dims*/, it/*num_parts*/));
    xfers.push_back(create_partition_add_combine(this, 1/*parallel_dims*/, it/*num_parts*/));
    xfers.push_back(create_partition_softmax_combine(this, 0/*softmax_dim*/, 1/*parallel_dims*/, it/*num_parts*/));
    {
      std::unordered_set<int> concat_num_inputs;
      for (size_t i = 0; i < layers.size(); i++)
        if (layers[i]->op_type == OP_CONCAT)
          concat_num_inputs.insert(layers[i]->numInputs);
      for (const auto& it2 : concat_num_inputs)
        xfers.push_back(create_partition_concat_combine(this, it2/*num_inputs*/, 0/*concat_dim*/, 1/*parallel_dims*/, it/*num_parts*/));
    }
  }

  auto started = std::chrono::high_resolution_clock::now();
  bool export_search_curve = !this->config.search_curve_file.empty();
  std::ofstream search_curve_stream;
  if (export_search_curve) {
    search_curve_stream.open(this->config.search_curve_file);
    search_curve_stream << "ms,iteration,best" << std::endl;
  }

  std::priority_queue<Graph*, std::vector<Graph*>, GraphCompare> candidates;
  std::unordered_set<size_t> hashmap;
  candidates.push(graph);
  hashmap.insert(graph->hash());
  best_graph = new Graph(*graph);
  float best_cost = best_graph->optimal_cost();
  optimal_views = best_graph->optimal_views();
  this->convert_graph_to_layers(best_graph, optimal_views);
  float best_sim_time = this->simulator->simulate_runtime(this, COMP_MODE_TRAINING, "simulated_thing.dot");
  printf("   First best time: %fms\n", best_sim_time);
  int iter = 0;
  if (export_search_curve) {
    auto done = std::chrono::high_resolution_clock::now();
    auto num_millis = std::chrono::duration_cast<std::chrono::milliseconds>(done - started).count();
    search_curve_stream << num_millis << ","
                        << iter << ","
                        << best_sim_time << std::endl;
  }
  int counter = 0;
  while (false && !candidates.empty()) {
    Graph *cur_graph = candidates.top();
    candidates.pop();
    iter++;
    if (cur_graph->optimal_cost() < best_graph->optimal_cost()) {
      delete best_graph;
      best_graph = cur_graph;
      best_cost = cur_graph->optimal_cost();
      optimal_views = best_graph->optimal_views();
      this->convert_graph_to_layers(best_graph, optimal_views);
      best_sim_time = this->simulator->simulate_runtime(this, COMP_MODE_TRAINING, "");
      printf("  New best time: %fms\n", best_sim_time);
    }     
    if (export_search_curve && (counter % this->config.search_curve_interval == 0)) {
      auto done = std::chrono::high_resolution_clock::now();
      auto num_millis = std::chrono::duration_cast<std::chrono::milliseconds>(done - started).count();
      search_curve_stream << num_millis << ","
                          << iter << ","
                          << best_sim_time << std::endl;
    }
    if (cur_graph->optimal_cost() > best_cost * 1.2) {
      break;
    }

    if (counter > config.search_budget)
      break;
    printf("    [%d] cur_cost(%.4lf) best_cost(%.4lf) candidates.size(%zu)\n",
           counter, cur_graph->optimal_cost(), best_cost, candidates.size());
    counter ++;
    for (size_t i = 0; i < xfers.size(); i++) {
      xfers[i]->run(0, cur_graph, candidates, hashmap, best_cost * 1.2, 1000);
      std::cout << "." << std::flush;
    }
    std::cout << std::endl;
    if (best_graph != cur_graph) {
      delete cur_graph;
    }
  }
  // Run DP
  printf("best_cost = %.4lf\n", best_cost);
  //best_graph->print();
  optimal_views = best_graph->optimal_views();
  // Export results
  if (!this->config.export_strategy_computation_graph_file.empty()) {
    best_graph->export_strategy_computation_graph(optimal_views, this->config.export_strategy_computation_graph_file);
  }
  printf("Optimal Views...\n");
  for (const auto& it : optimal_views) {
    printf("node[%zu]: type(%s) view(%d %d %d) ", it.first.guid,
           it.first.to_string().c_str(),
           it.second.ndims,
           it.second.dim[0],
           it.second.start_device_id);
    const auto& list = best_graph->inEdges.at(it.first);
    for (const auto& it2 : list) {
      Edge e = it2;
      printf(" inEdge(node(%zu) idx(%d))", e.srcOp.guid, e.srcIdx);
    }
    printf("\n");
  }
  std::abort();
}

bool FFModel::convert_graph_to_layers(const Graph* graph,
                                      const std::unordered_map<Node, MachineView>& optimal_views)
{
  std::unordered_map<Node, int> todos;
  std::unordered_map<Node, Op*> node_to_op;
  std::vector<Node> queue;
  for (const auto& it : graph->inEdges) {
    const auto& inList = it.second;
    if (inList.size() == 0) {
      queue.push_back(it.first);
    } else {
      todos[it.first] = (int)inList.size();
    }
  }
  size_t index = 0;
  /* for (Op *op : layers) { */
  /*   delete op; */
  /* } */
  layers.clear();
  while (index < queue.size()) {
    Node node = queue[index++];
    assert(node.ptr != NULL);
    const auto& inList = graph->inEdges.find(node)->second;
    Tensor inputs[MAX_NUM_INPUTS];
    int num_inputs = 0;
    for (const auto& e : inList) {
      inputs[e.dstIdx] = node_to_op[e.srcOp]->outputs[e.srcIdx];
      assert(e.dstIdx < (int)inList.size());
      num_inputs++;
    }
    Op* new_op = NULL;
    switch (node.ptr->op_type) {
      case OP_INPUT:
      {
        new_op = new NoOp(*this, OP_INPUT, node.ptr->outputs[0]);
        break;
      }
      case OP_CONCAT:
      {
        Concat* concat = (Concat*) node.ptr;
        new_op = new Concat(*this, (int)inList.size(), inputs, concat->axis, NULL);
        break;
      }
      case OP_EMBEDDING:
      {
        new_op = new Embedding(*this, *(Embedding*)node.ptr, inputs[0], true);
        break;
      }
      case OP_EW_ADD:
      {
        assert(inList.size() == 2);
        ElementBinary* eb = (ElementBinary*) node.ptr;
        new_op = new ElementBinary(*this, eb->op_type, inputs[0], inputs[1],
                                   eb->inplace_a, NULL);
        break;
      }
      case OP_POOL2D:
      {
        new_op = new Pool2D(*this, *(Pool2D*)node.ptr, inputs[0]);
        break;
      }
      case OP_CONV2D:
      {
        new_op = new Conv2D(*this, *(Conv2D*)node.ptr, inputs[0], true);
        break;
      }
      case OP_LINEAR:
      {
        new_op = new Linear(*this, *(Linear*)node.ptr, inputs[0], true);
        break;
      }
      case OP_MULTIHEAD_ATTENTION:
      {
        assert(inList.size() == 3);
        MultiHeadAttention* attn = (MultiHeadAttention*) node.ptr;
        // Create weight tensor
        Tensor kernel;
        {
          int num_dims = inputs[0]->num_dims;
          // Compute weight size
          int qParas = attn->qProjSize * attn->qSize;
          int kParas = attn->kProjSize * attn->kSize;
          int vParas = attn->vProjSize * attn->vSize;
          int oParas = attn->oProjSize * (attn->vProjSize > 0 ? attn->vProjSize : attn->vSize);
          ParallelDim dims[3];
          dims[0] = inputs[0]->dims[num_dims-2];
          dims[0].size = dims[0].degree;
          dims[1] = inputs[0]->dims[num_dims-1];
          dims[1].size = attn->num_heads;
          dims[2].size = qParas + kParas + vParas + oParas;
          int seed = std::rand();
          Initializer* initializer = new GlorotUniform(seed);
#ifdef FF_USE_NCCL
          ParameterSyncType comm_type = ParameterSyncType::NCCL;
#else
          ParameterSyncType comm_type = ParameterSyncType::PS;
#endif
          kernel = create_weight<3>(dims, DT_FLOAT, NULL/*owner_op*/,
                                    true/*create_grad*/, initializer,
                                    comm_type);
        }
        new_op = new MultiHeadAttention(*this, inputs[0], inputs[1], inputs[2], kernel,
                                        attn->oProjSize, attn->num_heads,
                                        attn->qProjSize, attn->vProjSize,
                                        attn->dropout, attn->bias,
                                        attn->add_bias_kv, attn->add_zero_attn, NULL);
        break;
      }
      case OP_SOFTMAX:
      {
        assert(inList.size() == 1);
        Softmax* softmax = (Softmax*) node.ptr;
        new_op = new Softmax(*this, inputs[0], softmax->dim, NULL);
        break;
      }
      case OP_COMBINE:
      {
        assert(inList.size() == 1);
        Combine* combine = (Combine*) node.ptr;
        new_op = new Combine(*this, inputs[0], combine->combine_dim,
                             combine->combine_degree);
        break;
      }
      case OP_REPARTITION:
      {
        assert(inList.size() == 1);
        Repartition* repart = (Repartition*) node.ptr;
        new_op = new Repartition(*this, inputs[0], repart->repartition_dim,
                                 repart->repartition_degree);
        break;
      }
      case OP_REPLICATE:
      {
        assert(inList.size() == 1);
        Replicate* replicate = (Replicate*) node.ptr;
        new_op = new Replicate(*this, inputs[0], replicate->replicate_dim,
                               replicate->replicate_degree);
        break;
      }
      case OP_REDUCTION:
      {
        assert(inList.size() == 1);
        Reduction* reduction = (Reduction*) node.ptr;
        new_op = new Reduction(*this, inputs[0], reduction->reduction_dim,
                               reduction->reduction_degree);
        break;
      }
      case OP_FUSED_PARALLEL:
      {
        assert(inList.size() == 1);
        FusedParallelOp* fused = (FusedParallelOp*) node.ptr;
        std::vector<ParallelOpInfo> parallel_ops;
        for (int i = 0; i < fused->num_parallel_ops; i++)
          parallel_ops.push_back(fused->parallel_ops[i]);
        new_op = new FusedParallelOp(*this, inputs[0], parallel_ops);
        break;
      }
      default:
      {
        new_op = node.ptr->materialize(*this, inputs, num_inputs);
        break;
      }
    }
    // Set machine view for the output tensors of this operator
    assert(optimal_views.find(node) != optimal_views.end());
    MachineView view = optimal_views.find(node)->second;
    for (int i = 0; i < new_op->numOutputs; i++) {
      new_op->outputs[i]->machine_view = view;
    }
    // Set machine view for the weight tensors of this operator
    for (int i = 0; i < new_op->numWeights; i++) {
      new_op->weights[i]->machine_view = view;
    }
    node_to_op[node] = new_op;
    layers.push_back(new_op);
    // Decrease the todos
    const auto& outList = graph->outEdges.find(node)->second;
    for (const auto& it : outList) {
      todos[it.dstOp] -= 1;
      if (todos[it.dstOp] == 0) {
        queue.push_back(it.dstOp);
      }
    }
  }
  assert(queue.size() == graph->inEdges.size());
  // Remove the final parallel operators
  while (layers[layers.size()-1]->is_parallel_op()) {
    Op* op = layers[layers.size()-1];
    if (op->op_type == OP_REDUCTION)
      break;
    if (op->op_type == OP_FUSED_PARALLEL) {
      FusedParallelOp* fused_op = (FusedParallelOp*) op;
      bool has_reduction = false;
      for (int i = 0; i < fused_op->num_parallel_ops; i++) {
        if (fused_op->parallel_ops[i].op_type == OP_REDUCTION)
          has_reduction = true;
      }
      if (has_reduction)
        break;
    }
    layers.pop_back();
  }
  return true;
}

template <>
OpX* GraphXfer::create_opx<Conv2D>(const TensorX& input, const OpX* matchOpX) {
  return this->create_conv2d(input, matchOpX);
}

template <>
OpX* GraphXfer::create_opx<Pool2D>(const TensorX& input, const OpX* matchOpX) {
  OpX* pool = new OpX(OP_POOL2D, 1, 1, input);
  pool->matchOpX = matchOpX;
  return pool;
}

template <>
OpX* GraphXfer::create_opx<Flat>(const TensorX& input, const OpX* matchOpX) {
  OpX* flat = new OpX(OP_FLAT, 1, 1, input);
  flat->matchOpX = matchOpX;
  return flat;
}

GraphXfer* create_partition_linear_combine(FFModel* model,
                                           int num_dims,
                                           int num_parts,
                                           ActiMode activation,
                                           bool use_bias)
{
  GraphXfer* subst = new GraphXfer(model);
  TensorX input = subst->new_tensor();
  OpX* linear1 = subst->create_linear(input, NULL/*matchOpX*/, num_dims,
                                      activation, use_bias);
  OpX* repartition = subst->create_repartition(input, num_dims-2, num_parts);
  OpX* linear2 = subst->create_linear(repartition->outputs[0], linear1/*matchOpX*/,
                                      num_dims, activation, use_bias);
  OpX* combine = subst->create_combine(linear2->outputs[0], num_dims-2, num_parts);
  subst->map_output(linear1->outputs[0], combine->outputs[0]);
  subst->srcOps.push_back(linear1);
  subst->dstOps.push_back(repartition);
  subst->dstOps.push_back(linear2);
  subst->dstOps.push_back(combine);
  return subst;
}

GraphXfer* create_partition_attention_combine(FFModel* model,
                                              int num_heads,
                                              int num_parts)
{
  GraphXfer* subst = new GraphXfer(model);
  TensorX input = subst->new_tensor();
  OpX* attn1 = subst->create_attention(input, input, input, NULL/*matchOpX*/, num_heads);
  OpX* repart = subst->create_repartition(input, 2, num_parts);
  OpX* attn2 = subst->create_attention(repart->outputs[0], repart->outputs[0], repart->outputs[0],
                                       attn1/*matchOpX*/, num_heads);
  OpX* combine = subst->create_combine(attn2->outputs[0], 2, num_parts);
  subst->map_output(attn1->outputs[0], combine->outputs[0]);
  subst->srcOps.push_back(attn1);
  subst->dstOps.push_back(repart);
  subst->dstOps.push_back(attn2);
  subst->dstOps.push_back(combine);
  return subst;
}

GraphXfer* create_replicate_attention_reduce(FFModel* model,
                                             int num_heads,
                                             int num_parts)
{
  assert(num_heads % num_parts == 0);
  GraphXfer* subst = new GraphXfer(model);
  TensorX input = subst->new_tensor();
  OpX* attn1 = subst->create_attention(input, input, input, NULL/*matchOpX*/, num_heads);
  OpX* repl = subst->create_replicate(input, 3, num_parts);
  OpX* attn2 = subst->create_attention(repl->outputs[0], repl->outputs[0], repl->outputs[0],
                                       attn1/*matchOpX*/, num_heads / num_parts);
  OpX* reduce = subst->create_reduction(attn2->outputs[0], 3, num_parts);
  subst->map_output(attn1->outputs[0], reduce->outputs[0]);
  subst->srcOps.push_back(attn1);
  subst->dstOps.push_back(repl);
  subst->dstOps.push_back(attn2);
  subst->dstOps.push_back(reduce);
  return subst;
}

GraphXfer* create_replicate_linear_combine(FFModel* model,
                                           int num_dims,
                                           int num_parts,
                                           ActiMode activation,
                                           bool use_bias)
{
  GraphXfer* subst = new GraphXfer(model);
  TensorX input = subst->new_tensor();
  OpX* linear1 = subst->create_linear(input, NULL/*matchOpX*/, num_dims,
                                      activation, use_bias);
  OpX* replicate = subst->create_replicate(input, num_dims-1, num_parts);
  OpX* linear2 = subst->create_linear(replicate->outputs[0], linear1/*matchOpX*/,
                                      num_dims, activation, use_bias);
  OpX* combine = subst->create_combine(linear2->outputs[0], 0, num_parts);
  subst->map_output(linear1->outputs[0], combine->outputs[0]);
  subst->srcOps.push_back(linear1);
  subst->dstOps.push_back(replicate);
  subst->dstOps.push_back(linear2);
  subst->dstOps.push_back(combine);
  return subst;
}

GraphXfer* eliminate_combine_partition(FFModel* model,
                                       int parallel_dim,
                                       int num_parts)
{
  GraphXfer* subst = new GraphXfer(model);
  TensorX input = subst->new_tensor();
  OpX* combine = subst->create_combine(input, parallel_dim, num_parts);
  OpX* repartition = subst->create_repartition(combine->outputs[0],
                                               parallel_dim, num_parts);
  OpX* noop = subst->create_noop(input);
  subst->map_output(repartition->outputs[0], noop->outputs[0]);
  subst->srcOps.push_back(combine);
  subst->srcOps.push_back(repartition);
  subst->dstOps.push_back(noop);
  return subst;
}

GraphXfer* create_partition_add_combine(FFModel* model,
                                        int parallel_dim,
                                        int num_parts)
{
  GraphXfer* subst = new GraphXfer(model);
  TensorX input1 = subst->new_tensor();
  TensorX input2 = subst->new_tensor();
  OpX* add1 = subst->create_element_binary(input1, input2, OP_EW_ADD);
  OpX* repartition1 = subst->create_repartition(input1, parallel_dim, num_parts);
  OpX* repartition2 = subst->create_repartition(input2, parallel_dim, num_parts);
  OpX* add2 = subst->create_element_binary(repartition1->outputs[0],
                                           repartition2->outputs[0],
                                           OP_EW_ADD);
  OpX* combine = subst->create_combine(add2->outputs[0], parallel_dim, num_parts);
  subst->map_output(add1->outputs[0], combine->outputs[0]);
  subst->srcOps.push_back(add1);
  subst->dstOps.push_back(repartition1);
  subst->dstOps.push_back(repartition2);
  subst->dstOps.push_back(add2);
  subst->dstOps.push_back(combine);
  return subst;
}

GraphXfer* create_partition_concat_combine(FFModel* model,
                                           int num_inputs,
                                           int concat_dim,
                                           int parallel_dim,
                                           int num_parts)
{
  GraphXfer* subst = new GraphXfer(model);
  assert(num_inputs <= MAX_NUM_INPUTS);
  TensorX inputs[MAX_NUM_INPUTS];
  for (int i = 0; i < num_inputs; i++)
    inputs[i] = subst->new_tensor();
  OpX* concat = subst->create_concat(inputs, num_inputs, NULL/*matchOpX*/, concat_dim);
  subst->srcOps.push_back(concat);
  TensorX new_inputs[MAX_NUM_INPUTS];
  for (int i = 0; i < num_inputs; i++) {
    OpX* repartition = subst->create_repartition(inputs[i], parallel_dim, num_parts);
    new_inputs[i] = repartition->outputs[0];
    subst->dstOps.push_back(repartition);
  }
  OpX* concat2 = subst->create_concat(new_inputs, num_inputs, concat/*matchOpX*/,
                                      concat_dim);
  subst->dstOps.push_back(concat2);
  OpX* combine = subst->create_combine(concat2->outputs[0], parallel_dim, num_parts);
  subst->dstOps.push_back(combine);
  subst->map_output(concat->outputs[0], combine->outputs[0]);
  return subst;
}

GraphXfer* create_partition_softmax_combine(FFModel* model,
                                            int softmax_dim,
                                            int parallel_dim,
                                            int num_parts)
{
  assert(parallel_dim != softmax_dim);
  GraphXfer* subst = new GraphXfer(model);
  TensorX input = subst->new_tensor();
  OpX* softmax1 = subst->create_softmax(input, softmax_dim);
  OpX* repartition = subst->create_repartition(input, parallel_dim, num_parts);
  OpX* softmax2 = subst->create_softmax(repartition->outputs[0], softmax_dim);
  OpX* combine = subst->create_combine(softmax2->outputs[0], parallel_dim, num_parts);
  subst->map_output(softmax1->outputs[0], combine->outputs[0]);
  subst->srcOps.push_back(softmax1);
  subst->dstOps.push_back(repartition);
  subst->dstOps.push_back(softmax2);
  subst->dstOps.push_back(combine);
  return subst;
}
