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
using namespace Legion;

const TensorX TensorX::NO_TX = TensorX();

GraphXfer* create_partition_linear_combine(FFModel* model,
                                           int num_dims,
                                           int out_channels,
                                           int num_parts,
                                           ActiMode activation,
                                           bool use_bias);

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
    assert(op->mapOp != NULL);
    return op->mapOp->outputs[idx];
  } else {
    const auto& it = xfer->mappedInputs.find(idx);
    assert(it != xfer->mappedInputs.end());
    const Op* op = it->second.first;
    int outIdx = it->second.second;
    return op->outputs[outIdx];
  }
}

OpX::OpX(const OperatorType _type,
         int num_inputs,
         int num_outputs,
         const TensorX& input0,
         const TensorX& input1,
         const TensorX& input2,
         const TensorX& input3)
: type(_type)
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

bool GraphXfer::can_match(OpX* srcOp, const Op* op, Graph* graph)
{
  if (srcOp->type != op->op_type) return false;
  // check num input tensors
  if ((int)srcOp->inputs.size() != op->numInputs) return false;
  // check pmConstraints
  for (size_t i = 0; i < srcOp->pmConstraints.size(); i++) {
    PMConstraint pmc = srcOp->pmConstraints[i];
    int actValue = 0;
    assert(op->get_int_parameter(pmc.para, &actValue));
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
  std::map<int, std::pair<const Op*, int> > newMapInputs;
  for (size_t i = 0; i < srcOp->inputs.size(); i++) {
    TensorX in = srcOp->inputs[i];
    if (in.op == NULL) {
      // input tensor
      std::multimap<int, std::pair<const Op*, int> >::const_iterator it;
      it = mappedInputs.find(in.idx);
      if (it != mappedInputs.end()) {
        const Op* mappedOp = it->second.first;
        int mappedIdx = it->second.second;
        if (!(graph->has_edge(mappedOp, op, mappedIdx, i, false/*weight*/)))
          return false;
      } else {
        std::map<int, std::pair<const Op*, int> >::const_iterator newit;
        newit = newMapInputs.find(in.idx);
        if (newit != newMapInputs.end()) {
          const Op* mappedOp = newit->second.first;
          int mappedIdx = newit->second.second;
          if (!(graph->has_edge(mappedOp, op, mappedIdx, i, false/*weight*/)))
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
      assert(in.op->mapOp != NULL);
      if (!(graph->has_edge(in.op->mapOp, op, in.idx, i, false/*weight*/)))
        return false;
    }
  }
  // check weights
  for (size_t i = 0; i < srcOp->weights.size(); i++) {
    TensorX weight = srcOp->weights[i];
    if (weight.op == NULL) {
      // weight tensor
      std::multimap<int, std::pair<const Op*, int> >::const_iterator it;
      it = mappedInputs.find(weight.idx);
      if (it != mappedInputs.end()) {
        const Op* mappedOp = it->second.first;
        int mappedIdx = it->second.second;
        if (!(graph->has_edge(mappedOp, op, mappedIdx, i, true/*weight*/)))
          return false;
      } else {
        std::map<int, std::pair<const Op*, int> >::const_iterator newit;
        newit = newMapInputs.find(weight.idx);
        if (newit != newMapInputs.end()) {
          const Op* mappedOp = newit->second.first;
          int mappedIdx = newit->second.second;
          if (!(graph->has_edge(mappedOp, op, mappedIdx, i, true/*weight*/)))
            return false;
        } else {
          const auto& list = graph->inEdges.find(op)->second;
          for (const auto& e : list) {
            if (e.dstIdx == (int)i) {
              newMapInputs.insert(std::make_pair(weight.idx,
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
      assert(weight.op->mapOp != NULL);
      if (!(graph->has_edge(weight.op->mapOp, op, weight.idx, i, true/*weight*/)))
        return false;
    }
  }
  // check tnConstraints
  for (size_t i = 0; i < srcOp->tnConstraints.size(); i++) {
    TNConstraint tnc = srcOp->tnConstraints[i];
    int actValue = 0, expValue = 0;
    if (tnc.singlePara) {
      assert(op->get_tensor_parameter(tnc.para1, tnc.dim1, &actValue));
      expValue = tnc.value;
    } else {
      assert(op->get_tensor_parameter(tnc.para1, tnc.dim1, &actValue));
      assert(op->get_tensor_parameter(tnc.para2, tnc.dim2, &expValue));
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

void GraphXfer::match(OpX* srcOp, const Op* op, Graph* graph)
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

void GraphXfer::unmatch(OpX* srcOp, const Op* op, Graph* graph)
{
  for (size_t i = 0; i < srcOp->inputs.size(); i++) {
    TensorX in = srcOp->inputs[i];
    if (in.op == NULL) {
      // Update mappedInputsa
      std::multimap<int, std::pair<const Op*, int> >::iterator it;
      it = mappedInputs.find(in.idx);
      mappedInputs.erase(it);
    }
  }
  // Unmap op
  mappedOps.erase(op);
  srcOp->mapOp = NULL;
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
    if (newGraph->total_cost() < threshold && (int)newGraph->inEdges.size() < maxNumOps) {
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
        const Op* op = it.first;
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
          newGraph->add_edge(dstTen.op->mapOp, it.dstOp, dstTen.idx, it.dstIdx, it.weightEdge);
        } else {
          // unmapped src -> unmmaped dst
          newGraph->add_edge(it.srcOp, it.dstOp, it.srcIdx, it.dstIdx, it.weightEdge);
        }
    }
  // Step 3: add edges for mapped ops
  for (dstIt = dstOps.begin(); dstIt != dstOps.end(); dstIt ++) {
    OpX* dstOp = *dstIt;
    // inputs
    for (size_t i = 0; i < dstOp->inputs.size(); i++)
      if (dstOp->inputs[i].op == NULL) {
        // unmapped src -> mapped dst
        std::multimap<int, std::pair<const Op*, int> >::const_iterator it
            = mappedInputs.find(dstOp->inputs[i].idx);
        assert(it != mappedInputs.end());
        std::pair<const Op*, int> srcEdge = it->second;
        newGraph->add_edge(srcEdge.first, dstOp->mapOp, srcEdge.second, i, false);
      } else {
        // mapped src -> mapped dst
        OpX* srcOp = dstOp->inputs[i].op;
        int srcIdx = dstOp->inputs[i].idx;
        newGraph->add_edge(srcOp->mapOp, dstOp->mapOp, srcIdx, i, false);
      }
    // weights
    for (size_t i = 0; i < dstOp->weights.size(); i++)
      if (dstOp->weights[i].op == NULL) {
        // unmapped src -> mapped dst
        std::multimap<int, std::pair<const Op*, int> >::const_iterator it
            = mappedInputs.find(dstOp->weights[i].idx);
        assert(it != mappedInputs.end());
        std::pair<const Op*, int> srcEdge = it->second;
        newGraph->add_edge(srcEdge.first, dstOp->mapOp, srcEdge.second, i, true);
      } else {
        // mapped src -> mapped dst
        OpX* srcOp = dstOp->weights[i].op;
        int srcIdx = dstOp->weights[i].idx;
        newGraph->add_edge(srcOp->mapOp, dstOp->mapOp, srcIdx, i, true);
      }
  }
  return newGraph;
}

bool GraphXfer::create_new_operator(const OpX* opx, const Op*& op)
{
  Tensor inputs[MAX_NUM_INPUTS];
  for (size_t i = 0; i < opx->inputs.size(); i++)
    inputs[i] = opx->inputs[i].to_tensor(this);
  switch (opx->type) {
    case OP_LINEAR:
    {
      int output_channels, activation;
      assert(opx->get_pm_constraint(PM_OUTPUT_CHANNELS, output_channels));
      assert(opx->get_pm_constraint(PM_ACTI, activation));
      op = model->get_or_create_linear(inputs[0], output_channels,
                                       (ActiMode)activation, false);
      break;
    }
    case OP_REPARTITION:
    {
      int repartition_dim, repartition_degree;
      assert(opx->get_pm_constraint(PM_REPARTITION_DIM, repartition_dim));
      assert(opx->get_pm_constraint(PM_NUM_PARTITIONS, repartition_degree));
      op = model->get_or_create_repartition(inputs[0], repartition_dim,
                                            repartition_degree);
      break;
    }
    case OP_COMBINE:
    {
      int combine_dim, combine_degree;
      assert(opx->get_pm_constraint(PM_COMBINE_DIM, combine_dim));
      assert(opx->get_pm_constraint(PM_NUM_PARTITIONS, combine_degree));
      op = model->get_or_create_combine(inputs[0], combine_dim,
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
  if (op == NULL)
    return false;
  // Check tnConstraints
  for (size_t i = 0; i < opx->tnConstraints.size(); i++) {
    TNConstraint tnc = opx->tnConstraints[i];
    int actValue = 0, expValue = 0;
    if (tnc.singlePara) {
      assert(op->get_tensor_parameter(tnc.para1, tnc.dim1, &actValue));
      expValue = tnc.value;
    } else {
      assert(op->get_tensor_parameter(tnc.para1, tnc.dim1, &actValue));
      assert(op->get_tensor_parameter(tnc.para2, tnc.dim2, &expValue));
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

OpX* GraphXfer::create_linear(const TensorX& input,
                              int num_dims,
                              int out_channels,
                              ActiMode acti_mode,
                              bool use_bias)
{
  OpX* li = new OpX(OP_LINEAR, 1, 1, input);
  li->add_pm_constraint(COMPARE_EQ, PM_OUTPUT_CHANNELS, out_channels);
  li->add_pm_constraint(COMPARE_EQ, PM_ACTI, acti_mode);
  li->add_input_constraint(COMPARE_EQ, INPUT_0, DIM_ND, num_dims);
  return li;
}

OpX* GraphXfer::create_repartition(const TensorX& input,
                                   int repartition_dim,
                                   int num_parts)
{
  OpX* part = new OpX(OP_REPARTITION, 1, 1, input);
  part->add_pm_constraint(COMPARE_EQ, PM_REPARTITION_DIM, repartition_dim);
  part->add_pm_constraint(COMPARE_EQ, PM_NUM_PARTITIONS, num_parts);
  return part;
}

OpX* GraphXfer::create_combine(const TensorX& input,
                               int combine_dim,
                               int num_parts)
{
  OpX* part = new OpX(OP_COMBINE, 1, 1, input);
  part->add_pm_constraint(COMPARE_EQ, PM_COMBINE_DIM, combine_dim);
  part->add_pm_constraint(COMPARE_EQ, PM_NUM_PARTITIONS, num_parts);
  return part;
}

void FFModel::dp_optimize()
{
  // Construct graph structure
  Graph* graph = new Graph(this);
  for (size_t l = 0; l < layers.size(); l++) {
    const Op* op = layers[l];
    for (int j = 0; j < op->numInputs; j++) {
      graph->add_edge(op->inputs[j]->owner_op, op, op->inputs[j]->owner_idx, j,
                      false/*weight*/);
    }
  }
  // Construct graph substitutions
  std::vector<GraphXfer*> xfers;
  xfers.push_back(create_partition_linear_combine(this, 3, 4096, 4, AC_MODE_RELU, false));
  xfers.push_back(create_partition_linear_combine(this, 3, 4096, 4, AC_MODE_NONE, false));

  std::priority_queue<Graph*, std::vector<Graph*>, GraphCompare> candidates;
  std::unordered_set<size_t> hashmap;
  candidates.push(graph);
  hashmap.insert(graph->hash());
  Graph* best_graph = graph;
  float best_cost = graph->total_cost();
  int counter = 0;
  while (!candidates.empty()) {
    Graph *cur_graph = candidates.top();
    candidates.pop();
    if (cur_graph->total_cost() < best_cost) {
      delete best_graph;
      best_graph = cur_graph;
      best_cost = cur_graph->total_cost();
    }
    printf("    [%d] cur_cost(%.4lf) best_cost(%.4lf) candidates.size(%zu)\n",
           counter, cur_graph->total_cost(), best_cost, candidates.size());
    counter ++;
    for (size_t i = 0; i < xfers.size(); i++) {
      xfers[i]->run(0, cur_graph, candidates, hashmap, best_cost * 1.05, 100000);
    }
    if (best_graph != cur_graph) {
      delete cur_graph;
    }
  }
  // Run DP
  printf("best_cost = %.4lf\n", best_cost);
  best_graph->print();
}

GraphXfer* create_partition_linear_combine(FFModel* model,
                                           int num_dims,
                                           int out_channels,
                                           int num_parts,
                                           ActiMode activation,
                                           bool use_bias)
{
  GraphXfer* subst = new GraphXfer(model);
  TensorX input = subst->new_tensor();
  OpX* linear1 = subst->create_linear(input, num_dims, out_channels,
                                      activation, use_bias);
  OpX* repartition = subst->create_repartition(input, num_dims-2, num_parts);
  OpX* linear2 = subst->create_linear(repartition->outputs[0], num_dims,
                                      out_channels, activation, use_bias);
  OpX* combine = subst->create_combine(linear2->outputs[0], num_dims-2, num_parts);
  subst->map_output(linear1->outputs[0], combine->outputs[0]);
  subst->srcOps.push_back(linear1);
  subst->dstOps.push_back(repartition);
  subst->dstOps.push_back(linear2);
  subst->dstOps.push_back(combine);
  return subst;
}

