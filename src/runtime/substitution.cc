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

#include <iomanip>
#include "flexflow/substitution.h"
#include <chrono>
#include "flexflow/utils/dot_file.h"
#include "flexflow/dominators.h"
#include "flexflow/ops/embedding.h"
#include "flexflow/ops/linear.h"
#include "flexflow/ops/conv_2d.h"
#include "flexflow/ops/dropout.h"
#include "flexflow/ops/pool_2d.h"
#include "flexflow/ops/attention.h"
#include "flexflow/ops/flat.h"
#include "flexflow/ops/element_binary.h"
#include "flexflow/ops/split.h"
#include "flexflow/ops/noop.h"
#include "flexflow/ops/softmax.h"
#include "flexflow/ops/concat.h"
#include "flexflow/parallel_ops/combine.h"
#include "flexflow/parallel_ops/partition.h"
#include "flexflow/parallel_ops/replicate.h"
#include "flexflow/parallel_ops/fused_parallel_op.h"
#include "flexflow/parallel_ops/reduction.h"
#include "flexflow/graph.h"

namespace FlexFlow::PCG {

using namespace Legion;

LegionRuntime::Logger::Category log_xfers("xfers");
LegionRuntime::Logger::Category log_xfer_matches("xfer_matches");

const TensorX TensorX::NO_TX = TensorX();

GraphXfer* create_combine_inception(FFModel* model,
                                    int num_convs,
                                    int num_dims,
                                    int num_parts);

GraphXfer* create_combine_concat(FFModel* model,
                                 int num_inputs,
                                 int num_dims,
                                 int num_parts);

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

GraphXfer* create_partition_conv2d_combine(FFModel* model,
                                           int num_dims,
                                           int num_parts);

GraphXfer* create_partition_attention_combine(FFModel* model,
                                              int num_heads,
                                              int num_parts);

GraphXfer* create_replicate_attention_reduce(FFModel* model,
                                             int num_heads,
                                             int num_parts);

GraphXfer* create_partition_add_combine(FFModel* model,
                                        int parallel_dim,
                                        int num_parts);
GraphXfer* create_partition_relu_combine(FFModel* model,
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
GraphXfer* leading_relu_branch_combine(FFModel* model, 
                                       int parallel_dim, 
                                       int num_parts, 
                                       int num_combines);
GraphXfer* leading_relu_branch_partition(FFModel* model, 
                                        int parallel_dim, 
                                        int num_parts, 
                                        int num_partitions);
GraphXfer* create_linear_relu_merge(FFModel* model, int num_dims, bool use_bias);

PMConstraint::PMConstraint(Compare c, PMParameter p, int v)
: comp(c), para(p), value(v) {}

TNConstraint::TNConstraint(Compare c, TNParameter p, DIMParameter d, int v)
: singlePara(true), comp(c), para1(p), dim1(d), value(v) {}

TNConstraint::TNConstraint(Compare c, TNParameter p1, DIMParameter d1,
                           TNParameter p2, DIMParameter d2)
: singlePara(false), comp(c), para1(p1), para2(p2), dim1(d1), dim2(d2) {}

tl::optional<ParallelTensor> TensorX::to_tensor(const GraphXfer* xfer) const
{
  if (op != NULL) {
    assert(op->mapOp.ptr != NULL);
    return op->mapOp.ptr->outputs[idx];
  } else {
    const auto& it = xfer->mappedInputs.find(idx);
    if (it == xfer->mappedInputs.end()) {
      return tl::nullopt;
    }
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

bool GraphXfer::can_match(OpX* srcOp, const Node& op, Graph const *graph)
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

void GraphXfer::match(OpX* srcOp, const Node& op, Graph const *graph) 
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

void GraphXfer::unmatch(OpX* srcOp, const Node& op, Graph const *graph)
{
  for (size_t i = 0; i < srcOp->inputs.size(); i++) {
    log_xfer_matches.spew() << "umatch iteration " << i;
    TensorX in = srcOp->inputs[i];
    if (in.op == NULL) {
      // Update mappedInputsa
      std::multimap<int, std::pair<Node, int> >::iterator it;
      log_xfer_matches.spew() << "Starting find";
      it = mappedInputs.find(in.idx);
      log_xfer_matches.spew() << "Finished find";
      if (it != mappedInputs.end()) {
        mappedInputs.erase(it);
      }
    }
  }
  log_xfer_matches.spew() << "Finished the unmatch loop";
  // Unmap op
  mappedOps.erase(op);
  srcOp->mapOp.guid = 0;
  srcOp->mapOp.ptr = NULL;
  log_xfer_matches.spew() << "Returning from unmatch";
}

GraphXferMatch::GraphXferMatch(GraphXfer const *xfer) 
  : xfer(xfer) 
{ }

void GraphXferMatch::add_mapping(Node const &node, OpX* opx) {
  this->nodeToOpX[node] = opx;
  this->opXToNode[opx] = node;
}

void GraphXferMatch::add_mapping(OpX* opx, Node const &node) {
  this->add_mapping(node, opx);
}

void GraphXferMatch::add_output_mapping(TensorX const &src, TensorX const &dst) {
  this->mappedOutputs[src] = dst;
}

OpX* GraphXferMatch::at(Node const &n) const {
  return this->nodeToOpX.at(n);
}

Node GraphXferMatch::at(OpX *opx) const {
  return this->opXToNode.at(opx);
}

void GraphXferMatch::set_graph(Graph const *g) {
  this->graph_hash = g->hash();
}

bool GraphXferMatch::containsNode(Graph const *g, Node const &n) const {
  assert (g->hash() == this->graph_hash);

  return this->nodeToOpX.find(n) != this->nodeToOpX.end();
}

bool GraphXferMatch::containsEdge(Graph const *g, Edge const &e) const {
  assert (g->hash() == this->graph_hash);

  bool contains_src = this->containsNode(g, e.srcOp);
  bool contains_dst = this->containsNode(g, e.dstOp);

  return contains_src && contains_dst;
}

GraphXfer const *GraphXferMatch::get_xfer() const {
  return this->xfer;
}

std::unordered_set<Node> GraphXferMatch::get_nodes() const {
  std::unordered_set<Node> nodes;
  for (auto const &kv : nodeToOpX) {
    nodes.insert(kv.first);
  }

  return nodes;
}

GraphXferMatch GraphXfer::get_match_record(Graph const *g) const {
  GraphXferMatch match(this);

  for (auto const &kv : this->mappedOps) {
    match.add_mapping(kv.first, kv.second);
  }

  for (auto const &kv : this->mappedOutputs) {
    match.add_output_mapping(kv.first, kv.second);
  }

  match.set_graph(g);

  return match;
}

void GraphXfer::find_matches(Graph const *graph, std::vector<GraphXferMatch>& matches) {
  this->find_matches(0, graph, matches);
}

void GraphXfer::find_matches(int depth, Graph const *graph, std::vector<GraphXferMatch>& matches) {
  log_xfer_matches.spew() << "find_matches at depth: " << depth;
  if (depth >= (int)srcOps.size()) {
    log_xfer_matches.spew() << "Achieved adequate depth";
    // Create dst operators
    bool pass = true;
    for (OpX *dstOp : this->dstOps) {
        pass &= create_new_operator(dstOp, dstOp->mapOp);
      if (!pass) {
        break;
      }
    }
    log_xfer_matches.spew() << "Completed create dst operators";
    if (!pass) {
      log_xfer_matches.spew() << "Did not pass. Returning.";
      return;
    }
    log_xfer_matches.spew() << "Checking external edges";
    // Check that output tensors with external edges are mapped
    for (const auto& opIt : mappedOps) {
      const auto& list = graph->outEdges.at(opIt.first);
      for (const auto& e : list) {
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
    }
    log_xfer_matches.spew() << "Completed checking external edges";
    // Generate a new graph by applying xfer rule
    log_xfer_matches.spew() << "Creating new graph";
    SimplificationSettings settings; // leave everything disabeld since we don't care about cost
    Graph* newGraph = this->create_new_graph(graph, settings);
    log_xfer_matches.spew() << "Completed creating new graph";

    // Check that the new graph should not have any loop
    log_xfer_matches.spew() << "Checking for loop";
    if (newGraph->has_loop()) {
      printf("Found a new graph with LOOP!!!!\n");
      newGraph->print();
      delete newGraph;
      return;
    }
    log_xfer_matches.spew() << "Finished checking for loop";
    // TODO: remove me for better performance
    log_xfer_matches.spew() << "Checking correctness";
    assert(newGraph->check_correctness());
    log_xfer_matches.spew() << "Finished checking correctness";
    log_xfer_matches.spew() << "Getting match record";
    GraphXferMatch match_record = this->get_match_record(graph);
    log_xfer_matches.spew() << "Finished getting match record";
    matches.push_back(match_record);
  } else {
    OpX* srcOp = srcOps[depth];
    for (const auto& it : graph->inEdges) {
      log_xfer_matches.spew() << "Exploring node " << it.first.to_string();
      //printf("can_match(%d)\n", can_match(srcOp, it->first, graph));
      if (can_match(srcOp, it.first, graph)
      && (mappedOps.find(it.first) == mappedOps.end())) {
        Node op = it.first;
        // Check mapOutput
        this->match(srcOp, op, graph);
        this->find_matches(depth + 1, graph, matches);
        log_xfer_matches.spew() << "Completed find matches. Unmatching";
        this->unmatch(srcOp, op, graph);
        log_xfer_matches.spew() << "Finished unmatching";
      }
    }
  }
}

void GraphXfer::run(int depth, Graph* graph,
                    std::priority_queue<Graph*, std::vector<Graph*>, GraphCompare>& candidates,
                    std::unordered_set<size_t>& hashmap, float threshold, int maxNumOps, 
                    SimplificationSettings const &simplification_settings,
                    int& num_matches_found, int& num_matches_rejected)
{
  //printf("run: depth(%d) srcOps.size(%zu) graph.size(%zu) candidates(%zu)\n", depth, srcOps.size(), graph->inEdges.size(), candidates.size());
  if (depth >= (int)srcOps.size()) {
    // Create dst operators
    bool pass = true;
    for (OpX *dstOp : this->dstOps) {
      if (pass) {
        pass &= create_new_operator(dstOp, dstOp->mapOp);
      }
    }
    if (!pass) return;
    // Check that output tensors with external edges are mapped
    for (const auto& opIt : mappedOps) {
      const auto& list = graph->outEdges[opIt.first];
      for (const auto& e : list) {
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
    }
    // Generate a new graph by applying xfer rule
    log_xfers.spew() << "Found a match for xfer: " << this->get_name();
    num_matches_found++;
    Graph* newGraph = this->create_new_graph(graph, simplification_settings);
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
        log_xfers.spew() << "Found new candidate";
        // newGraph->print_dot();
        candidates.push(newGraph);
      }
    } else {
      num_matches_rejected++;
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
        run(depth + 1, graph, candidates, hashmap, threshold, maxNumOps, simplification_settings, num_matches_found, num_matches_rejected);
        unmatch(srcOp, op, graph);
      }
    }
  }
}

Node Graph::find_source_node() const {
  using FlexFlow::PCG::Utils::roots;

  std::unordered_set<Node> source_nodes = roots(*this);
  assert (source_nodes.size() == 1);
  
  return *source_nodes.begin();
}

Node Graph::find_sink_node() const {
  using FlexFlow::PCG::Utils::leaves;

  std::unordered_set<Node> sink_nodes = leaves(*this);
  assert (sink_nodes.size() == 1);

  return *sink_nodes.begin();
}

void Graph::reshape_output_tensor(ParallelTensorShape const &desired_shape) {
  Node output_node = this->find_sink_node();

  assert (output_node.ptr->numOutputs == 1);
  ParallelTensor output_tensor = output_node.ptr->outputs[0];

  assert (output_tensor->num_dims == desired_shape.num_dims);

  for (int i = 0; i < output_tensor->num_dims; i++) {
    int current_size = output_tensor->dims[i].size;
    int current_degree = output_tensor->dims[i].degree;

    int desired_size = desired_shape.dims[i].size;
    int desired_degree = desired_shape.dims[i].degree;

    assert (current_size == desired_size);

    if (current_degree < desired_degree) {
      // we need to partition
      assert (desired_degree % current_degree == 0);
      int partition_factor = desired_degree / current_degree;

      Node partition_node = model->get_or_create_repartition_node(output_tensor, i, partition_factor);
      this->add_edge(output_node, partition_node, 0, 0);

      output_node = partition_node;
      output_tensor = partition_node.ptr->outputs[0];
      current_degree *= partition_factor;

    } else if (current_degree > desired_degree) {
      // we need to combine
      assert (current_degree % desired_degree == 0);
      int combine_factor = current_degree / desired_degree;

      Node combine_node = model->get_or_create_combine_node(output_tensor, i, combine_factor);
      this->add_edge(output_node, combine_node, 0, 0);

      output_node = combine_node;
      output_tensor = combine_node.ptr->outputs[0];
      current_degree /= combine_factor;
    }

    assert (current_degree == desired_degree);
  }

  assert (output_tensor == output_node.ptr->outputs[0]);
  assert (output_tensor->num_dims == desired_shape.num_dims);
  for (int i = 0; i < desired_shape.num_dims; i++) {
    assert (output_tensor->dims[i].size == desired_shape.dims[i].size);
    assert (output_tensor->dims[i].degree == desired_shape.dims[i].degree);
  }
}

std::unique_ptr<Graph> Graph::with_output_tensor_reshaped_to(ParallelTensorShape const &shape) const {
  auto g = std::unique_ptr<Graph>(new Graph(*this));
  g->reshape_output_tensor(shape);
  return g;
}

/* Graph::Graph(Graph const &graph) */
/*   : Graph(&graph) */
/* { } */

/* Graph::Graph(Graph const *graph) */ 
/*   : Graph(graph->model) */
/* { */
/*   for (auto const &kv : graph->inEdges) { */
/*     Node const &node = kv.first; */
/*     std::unordered_set<Edge> const &edge_set = kv.second; */
    
/*     for (auto const &edge : edge_set) { */
/*       this->add_edge(edge.srcOp, edge.dstOp, edge.srcIdx) */
/*     } */
/*   } */
/* } */


Graph* GraphXfer::create_new_graph(Graph const *graph, SimplificationSettings const &simplification_settings)
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
  newGraph->simplify(simplification_settings);

  return newGraph;
}

bool GraphXfer::create_new_operator(const OpX* opx, Node& op)
{
  ParallelTensor inputs[MAX_NUM_INPUTS];
  for (size_t i = 0; i < opx->inputs.size(); i++) {
    tl::optional<ParallelTensor> mapped = opx->inputs[i].to_tensor(this);
    if (!mapped.has_value()) {
      return false;
    }
    inputs[i] = mapped.value();
  }
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
    case OP_RELU:
    {
      op = model->get_or_create_element_unary_node(inputs[0], opx->type, false, 0.0f);
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
      op = model->get_or_create_linear_node(linear->layer_guid, inputs[0], linear->out_channels,
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

OpX* GraphXfer::create_element_unary(const TensorX& input,
                                     OperatorType op_type)
{
  OpX* eu = new OpX(op_type, 1/*numInputs*/, 1, input);
  return eu;
}

OpX* GraphXfer::create_relu(const TensorX& input) 
{
  return this->create_element_unary(input, OP_RELU);
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
  // TODO FIXME @lockshaw @zhihao use_bias is completely unused
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

OpX* GraphXfer::create_pool2d(const TensorX& input,
                              const OpX* matchOpX) {
  OpX* pool = new OpX(OP_POOL2D, 1, 1, input);
  pool->matchOpX = matchOpX;
  return pool;
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

void Graph::print_strategy_computation_graph(std::unordered_map<Node, MachineView> const &strategy) const {
  DotFile<Node> dot(std::cout);
  this->export_strategy_computation_graph(strategy, dot);
}

void Graph::export_strategy_computation_graph(std::unordered_map<Node, MachineView> const &strategy, std::string const &out_filename) const {
  DotFile<Node> dot(out_filename);

  this->export_strategy_computation_graph(strategy, dot);
}

void Graph::export_strategy_computation_graph(std::unordered_map<Node, MachineView> const &strategy, DotFile<Node> &dot) const {
  using FlexFlow::PCG::Utils::GraphStructure;

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

    std::ostringstream oss;
    std::string op_type_name = model->get_operator_type_name(new_op->type);
    std::transform(op_type_name.begin(), op_type_name.end(), op_type_name.begin(),
        [](unsigned char c) { return std::tolower(c); });
    oss << "partition_" << op_type_name << "_combine["
        << "input_dim=" << input_dim
        << ",degree=" << degree
        << "]";
    subst->name = oss.str();

    xfers.push_back(subst);
  }
}

std::string GraphXfer::get_name() const {
  if (this->name.has_value()) {
    return this->name.value();
  } else {
    std::ostringstream oss;
    oss << "unknown_xfer(" << this << ")";
    return oss.str();
  }
}

GraphSearchHelper::GraphSearchHelper(FFModel *model) 
  : model(model), config(model->config)
{ 
  this->logger = std::unique_ptr<RecursiveLogger>(new RecursiveLogger("gs"));
  generate_all_pcg_xfers();
}

void GraphSearchHelper::load_graph_substitutions(std::vector<GraphXfer*> &xfers) const
{
  xfers = all_pcg_xfers;
}

void GraphSearchHelper::generate_all_pcg_xfers()
{
  std::vector<int> all_parallel_degrees, single_node_parallel_degrees;
  auto const &config = this->model->config;
  int workersPerNode = config.search_num_workers.value_or(config.workersPerNode);
  int numNodes = config.search_num_nodes.value_or(config.numNodes);
  log_xfers.debug() << "Generating parallel degrees for workersPerNode " 
                    << workersPerNode
                    << " and numNodes "
                    << numNodes;
  for (int i = 2; i <= workersPerNode; i++) {
    if (workersPerNode % i == 0) {
      single_node_parallel_degrees.push_back(i);
      all_parallel_degrees.push_back(i);
    }
  }
  for (int i = 2; i <= numNodes; i++) {
    if (numNodes % i == 0) {
      all_parallel_degrees.push_back(i * workersPerNode);
    }
  }
  for (const auto& it : single_node_parallel_degrees) {
    all_pcg_xfers.push_back(create_replicate_linear_combine(this->model, 3, it, AC_MODE_RELU, false));
    all_pcg_xfers.push_back(create_replicate_linear_combine(this->model, 3, it, AC_MODE_SIGMOID, false));
    all_pcg_xfers.push_back(create_replicate_linear_combine(this->model, 3, it, AC_MODE_NONE, false));
    if (16 % it == 0) {
      all_pcg_xfers.push_back(create_replicate_attention_reduce(this->model, 16/*num_heads*/, it));
    }
  }

  {
    std::ostringstream oss;
    oss << "Generating all_pcg_xfers for all parallel degrees: ";
    for (int parallel_degree : all_parallel_degrees) { 
      oss << parallel_degree << " ";
    }

    log_xfers.debug() << oss.str();
  }

  for (int num_dims = 3; num_dims <=4; num_dims++) {
    all_pcg_xfers.push_back(create_linear_relu_merge(this->model, num_dims, true));
    all_pcg_xfers.push_back(create_linear_relu_merge(this->model, num_dims, false));
  }
  for (const int degree : all_parallel_degrees) {
    create_mapping_xfers<Conv2D>(this->model, degree, all_pcg_xfers);
    create_mapping_xfers<Pool2D>(this->model, degree, all_pcg_xfers);
    create_mapping_xfers<Flat>(this->model, degree, all_pcg_xfers);
  }
  for (const auto& it : all_parallel_degrees) {
    all_pcg_xfers.push_back(create_partition_attention_combine(this->model, 16/*num_heads*/, it));
    // rewrites for the inception model
    for (int i = 3; i <= 6; i++) {
      all_pcg_xfers.push_back(create_combine_inception(this->model, i-1/*num_convs*/, 5/*num_dims*/, it));
      all_pcg_xfers.push_back(create_combine_concat(this->model, i/*num_inputs*/, 5/*num_dims*/, it));
    }
    //all_pcg_xfers.push_back(create_partition_conv2d_combine(this->model, 5/*num_dims*/, it));
    all_pcg_xfers.push_back(create_partition_linear_combine(this->model, 3/*num_dims*/, it, AC_MODE_RELU, false));
    all_pcg_xfers.push_back(create_partition_linear_combine(this->model, 3/*num_dims*/, it, AC_MODE_SIGMOID, false));
    all_pcg_xfers.push_back(create_partition_linear_combine(this->model, 3/*num_dims*/, it, AC_MODE_NONE, false));
    all_pcg_xfers.push_back(create_partition_linear_combine(this->model, 4/*num_dims*/, it, AC_MODE_RELU, false));
    all_pcg_xfers.push_back(create_partition_linear_combine(this->model, 4/*num_dims*/, it, AC_MODE_SIGMOID, false));
    all_pcg_xfers.push_back(create_partition_linear_combine(this->model, 4/*num_dims*/, it, AC_MODE_NONE, false));
    all_pcg_xfers.push_back(create_partition_add_combine(this->model, 2/*parallel_dims*/, it/*num_parts*/));
    all_pcg_xfers.push_back(create_partition_add_combine(this->model, 1/*parallel_dims*/, it/*num_parts*/));
    all_pcg_xfers.push_back(create_partition_add_combine(this->model, 3/*parallel_dims*/, it/*num_parts*/));
    all_pcg_xfers.push_back(create_partition_relu_combine(this->model, 3/*parallel_dims*/, it/*num_parts*/));
    all_pcg_xfers.push_back(create_partition_softmax_combine(this->model, 0/*softmax_dim*/, 1/*parallel_dims*/, it/*num_parts*/));
    for (int num_combines = 1; num_combines < 5; num_combines++) {
      all_pcg_xfers.push_back(leading_relu_branch_combine(this->model, 3/*parallel_dim*/, it/*num_parts*/, num_combines));
      all_pcg_xfers.push_back(leading_relu_branch_partition(this->model, 3/*parallel_dim*/, it/*num_parts*/, num_combines));
    }
    {
      std::unordered_set<int> concat_num_inputs;
      for (size_t i = 0; i < this->model->operators.size(); i++)
        if (this->model->operators[i]->op_type == OP_CONCAT)
          concat_num_inputs.insert(this->model->operators[i]->numInputs);
      for (const auto& it2 : concat_num_inputs) {
        all_pcg_xfers.push_back(create_partition_concat_combine(this->model, it2/*num_inputs*/, 0/*concat_dim*/, 1/*parallel_dims*/, it/*num_parts*/));
        all_pcg_xfers.push_back(create_partition_concat_combine(this->model, it2/*num_inputs*/, 2/*concat_dim*/, 3/*parallel_dims*/, it/*num_parts*/));
      }
    }
  }
}

Graph *GraphSearchHelper::construct_graph() {
  Graph* graph = new Graph(this->model);
  std::unordered_map<const FlexFlow::Op*, Node> op_to_node_map;
  for (const FlexFlow::Op* dstOp : this->model->operators) {
    Node dstNode;
    dstNode.ptr = dstOp;
    dstNode.guid = this->model->node_global_guid++;
    op_to_node_map[dstOp] = dstNode;
    for (int j = 0; j < dstOp->numInputs; j++) {
      const FlexFlow::Op* srcOp = dstOp->inputs[j]->owner_op;
      assert(op_to_node_map.find(srcOp) != op_to_node_map.end());
      Node srcNode = op_to_node_map[srcOp];
      graph->add_edge(srcNode, dstNode, dstOp->inputs[j]->owner_idx, j);
    }
  }

  return graph;
}

void GraphSearchHelper::graph_optimize(size_t budget,
                             bool only_data_parallel,
                             std::unique_ptr<Graph>& best_graph,
                             std::unordered_map<Node, MachineView>& optimal_views)
{
  // Construct graph structure
  this->logger->debug() << "Starting graph optimization";

  Graph *graph = this->construct_graph();
  std::unordered_map<Node, MachineView> empty_strategy;
  if (!this->config.export_strategy_computation_graph_file.empty()) {
    graph->export_strategy_computation_graph(empty_strategy, this->config.export_strategy_computation_graph_file);
  }
  
  Node sink_node = graph->find_sink_node();
  GraphOptimizeResult optimal = this->generic_sequence_optimize<GraphOptimizeResult>(graph, sink_node, tl::nullopt/*output_shape*/, tl::nullopt/*input_shape*/);
  this->logger->debug() << "Total cache size: " << this->cached_optimized_graphs.size();
  std::cout << "Optimal cost: " << optimal.cost << std::endl;
  SimplificationSettings settings;
  settings.fuse_parallel_ops = true;
  settings.remove_noops = true;
  settings.remove_trailing_parallel_ops = true;
  settings.simplify_parallel_ops = true;
  best_graph = std::unique_ptr<Graph>(new Graph(optimal.graph.value()));
  best_graph->simplify(settings);
  best_graph->print_strategy_computation_graph(optimal.views);
  optimal_views = best_graph->optimal_views();
  // for (auto const &kv : optimal.views) {
  //   std::cout << "Node " << kv.first.to_string() << " View " << kv.second << std::endl;
  // }
}

void GraphSearchHelper::graph_optimize_no_split(
         size_t budget,
         bool only_data_parallel,
         std::unique_ptr<Graph>& best_graph,
         std::unordered_map<Node, MachineView>& optimal_views) {
  // Construct graph structure
  this->logger->debug() << "Starting graph optimization without split";

  Graph *graph = this->construct_graph();
  std::unordered_map<Node, MachineView> empty_strategy;
  if (!this->config.export_strategy_computation_graph_file.empty()) {
    graph->export_strategy_computation_graph(empty_strategy, this->config.export_strategy_computation_graph_file);
  }
  
  SimplificationSettings settings;
  settings.simplify_parallel_ops = true;
  best_graph = this->base_optimize(graph, settings);
  optimal_views = best_graph->optimal_views();

  this->logger->debug() << "Total cache size: " << this->cached_optimized_graphs.size();
  std::cout << "Optimal cost: " << best_graph->optimal_cost() << std::endl;
}

static void graph_log_representation(Graph const *graph, RecursiveLogger &logger) {
  using FlexFlow::PCG::Utils::topo_sort;

  std::vector<Node> topo_sorted;
  topo_sort(*graph, &topo_sorted);
  std::ostringstream oss;
  for (Node const &n : topo_sorted) {
    logger.spew() << n.to_string();
  }
}

void GraphSearchHelper::find_rewrite_matches(Graph const *graph, std::vector<GraphXferMatch>& matches) const {
  std::vector<GraphXfer*> xfers;
  this->load_graph_substitutions(xfers);

  for (GraphXfer* xfer : xfers) {
    log_xfer_matches.debug() << "Finding matches for xfer: " << xfer->get_name();
    xfer->find_matches(graph, matches);
  }
  log_xfer_matches.debug() << "Finished finding xfer matches";
}

tl::optional<Node> GraphSearchHelper::find_split_node(Graph const *graph, int base_optimize_threshold) const {
  using FlexFlow::PCG::Utils::get_edges;
  using FlexFlow::PCG::Utils::nodes;
  using FlexFlow::PCG::Utils::post_dominators;

  this->logger->enter();

  int graph_size = nodes(*graph).size();
  this->logger->debug() << "Finding split node for graph (size " << graph_size 
                        << ") with threshold " << base_optimize_threshold;

  if (graph_size <= base_optimize_threshold) {
    this->logger->debug() << "Graph size underneath threshold. Returning nullopt";
    return tl::nullopt;
  }

  std::vector<Edge> edges = get_edges(*graph);
  std::unordered_map<Edge, int>  edge_scores;

  for (Edge const &e : edges) {
    edge_scores[e] = 0;
  }

  std::vector<GraphXferMatch> matches;
  this->find_rewrite_matches(graph, matches);
  this->logger->debug() << "Found " << matches.size() << " rewrite matches";
  this->logger->enter();
  for (GraphXferMatch const &match : matches) {
    auto msg = this->logger->spew();
    msg << match.get_xfer()->get_name() << " : ";
    std::unordered_set<Node> nodes = match.get_nodes();
    for (Node const &node : nodes) { 
      msg << node.to_string() << " ";
    }
  }
  this->logger->leave();

  for (GraphXferMatch const &match : matches) { 
    for (Edge const &e : edges) {
      if (match.containsEdge(graph, e)) {
        edge_scores[e]++;
      }
    }
  }

  this->logger->debug() << "Edge weights: ";
  this->logger->enter();
  for (Edge const &e : edges) {
    this->logger->debug() << e.srcOp.to_string() << "/" << e.srcIdx << " -> " 
                          << e.dstOp.to_string() << "/" << e.dstIdx << " : " 
                          << edge_scores.at(e);
  }
  this->logger->leave();

  std::unordered_map<Node, std::unordered_set<Node>> post_dominator_map = post_dominators(*graph);
  Node source_node = graph->find_source_node();
  std::unordered_set<Node> possible_bottlenecks = post_dominator_map.at(source_node);
  Node sink_node = graph->find_sink_node();

  int best_weight = 0;
  tl::optional<Node> best = tl::nullopt;
  int best_size = graph_size;
  this->logger->enter();
  for (Node const &possible_bottleneck : possible_bottlenecks) {
    if (possible_bottleneck == sink_node || possible_bottleneck == source_node) {
      continue;
    }

    int weight = 0; 
    for (Edge const &e : graph->outEdges.at(possible_bottleneck)) {
      weight += edge_scores.at(e);
    }
    this->logger->debug() << "Potential bottleneck node " << possible_bottleneck.to_string() << " has weight " << weight;
    if (weight < best_weight) {
      best_weight = weight;
      best = possible_bottleneck;
    } else if (weight == best_weight) {
      // break ties by trying to choosing the split that produces the pre_graph with size closest to the threshold, 
      // favoring everything with smaller size over everything with larger size
      std::unique_ptr<Graph> pre_graph, post_graph;
      std::tie(pre_graph, post_graph) = graph->split_at_node(possible_bottleneck);
      int current_size = nodes(*pre_graph).size();

      bool best_is_under = best_size <= base_optimize_threshold;
      bool current_is_under = current_size <= base_optimize_threshold;

      bool condition1 = current_is_under && !best_is_under;
      bool condition2 = current_is_under && best_is_under && current_size > best_size;
      bool condition3 = !current_is_under && !best_is_under && current_size < best_size;

      if (condition1 || condition2 || condition3) {
        best_weight = weight;
        best = possible_bottleneck;
        best_size = current_size;
      }
    }
  }
  this->logger->leave();
  this->logger->leave();

  return best;
}

std::unique_ptr<Graph> GraphSearchHelper::base_optimize(Graph const *r_graph, SimplificationSettings const &simplification_settings) {
  // Construct graph substitutions
  this->logger->enter();

  this->logger->debug() << "Optimizing base graph: ";
  this->logger->enter();
  /* graph_log_representation(r_graph, *this->logger); */
  // r_graph->print_dot();
  this->logger->leave();
  this->logger->debug() << "Starting cost: " << r_graph->optimal_cost();

  std::vector<GraphXfer*> xfers;
  this->load_graph_substitutions(xfers);

  Graph *graph = new Graph(*r_graph);

  std::priority_queue<Graph*, std::vector<Graph*>, GraphCompare> candidates;
  std::unordered_set<size_t> hashmap;
  candidates.push(graph);
  hashmap.insert(graph->hash());
  Graph *best_graph = new Graph(*graph);
  float best_cost = best_graph->optimal_cost();
  int counter = 0;
  const float alpha = this->model->config.search_alpha;

  int budget = model->config.search_budget;
  if (budget == 0) {
    log_xfers.warning() << "Base search budget is set to 0. This is probably not what you want (use the --budget flag to set the base search budget)";
  }
  for (int iter = 0; iter < budget || budget == -1; iter++) {
    log_xfers.spew() << "Considering " << candidates.size() << " candidates";
    if (candidates.empty()) {
      break;
    }

    Graph *cur_graph = candidates.top();
    candidates.pop();
    if (cur_graph->optimal_cost() < best_graph->optimal_cost()) {
      delete best_graph;
      best_graph = cur_graph;
      best_cost = cur_graph->optimal_cost();
    } else if (cur_graph->optimal_cost() > best_cost * alpha) {
      continue;
    }

    log_xfers.info("[%d] cur_cost(%.4lf) best_cost(%.4lf) candidates.size(%zu)",
           counter, cur_graph->optimal_cost(), best_cost, candidates.size());

    log_xfers.debug() << "Considering " << xfers.size() << " possible xfers";
    for (size_t i = 0; i < xfers.size(); i++) {
      int num_matches_found = 0,
          num_matches_rejected = 0;
      log_xfers.debug() << "Considering xfer: " << xfers[i]->get_name();
      xfers[i]->run(0, cur_graph, candidates, hashmap, best_cost * alpha, 1000, simplification_settings, num_matches_found, num_matches_rejected);
      log_xfers.debug() << "Rejected [ " << num_matches_rejected << " / " << num_matches_found << " ] matches";
      /* std::cout << "." << std::flush; */
    }
    /* std::cout << std::endl; */
    if (best_graph != cur_graph) {
      delete cur_graph;
    }
  }

  this->logger->debug() << "Optimized cost: " << best_graph->optimal_cost();
  //best_graph->print_dot();
  this->logger->leave();
  return std::unique_ptr<Graph>(best_graph);
}

size_t gs_dp_state_hash(Graph const *graph, 
                        Node const &sink_node,
                        tl::optional<ParallelTensorShape> const &output_shape,
                        tl::optional<ParallelTensorShape> const &input_shape)
{
  size_t key = graph->hash();
  hash_combine(key, sink_node.ptr);
  hash_combine(key, output_shape);
  hash_combine(key, input_shape);
  return key;
}

float GraphSearchHelper::sequence_optimize(
    Graph const *graph, 
    Node const &sink_node,
    tl::optional<ParallelTensorShape> const &output_shape,
    tl::optional<ParallelTensorShape> const &input_shape) 
{
  return this->generic_sequence_optimize<float>(graph, sink_node, output_shape, input_shape);
}

template <>
tl::optional<float> GraphSearchHelper::try_get_cost_from_cache<float>(size_t hash) const {
  if (this->cached_optimized_graphs.find(hash) == this->cached_optimized_graphs.end()) {
    return tl::nullopt;
  } else {
    return this->cached_optimized_graphs.at(hash);
  }
}

template <>
float GraphSearchHelper::get_optimal_cost<float>(std::unique_ptr<Graph> optimized) const {
  return optimized->generic_optimal_cost<float>(); 
}

template <>
GraphCostResult GraphSearchHelper::get_optimal_cost<GraphCostResult>(std::unique_ptr<Graph> optimized) const {
  return optimized->generic_optimal_cost<GraphCostResult>();
}

template <>
GraphOptimizeResult GraphSearchHelper::get_optimal_cost<GraphOptimizeResult>(std::unique_ptr<Graph> optimized) const {
  GraphOptimizeResult result;
  result.graph = *optimized;
  GraphCostResult gcr = optimized->generic_optimal_cost<GraphCostResult>();
  result.cost = gcr.cost;
  result.views = gcr.views;
  return result;
}

template <>
tl::optional<GraphCostResult> GraphSearchHelper::try_get_cost_from_cache<GraphCostResult>(size_t hash) const {
  return tl::nullopt;
}

template <>
tl::optional<GraphOptimizeResult> GraphSearchHelper::try_get_cost_from_cache<GraphOptimizeResult>(size_t hash) const {
  return tl::nullopt;
}

template <>
void GraphSearchHelper::try_cache_result<float>(size_t hash, float const &value) {
  this->cached_optimized_graphs[hash] = value;
}

template <>
void GraphSearchHelper::try_cache_result<GraphCostResult>(size_t hash, GraphCostResult const &value) { }

template <>
void GraphSearchHelper::try_cache_result<GraphOptimizeResult>(size_t hash, GraphOptimizeResult const &value) { }

template <typename T>
T GraphSearchHelper::execute_sequence_split(
    std::unique_ptr<Graph> const &pre_graph,
    std::unique_ptr<Graph> const &post_graph,
    tl::optional<ParallelTensorShape> const &output_shape,
    tl::optional<ParallelTensorShape> const &input_shape,
    Node const &sink_node,
    Node const &bottleneck, 
    ParallelTensorShape const &bottleneck_output_shape)
{
  return sequence_cost<T>(
    this->generic_sequence_optimize<T>(pre_graph.get(), bottleneck, bottleneck_output_shape, input_shape),
    this->generic_sequence_optimize<T>(post_graph.get(), sink_node, output_shape, bottleneck_output_shape)
  );
}

template <typename T>
T GraphSearchHelper::generic_sequence_optimize(
    Graph const *graph, 
    Node const &sink_node, 
    tl::optional<ParallelTensorShape> const &output_shape, 
    tl::optional<ParallelTensorShape> const &input_shape)
{
  /* int starting_depth = this->logger->get_depth(); */

  this->logger->enter();

  size_t hash = gs_dp_state_hash(graph, sink_node, output_shape, input_shape);
  tl::optional<T> cached = this->try_get_cost_from_cache<T>(hash);
  if (cached.has_value()) {
    this->logger->spew() << "Optimizing graph with " << graph->inEdges.size() << " nodes";
    this->logger->enter();
    this->logger->spew() << "Nodes: ";
    this->logger->enter();
    graph_log_representation(graph, *this->logger);
    this->logger->leave();
    this->logger->spew() << "Retrieved value from cache: " << cached.value();
    this->logger->leave();
    this->logger->leave();

    /* this->logger->check_same_as(starting_depth); */
    return cached.value();
  }

  this->logger->debug() << "Optimizing graph with " << graph->inEdges.size() << " nodes";
  this->logger->enter();
  this->logger->spew() << "Nodes: ";
  this->logger->enter();
  graph_log_representation(graph, *this->logger);
  this->logger->leave();
  this->logger->debug() << "Graph hash: " << std::setw(32) << std::setfill('0') << graph->hash();
  if (input_shape.has_value()) {
    this->logger->debug() << "Input shape: " << input_shape.value();
  } else {
    this->logger->debug() << "Input shape: <none>";
  }
  if (output_shape.has_value()) {
    this->logger->debug() << "Output shape: " << output_shape.value();
  } else {
    this->logger->debug() << "Output shape: <none>";
  }

  tl::optional<Node> bottleneck = this->find_split_node(graph, this->config.base_optimize_threshold);
  /* Node bottleneck = graph->find_nontrivial_bottleneck_node(sink_node, source_node); */

  T return_value;
  if (!bottleneck.has_value()) {
    this->logger->debug() << "Applying base case";
    Graph to_optimize(*graph);
    if (input_shape.has_value()) {
      Node input_node = this->model->get_or_create_input_node(input_shape.value());
      Node noop_node = this->model->get_or_create_noop_node(input_node.ptr->outputs[0]);
      Graph input_graph(this->model);
      Edge e(input_node, noop_node, 0, 0);
      input_graph.add_edge(e);

      Node old_source_node = graph->find_source_node();
      ParallelTensorShape old_source_output_shape = old_source_node.ptr->outputs[0]->get_shape();
      input_graph.reshape_output_tensor(old_source_output_shape);

      Node new_sink_node = input_graph.find_sink_node();
      assert (new_sink_node.ptr->numOutputs == 1);
      assert (new_sink_node.ptr->outputs[0]->get_shape() == old_source_output_shape);

      to_optimize.replace_subgraph({old_source_node}, input_graph);
    }
    SimplificationSettings settings;
    if (output_shape.has_value()) {
      to_optimize.reshape_output_tensor(output_shape.value());
      Node sink_node = to_optimize.find_sink_node();
      Node noop_node = this->model->get_or_create_noop_node(sink_node.ptr->outputs[0]);
      to_optimize.add_edge(sink_node, noop_node, 0, 0);
    } else {
      settings.remove_trailing_parallel_ops = true;
    }
    settings.simplify_parallel_ops = true;
    std::unique_ptr<Graph> optimized = this->base_optimize(&to_optimize, settings);
    this->logger->leave();
    return_value = get_optimal_cost<T>(std::move(optimized)); //optimized->generic_optimal_cost<T>();
  } else {
    this->logger->debug() << "Applying recursive case on bottleneck " << bottleneck.value().guid;
    std::unique_ptr<Graph> pre_graph, post_graph;
    std::tie(pre_graph, post_graph) = graph->split_at_node(bottleneck.value());

    MachineResource resources(this->model->config);
    std::vector<MachineView> valid_machine_views = this->model->search->get_valid_machine_views(bottleneck.value().ptr, resources);

    float best_cost = std::numeric_limits<float>::infinity();
    tl::optional<ParallelTensorShape> best_shape = tl::nullopt;
    this->logger->enter();
    for (ParallelTensorShape const &bottleneck_output_shape : this->possible_split_output_tensor_shapes(bottleneck.value())) {
      this->logger->debug() << "Considering boundary shape " << bottleneck_output_shape;
      this->logger->enter();
      // TODO @lockshaw we really should create the merged graph here since it's possible though unlikely for there 
      // to be hidden transfer costs between modules due to device assignment changes across the boundaries
      
      // We wait to add the communication nodes between boundaries so we don't accidentally split on them 
      // and keep processing the pure computation graph
      // The bottleneck node is kept in the postgraph purely as a placeholder and will be replaced with an Input/NoOp
      // sequence before any rewrites are actually performed
      //this->logger->debug() << "Finding cost of pre_graph (" << bottleneck_output_shape << ")";
      //float pre_cost = this->generic_sequence_optimize<float>(pre_graph.get(), bottleneck.value(), bottleneck_output_shape, input_shape);
      //this->logger->debug() << "Cost of pre_graph (" << bottleneck_output_shape << "): " << pre_cost;
      //this->logger->debug() << "Finding cost of post_graph (" << bottleneck_output_shape << ")";
      //float post_cost = this->generic_sequence_optimize<float>(post_graph.get(), sink_node, output_shape, bottleneck_output_shape);
      //this->logger->debug() << "Cost of post_graph (" << bottleneck_output_shape << "): " << post_cost;
      //float current_cost = pre_cost + post_cost;
      float current_cost = this->execute_sequence_split<float>(
        pre_graph,
        post_graph,
        output_shape,
        input_shape,
        sink_node,
        bottleneck.value(),
        bottleneck_output_shape
      );

      if (current_cost < best_cost) {
        best_cost = current_cost;
        best_shape = bottleneck_output_shape;
      }
      this->logger->leave();
      this->logger->debug() << "Boundary shape " << bottleneck_output_shape << " has cost: " << current_cost;
    }
    this->logger->leave();

    if (best_shape.has_value()) {
      this->logger->debug() << "Best intermediate shape found: " << best_shape.value();
    } else {
      this->logger->debug() << "No valid intermediate shapes found";
    }

    if (best_cost != std::numeric_limits<float>::infinity()) {
      return_value = this->execute_sequence_split<T>(
        pre_graph,
        post_graph,
        output_shape,
        input_shape,
        sink_node,
        bottleneck.value(),
        best_shape.value()
      );
    }
  }

  this->try_cache_result<T>(hash, return_value);
  this->logger->leave();
  this->logger->leave();
  return return_value;
}

std::vector<ParallelTensorShape> GraphSearchHelper::possible_split_output_tensor_shapes(Node const &source_node) const {
  this->logger->enter();

  this->logger->debug() << "Finding possible output tensor shapes for node " << source_node.guid;
  assert (source_node.ptr->numOutputs == 1);
  ParallelTensor output_tensor = source_node.ptr->outputs[0];
  for (int i = 0; i < output_tensor->num_dims; i++) {
    assert (output_tensor->dims[i].degree == 1);
  }

  std::vector<ParallelTensorShape> without_replicas;

  int num_devices = this->config.numNodes * this->config.workersPerNode;
  int degrees[MAX_TENSOR_DIM];
  std::fill_n(degrees, MAX_TENSOR_DIM, 1);

  ParallelTensorShape base_shape;
  base_shape.num_dims = output_tensor->num_dims;
  for (int i = 0; i < output_tensor->num_dims; i++) {
    base_shape.dims[i].degree = 1;
    base_shape.dims[i].size = output_tensor->dims[i].size;
  }
  without_replicas.push_back(base_shape);

  this->logger->enter();
  while (true) {
    bool is_done = true;
    for (int i = 0; i < output_tensor->num_dims; i++) {
      degrees[i] *= 2;
      if (degrees[i] > num_devices) {
        degrees[i] = 1;
      } else {
        is_done = false;
        break;
      }
    }
    std::ostringstream oss;
    for (int i = 0; i < output_tensor->num_dims; i++) {
      oss << degrees[i] << " ";
    }
    this->logger->spew() << "Considering: " << oss.str();
    if (is_done) {
      break;
    }

    bool is_valid = true;
    int total_degree = 1;
    ParallelTensorShape shape;
    shape.num_dims = output_tensor->num_dims;
    for (int i = 0; i < output_tensor->num_dims; i++) {
      total_degree *= degrees[i];
      shape.dims[i].degree = degrees[i];
      shape.dims[i].size = output_tensor->dims[i].size;
      if (shape.dims[i].size % shape.dims[i].degree != 0) {
        is_valid = false;
      }
    }
    if (total_degree <= num_devices && is_valid) {
      without_replicas.push_back(shape);
    }
  }
  this->logger->leave();

  this->logger->debug() << "Found " << without_replicas.size() << " possible tensor output shapes without replicas";
  this->logger->debug() << "They are:";
  this->logger->enter();
  for (auto const & shape : without_replicas) {
    this->logger->debug() << shape;
  }
  this->logger->leave();
  this->logger->leave();
  return without_replicas;
}

void GraphSearchHelper::subgraph_optimize(Graph *subgraph) 
{
  
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

  std::ostringstream oss;
  oss << "partition_linear_combine["
      << "num_dims=" << num_dims
      << ",num_parts=" << num_parts
      << ",activation=" << activation
      << ",use_bias=" << use_bias
      << "]";
  subst->name = oss.str();

  return subst;
}

GraphXfer* create_partition_conv2d_combine(FFModel* model,
                                           int num_dims,
                                           int num_parts) {
  assert(num_dims == 5);
  GraphXfer* subst = new GraphXfer(model);
  TensorX input = subst->new_tensor();
  OpX* conv1 = subst->create_conv2d(input, NULL/*matchOpX*/);
  OpX* repartition = subst->create_repartition(input, num_dims-2, num_parts);
  OpX* conv2 = subst->create_conv2d(repartition->outputs[0], conv1/*matchOpX*/);
  OpX* combine = subst->create_combine(conv2->outputs[0], num_dims-2, num_parts);
  subst->map_output(conv1->outputs[0], combine->outputs[0]);
  subst->srcOps.push_back(conv1);
  subst->dstOps.push_back(repartition);
  subst->dstOps.push_back(conv2);
  subst->dstOps.push_back(combine);

  std::ostringstream oss;
  oss << "partition_conv2d_combine["
      << "num_dims=" << num_dims
      << ",num_parts=" << num_parts
      << "]";
  subst->name = oss.str();

  return subst;
}

GraphXfer* create_combine_inception(FFModel* model,
                                    int num_convs,
                                    int num_dims,
                                    int num_parts) {
  // 3 convs and 1 pool2d
  assert(num_dims == 5);
  GraphXfer* subst = new GraphXfer(model);
  TensorX input = subst->new_tensor();
  OpX* src_combine = subst->create_combine(input, num_dims-2, num_parts);
  subst->srcOps.push_back(src_combine);
  std::vector<OpX*> src_convs;
  for (int i = 0; i < num_convs; i++) {
    OpX* conv = subst->create_conv2d(src_combine->outputs[0], NULL/*matchOpX*/);
    src_convs.push_back(conv);
    subst->srcOps.push_back(conv);
  }
  OpX* src_pool = subst->create_pool2d(src_combine->outputs[0], NULL/*matchOpX*/);
  subst->srcOps.push_back(src_pool);
  // dst ops
  std::vector<OpX*> dst_convs;
  for (int i = 0; i < num_convs; i++) {
    OpX* conv = subst->create_conv2d(input, src_convs[i]/*matchOpX*/);
    OpX* comb = subst->create_combine(conv->outputs[0], num_dims-2, num_parts);
    subst->dstOps.push_back(conv);
    subst->dstOps.push_back(comb);
    subst->map_output(src_convs[i]->outputs[0], comb->outputs[0]);
  }
  OpX* dst_pool = subst->create_pool2d(input, src_pool/*matchOpX*/);
  OpX* dst_comb = subst->create_combine(dst_pool->outputs[0], num_dims-2, num_parts);
  subst->dstOps.push_back(dst_pool);
  subst->dstOps.push_back(dst_comb);
  subst->map_output(src_pool->outputs[0], dst_comb->outputs[0]);
  subst->name = "create_combine_inceptionA";
  return subst;
}

GraphXfer* create_combine_concat(FFModel* model,
                                 int num_inputs,
                                 int num_dims,
                                 int num_parts) {
  // assert 5D
  assert(num_dims == 5);
  GraphXfer* subst = new GraphXfer(model);
  std::vector<TensorX> inputs, concat_inputs;
  std::vector<OpX*> combines;
  for (int i = 0; i < num_inputs; i++) {
    inputs.push_back(subst->new_tensor());
    combines.push_back(subst->create_combine(inputs[i], num_dims-2, num_parts));
    concat_inputs.push_back(combines[i]->outputs[0]);
    subst->srcOps.push_back(combines[i]);
  }
  OpX* concat1 = subst->create_concat(concat_inputs.data(), num_inputs, NULL/*matchOpX*/, 2);
  subst->srcOps.push_back(concat1);
  OpX* concat2 = subst->create_concat(inputs.data(), num_inputs, concat1/*matchOpX*/, 2);
  OpX* combine = subst->create_combine(concat2->outputs[0], num_dims-2, num_parts);
  subst->dstOps.push_back(concat2);
  subst->dstOps.push_back(combine);
  subst->map_output(concat1->outputs[0], combine->outputs[0]);
  subst->name = "create_combine_concat";
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

  std::ostringstream oss;
  oss << "partition_attention_combine["
      << "num_heads=" << num_heads
      << ",num_parts=" << num_parts
      << "]";
  subst->name = oss.str();

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

  std::ostringstream oss;
  oss << "replicate_attention_reduce["
      << "num_heads=" << num_heads
      << ",num_parts=" << num_parts
      << "]";
  subst->name = oss.str();

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

  std::ostringstream oss;
  oss << "replicate_linear_combine["
      << "num_dims=" << num_dims
      << ",num_parts=" << num_parts
      << ",activation=" << activation
      << ",use_bias=" << use_bias
      << "]";
  subst->name = oss.str();

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

  std::ostringstream oss;
  oss << "partition_add_combine["
      << "parallel_dim=" << parallel_dim
      << ",num_parts=" << num_parts
      << "]";
  subst->name = oss.str();

  return subst;
}

GraphXfer* create_combine_add_partition(FFModel* model,
                                        int parallel_dim,
                                        int num_parts)
{
  GraphXfer* subst = new GraphXfer(model);
  TensorX input1 = subst->new_tensor();
  TensorX input2 = subst->new_tensor();
  OpX* add1 = subst->create_element_binary(input1, input2, OP_EW_ADD);

  OpX* combine1 = subst->create_combine(input1, parallel_dim, num_parts);
  OpX* combine2 = subst->create_combine(input2, parallel_dim, num_parts);
  OpX* add2 = subst->create_element_binary(combine1->outputs[0],
                                           combine2->outputs[0],
                                           OP_EW_ADD);
  OpX* repartition = subst->create_repartition(add2->outputs[0], parallel_dim, num_parts);
  subst->map_output(add1->outputs[0], repartition->outputs[0]);
  subst->srcOps.push_back(add1);
  subst->dstOps.push_back(combine1);
  subst->dstOps.push_back(combine2);
  subst->dstOps.push_back(add2);
  subst->dstOps.push_back(repartition);

  std::ostringstream oss;
  oss << "combine_add_partition["
      << "parallel_dim=" << parallel_dim
      << ",num_parts=" << num_parts
      << "]";
  subst->name = oss.str();

  return subst;
}

GraphXfer* create_partition_relu_combine(FFModel* model,
                                         int parallel_dim,
                                         int num_parts) 
{
  GraphXfer* subst = new GraphXfer(model);
  TensorX input = subst->new_tensor();
  OpX* relu1 = subst->create_element_unary(input, OP_RELU);

  OpX* partition = subst->create_repartition(input, parallel_dim, num_parts);
  OpX* relu2 = subst->create_element_unary(partition->outputs[0], OP_RELU);
  OpX* combine = subst->create_combine(relu2->outputs[0], parallel_dim, num_parts);

  subst->map_output(relu1->outputs[0], combine->outputs[0]);

  subst->srcOps.push_back(relu1);

  subst->dstOps.push_back(partition);
  subst->dstOps.push_back(relu2);
  subst->dstOps.push_back(combine);

  std::ostringstream oss;
  oss << "partition_relu_combine["
      << "parallel_dim=" << parallel_dim
      << ",num_parts=" << num_parts
      << "]";
  subst->name = oss.str();

  return subst;
}

GraphXfer* create_combine_relu_partition(FFModel* model,
                                         int parallel_dim,
                                         int num_parts) 
{
  GraphXfer* subst = new GraphXfer(model);
  TensorX input = subst->new_tensor();
  OpX* relu1 = subst->create_element_unary(input, OP_RELU);

  OpX* combine = subst->create_combine(input, parallel_dim, num_parts);
  OpX* relu2 = subst->create_element_unary(combine->outputs[0], OP_RELU);
  OpX* partition = subst->create_repartition(relu2->outputs[0], parallel_dim, num_parts);

  subst->map_output(relu1->outputs[0], partition->outputs[0]);

  subst->srcOps.push_back(relu1);

  subst->dstOps.push_back(combine);
  subst->dstOps.push_back(relu2);
  subst->dstOps.push_back(partition);

  std::ostringstream oss;
  oss << "combine_relu_partition["
      << "parallel_dim=" << parallel_dim
      << ",num_parts=" << num_parts
      << "]";
  subst->name = oss.str();

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

  std::ostringstream oss;
  oss << "partition_concat_combine[" 
      << "num_inputs=" << num_inputs
      << ",concat_dim=" << concat_dim
      << ",parallel_dim=" << parallel_dim
      << ",num_parts=" << num_parts
      << "]";
  subst->name = oss.str();

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

  std::ostringstream oss;
  oss << "partition_softmax_combine["
      << "softmax_dim=" << softmax_dim
      << ",parallel_dim=" << parallel_dim
      << ",num_parts=" << num_parts
      << "]";
  subst->name = oss.str();

  return subst;
}

GraphXfer* create_combine_softmax_partition(FFModel* model,
                                            int softmax_dim,
                                            int parallel_dim,
                                            int num_parts)
{
  assert(parallel_dim != softmax_dim);
  GraphXfer* subst = new GraphXfer(model);
  TensorX input = subst->new_tensor();
  OpX* softmax1 = subst->create_softmax(input, softmax_dim);
  OpX* combine = subst->create_combine(input, parallel_dim, num_parts);
  OpX* softmax2 = subst->create_softmax(combine->outputs[0], softmax_dim);
  OpX* repartition = subst->create_repartition(softmax2->outputs[0], parallel_dim, num_parts);
  subst->map_output(softmax1->outputs[0], repartition->outputs[0]);
  subst->srcOps.push_back(softmax1);
  subst->dstOps.push_back(combine);
  subst->dstOps.push_back(softmax2);
  subst->dstOps.push_back(repartition);

  std::ostringstream oss;
  oss << "combine_softmax_partition["
      << "softmax_dim=" << softmax_dim
      << ",parallel_dim=" << parallel_dim
      << ",num_parts=" << num_parts
      << "]";
  subst->name = oss.str();

  return subst;
}


GraphXfer* leading_relu_branch_combine(FFModel* model, int parallel_dim, int num_parts, int num_combines) 
{
  GraphXfer* subst = new GraphXfer(model);
  TensorX input = subst->new_tensor();
  OpX* old_partition = subst->create_repartition(input, parallel_dim, num_parts);
  std::vector<OpX*> old_combines;
  for (int i = 0; i < num_combines; i++) {
    old_combines.push_back(subst->create_combine(input, parallel_dim, num_parts));
  }

  OpX* new_partition = subst->create_repartition(input, parallel_dim, num_parts);
  std::vector<OpX*> new_noops;
  for (int i =0; i < num_combines; i++) {
    new_noops.push_back(subst->create_noop(input));
  }

  subst->map_output(old_partition->outputs[0], new_partition->outputs[0]);
  for (int i = 0; i < num_combines; i++) {
    subst->map_output(old_combines[i]->outputs[0], new_noops[i]->outputs[0]);
  }

  subst->srcOps.push_back(old_partition);
  subst->srcOps.insert(subst->srcOps.end(), old_combines.begin(), old_combines.end());
  subst->dstOps.push_back(new_partition);
  subst->dstOps.insert(subst->dstOps.end(), new_noops.begin(), new_noops.end());

  std::ostringstream oss;
  oss << "leading_relu_branch_combine["
      << "parallel_dim=" << parallel_dim
      << ",num_parts=" << num_parts
      << ",num_combines=" << num_combines
      << "]";
  subst->name = oss.str();

  return subst;
}

GraphXfer* leading_relu_branch_partition(FFModel* model, int parallel_dim, int num_parts, int num_partitions) 
{
  GraphXfer* subst = new GraphXfer(model);
  TensorX input = subst->new_tensor();
  OpX* old_combine = subst->create_combine(input, parallel_dim, num_parts);
  std::vector<OpX*> old_partitions;
  for (int i = 0; i < num_partitions; i++) {
    old_partitions.push_back(subst->create_repartition(input, parallel_dim, num_parts));
  }

  OpX* new_combine = subst->create_combine(input, parallel_dim, num_parts);
  std::vector<OpX*> new_noops;
  for (int i =0; i < num_partitions; i++) {
    new_noops.push_back(subst->create_noop(input));
  }

  subst->map_output(old_combine->outputs[0], new_combine->outputs[0]);
  for (int i = 0; i < num_partitions; i++) {
    subst->map_output(old_partitions[i]->outputs[0], new_noops[i]->outputs[0]);
  }

  subst->srcOps.push_back(old_combine);
  subst->srcOps.insert(subst->srcOps.end(), old_partitions.begin(), old_partitions.end());
  subst->dstOps.push_back(new_combine);
  subst->dstOps.insert(subst->dstOps.end(), new_noops.begin(), new_noops.end());

  std::ostringstream oss;
  oss << "leading_relu_branch_partition["
      << "parallel_dim=" << parallel_dim
      << ",num_parts=" << num_parts
      << ",num_partitions=" << num_partitions
      << "]";
  subst->name = oss.str();

  return subst;
}

GraphXfer* create_linear_relu_merge(FFModel* model, int num_dims, bool use_bias) {
  GraphXfer* subst = new GraphXfer(model);
  TensorX input = subst->new_tensor();
  OpX* old_linear = subst->create_linear(input, nullptr, num_dims, AC_MODE_NONE, use_bias);
  OpX* old_relu = subst->create_relu(old_linear->outputs[0]);

  OpX* new_linear = subst->create_linear(input, old_linear, num_dims, AC_MODE_RELU, use_bias);

  subst->map_output(old_relu->outputs[0], new_linear->outputs[0]);
  subst->srcOps.push_back(old_linear);
  subst->srcOps.push_back(old_relu);
  subst->dstOps.push_back(new_linear);

  std::ostringstream oss;
  oss << "linear_relu_merge["
      << "num_dims=" << num_dims
      << ",use_bias=" << use_bias 
      << "]";
  subst->name = oss.str();

  return subst;
}

}; // namespace FlexFlow::PCG

namespace FlexFlow {

using PCG::Graph;
using PCG::Node;
using PCG::Edge;

void FFModel::graph_optimize(size_t budget,
                             bool only_data_parallel,
                             std::unique_ptr<Graph>& best_graph,
                             std::unordered_map<Node, MachineView>& optimal_views)
{
  this->graph_search->graph_optimize(budget, only_data_parallel, best_graph, optimal_views);
}

bool FFModel::convert_graph_to_operators(const Graph* graph,
                                      const std::unordered_map<Node, MachineView>& optimal_views)
{
  // Clear operators
  operators.clear();
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
  while (index < queue.size()) {
    Node node = queue[index++];
    assert(node.ptr != NULL);
    const auto& inList = graph->inEdges.find(node)->second;
    ParallelTensor inputs[MAX_NUM_INPUTS];
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
        NoOp* noop = (NoOp*) node.ptr;
        new_op = new NoOp(*this, OP_INPUT, noop->input_tensor_guid, node.ptr->outputs[0]);
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
      case OP_DROPOUT:
      {
        new_op = new Dropout(*this, *(Dropout*)node.ptr, inputs[0]);
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
        ParallelTensor kernel;
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
          kernel = create_parallel_weight<3>(dims, DT_FLOAT, NULL/*owner_op*/,
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
    operators.push_back(new_op);
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
  while (operators[operators.size()-1]->is_parallel_op()) {
    Op* op = operators[operators.size()-1];
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
    operators.pop_back();
  }
  return true;
}


};
