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
#include "dominators.h"
#include "legion.h"
#include "legion/legion_utilities.h"
#include "ops/linear.h"
#include "ops/conv_2d.h"
#include "ops/pool_2d.h"
#include "ops/embedding.h"
#include "ops/element_unary.h"
#include "ops/flat.h"
#include "ops/attention.h"
#include "ops/softmax.h"
#include "ops/concat.h"
#include "parallel_ops/partition.h"
#include "parallel_ops/replicate.h"
#include "parallel_ops/reduction.h"
#include "parallel_ops/fused_parallel_op.h"
#include "parallel_ops/combine.h"

using namespace Legion;

LegionRuntime::Logger::Category log_dp("DP");
LegionRuntime::Logger::Category log_graph("graph");

const MachineView MachineView::NO_VIEW = MachineView();

const Node Node::INVALID_NODE = Node();

MachineView::MachineView()
: device_type(MachineView::GPU), ndims(0), start_device_id(0)
{
  for (int i = 0; i < MAX_TENSOR_DIM; i++) {
    dim[i] = stride[i] = 0;
  }
}

std::vector<int> MachineView::device_ids() const {
  std::vector<int> device_ids_list;

  if (this->ndims == 0) {
    return { this->start_device_id };
  }

  Domain d;
  d.dim = this->ndims;
  for (int i = 0; i < d.dim; i++) {
    d.rect_data[i] = 0;
    d.rect_data[i+d.dim] = this->dim[i]-1;
  }
  for (Domain::DomainPointIterator it(d); it; it++) {
    device_ids_list.push_back(this->get_device_id(*it));
  }

  return device_ids_list;
}

size_t MachineView::num_parts() const
{
  size_t parts = 1;
  for (int i = 0; i < ndims; i++) {
    parts *= dim[i];
  }
  return parts;
}

size_t MachineView::hash() const
{
  size_t ret = 17;
  ret = ret * 31 + std::hash<int>()(device_type);
  ret = ret * 31 + std::hash<int>()(ndims);
  ret = ret * 31 + std::hash<int>()(start_device_id);
  for (int i = 0; i < ndims; i++) {
    ret = ret * 31 + std::hash<int>()(dim[i]);
    ret = ret * 31 + std::hash<int>()(stride[i]);
  }
  return ret;
}

size_t MachineResource::hash() const
{
  size_t ret = 17;
  ret = ret * 31 + std::hash<int>()(num_nodes);
  ret = ret * 31 + std::hash<int>()(available_gpus_per_node);
  ret = ret * 31 + std::hash<int>()(available_cpus_per_node);
  ret = ret * 31 + std::hash<int>()(start_gpu_id);
  ret = ret * 31 + std::hash<int>()(start_cpu_id);
  return ret;
}

Node::Node(void)
: guid(0), ptr(NULL)
{}

std::string optype_to_string(OperatorType op_type)
{
  switch (op_type) {
    case OP_INPUT:
      return "Input";
    case OP_WEIGHT:
      return "Weight";
    case OP_NOOP:
      return "Noop";
    case OP_CONV2D:
      return "Conv";
    case OP_DROPOUT:
      return "Dropout";
    case OP_EMBEDDING:
      return "Embedding";
    case OP_LINEAR:
      return "Linear";
    case OP_POOL2D:
      return "Pool";
    case OP_RELU:
      return "Relu";
    case OP_SIGMOID:
      return "Sigmoid";
    case OP_TANH:
      return "TanH";
    case OP_BATCHNORM:
      return "Batchnorm";
    case OP_CONCAT:
      return "Concat";
    case OP_SPLIT:
      return "Split";
    case OP_RESHAPE:
      return "Reshape";
    case OP_TRANSPOSE:
      return "Transpose";
    case OP_EW_ADD:
      return "Add";
    case OP_EW_MUL:
      return "Mul";
    case OP_MATMUL:
      return "MatMul";
    case OP_MUL:
      return "Mul";
    case OP_ENLARGE:
      return "Enlarge";
    case OP_SQUEEZE:
      return "Squeeze";
    case OP_UNSQUEEZE:
      return "Unsqueeze";
    case OP_EW_SUB:
      return "Sub";
    case OP_EW_DIV:
      return "Div";
    case OP_EW_EQUAL:
      return "Equal";
    case OP_EW_GREATER:
      return "Greater";
    case OP_EW_LESS:
      return "Less";
    case OP_EW_MAX:
      return "Max";
    case OP_EW_MIN:
      return "Min";
    case OP_REDUCE_ARGMAX:
      return "ArgMax";
    case OP_REDUCE_ARGMIN:
      return "ArgMin";
    case OP_REDUCE_MAX:
      return "ReduceMax";
    case OP_REDUCE_MEAN:
      return "ReduceMean";
    case OP_REDUCE_MIN:
      return "ReduceMin";
    case OP_REDUCE_PROD:
      return "ReduceProd";
    case OP_REDUCE_SUM:
      return "ReduceSum";
    case OP_PAD:
      return "Pad";
    case OP_SHAPE:
      return "Shape";
    case OP_SIZE:
      return "Size";
    case OP_TOPK:
      return "TopK";
    case OP_WHERE:
      return "Where";
    case OP_CEIL:
      return "Ceil";
    case OP_CAST:
      return "Cast";
    case OP_EXP:
      return "Exp";
    case OP_ROUND:
      return "Round";
    case OP_LOG:
      return "Log";
    case OP_LOGICAL_NOT:
      return "Not";
    case OP_SQRT:
      return "Sqrt";
    case OP_LEAKYRELU:
      return "LeakyRelu";
    case OP_SLICE:
      return "Slice";
    case OP_RESIZE:
      return "Resize";
    case OP_SOFTMAX:
      return "Softmax";
    case OP_MULTIHEAD_ATTENTION:
      return "MultiHeadAttn";
    case OP_REPARTITION:
      return "Partition";
    case OP_REPLICATE:
      return "Replicate";
    case OP_REDUCTION:
      return "Reduction";
    case OP_COMBINE:
      return "Combine";
    case OP_FUSED_PARALLEL:
      return "FusedParallel";
    default:
      return "Unknown_" + std::to_string(op_type);
  }
}

std::string Node::op_to_string(const Op* op) const
{
  return optype_to_string(op->op_type);
}

Edge::Edge(void)
: srcOp(Node::INVALID_NODE), dstOp(Node::INVALID_NODE),
  srcIdx(-1), dstIdx(-1)
{}

Edge::Edge(const Node& _srcOp, const Node& _dstOp,
           int _srcIdx, int _dstIdx)
: srcOp(_srcOp), dstOp(_dstOp), srcIdx(_srcIdx), dstIdx(_dstIdx)
{}

bool Edge::operator==(const Edge& rhs) const
{
  if (srcOp != rhs.srcOp) return false;
  if (dstOp != rhs.dstOp) return false;
  if (srcIdx != rhs.srcIdx) return false;
  if (dstIdx != rhs.dstIdx) return false;
  return true;
}

SearchHelper::SearchHelper(FFModel *model)
  : model(model)
{ }

template <typename T>
T SearchHelper::execute_sequence_split(
    std::unique_ptr<Graph> const &pre_graph,
    std::unique_ptr<Graph> const &post_graph,
    NodeAssignment const &source,
    NodeAssignment const &sink,
    MachineResource const &resources,
    SequenceSplit const &bn) const 
{
  return sequence_cost<T>(
      this->graph_cost<T>(pre_graph.get(), source, bn, resources, true),
      this->graph_cost<T>(post_graph.get(), bn, sink, resources, false)
  );
}

template <typename T>
T SearchHelper::find_optimal_sequence_graph_time(
  Graph const *g,
  Node const &bn_node,
  NodeAssignment const &source,
  NodeAssignment const &sink,
  MachineResource const &resources
) const {
  std::unique_ptr<Graph> pre_graph;
  std::unique_ptr<Graph> post_graph;
  std::tie(pre_graph, post_graph) = g->split_at_node(bn_node);

  T optimal = this->infinity<T>();

  std::vector<MachineView> valid_views = this->get_valid_machine_views(bn_node.ptr, resources);
  // A Corner Case:
  // If bn_node is a parallel_op and an input to sink_node,
  // Add sink_node's view to the list, since sink_node's view
  // may not be a valid view for resources, but UniFlow support
  // this case since parallel_op does not trigger computation
  if (bn_node.ptr->is_parallel_op()) {
    bool found = false;
    const auto& inList = g->inEdges.find(sink.node)->second;
    for (const auto& e : inList) {
      if (e.srcOp == bn_node) {
        found = true;
        break;
      }
    }
    if (found) {
      for (int j = 0; j < bn_node.ptr->numOutputs; j++)
        if (!bn_node.ptr->outputs[j]->is_valid_machine_view(sink.view))
          found = false;
    }
    if (found) {
      valid_views.push_back(sink.view);
    }
  }

  if (valid_views.empty()) {
    return optimal;
  }

  float optimal_cost = std::numeric_limits<float>::infinity();
  MachineView best_view;

  for (MachineView const &bn_view : valid_views) {
    float cost = this->execute_sequence_split<float>(
        pre_graph,
        post_graph,
        source,
        sink,
        resources,
        {bn_node, bn_view}
    );

    if (cost < optimal_cost) {
      best_view = bn_view;
      optimal_cost = cost;
    }
  }

  if (optimal_cost != std::numeric_limits<float>::infinity()) {
    optimal = this->execute_sequence_split<T>(
        pre_graph,
        post_graph,
        source,
        sink,
        resources,
        {bn_node, best_view}
    );
  }

  check_matches_graph<T>(g, optimal, sink.node);

  return optimal;
}

Realm::LoggerMessage SearchHelper::debug() const {
  Realm::LoggerMessage msg = log_dp.debug();
  msg << this->depth;
  for (int i = 0; i < this->depth; i++) {
    msg << "  ";
  }

  return msg;
}

template <typename T>
T SearchHelper::execute_nonsequence_split(
  std::unique_ptr<Graph> const &first_graph,
  std::unique_ptr<Graph> const &second_graph,
  NodeAssignment const &source,
  NodeAssignment const &sink,
  MachineResource const &resources,
  NonsequenceSplit const &split) const 
{
  Graph const *first = first_graph.get();
  Graph const *second = second_graph.get();
  if (split.flip_graphs) {
    std::swap(first, second);
  }
  switch (split.type) {
    case SplitType::SEQUENTIAL:
      this->debug() << "Exploring sequential nonsequence split";
      return sequence_cost<T>(
          this->graph_cost<T>(first, source, sink, resources, false),
          this->graph_cost<T>(second, source, sink, resources, false)
      );
    case SplitType::VERTICAL:
    {
      this->debug() << "Exploring vertical nonsequence split (" << split.param << ", " << split.flip_graphs << ")";
      MachineResource firstRes = resources, 
                      secondRes = resources;
      firstRes.num_nodes = split.param;
      secondRes.num_nodes = resources.num_nodes - split.param;
      secondRes.start_gpu_id = resources.start_gpu_id + resources.all_gpus_per_node * split.param;

      return parallel_cost<T>(
          this->graph_cost<T>(first, source, sink, firstRes, false),
          this->graph_cost<T>(second, source, sink, secondRes, false)
      );
    }
    case SplitType::HORIZONTAL: 
    {
      this->debug() << "Exploring horizontal nonsequence split (" << split.param << ", " << split.flip_graphs << ")";
      MachineResource firstRes = resources, 
                      secondRes = resources;
      firstRes.available_gpus_per_node = split.param;
      secondRes.available_gpus_per_node = resources.available_gpus_per_node - split.param;
      secondRes.start_gpu_id = resources.start_gpu_id + split.param;

      return parallel_cost<T>(
          this->graph_cost<T>(first, source, sink, firstRes, false),
          this->graph_cost<T>(second, source, sink, secondRes, false)
      );
    }
    default:
      assert(false);
  }
}

/*static*/
NonsequenceSplit NonsequenceSplit::sequential() {
  NonsequenceSplit s;
  s.type = SplitType::SEQUENTIAL;
  s.flip_graphs = false;

  return s;
}

/*static*/
NonsequenceSplit NonsequenceSplit::vertical(int param, bool flip_graphs) {
  NonsequenceSplit s;
  s.type = SplitType::VERTICAL;
  s.param = param;
  s.flip_graphs = flip_graphs;

  return s;
}

/*static*/
NonsequenceSplit NonsequenceSplit::horizontal(int param, bool flip_graphs) {
  NonsequenceSplit s;
  s.type = SplitType::HORIZONTAL;
  s.param = param;
  s.flip_graphs = flip_graphs;

  return s;
}

template <typename T>
T SearchHelper::find_optimal_nonsequence_graph_time(
  Graph const *g,
  NodeAssignment const &source,
  NodeAssignment const &sink,
  MachineResource const &resources
) const {
  std::unique_ptr<Graph> first_graph;
  std::unique_ptr<Graph> second_graph;
  std::tie(first_graph, second_graph) = g->split_horizontal(source.node, sink.node);

  std::vector<NonsequenceSplit> potential_splits; 

  for (int i = 1; i < resources.num_nodes; i++) {
    potential_splits.push_back(NonsequenceSplit::vertical(i, false));
    potential_splits.push_back(NonsequenceSplit::vertical(i, true));
  }
  for (int i = 1; i < resources.available_gpus_per_node; i++) {
    potential_splits.push_back(NonsequenceSplit::horizontal(i, false));
    potential_splits.push_back(NonsequenceSplit::horizontal(i, true));
  }
  
  NonsequenceSplit best_split = NonsequenceSplit::sequential();
  float best_cost = this->execute_nonsequence_split<float>(
      first_graph, 
      second_graph, 
      source, 
      sink, 
      resources, 
      best_split
  );
  for (NonsequenceSplit const &split : potential_splits) {
    float cost = this->execute_nonsequence_split<float>(
        first_graph,
        second_graph,
        source, 
        sink,
        resources,
        split
    );
    this->debug() << "Found cost: " << cost;

    if (cost < best_cost) { 
      best_cost = cost;
      best_split = split;
    }
  }
  
  switch (best_split.type) {
    case SplitType::SEQUENTIAL:
      this->debug() << "Best split: SEQUENTIAL";
      break;
    case SplitType::VERTICAL:
      this->debug() << "Best split: VERTICAL(" << best_split.param << ", " << best_split.flip_graphs << ")";
      break;
    case SplitType::HORIZONTAL:
      this->debug() << "Best split: HORIZONTAL(" << best_split.param << ", " << best_split.flip_graphs << ")";
      break;
  }
  T optimal = this->execute_nonsequence_split<T>(
      first_graph,
      second_graph,
      source, 
      sink,
      resources,
      best_split
  );

  check_matches_graph<T>(g, optimal, sink.node);

  return optimal;
}

Graph::Graph(FFModel* _model)
: model(_model), search(_model->search)
{
}

void Graph::add_edge(const Node& srcOp,
                     const Node& dstOp,
                     int srcIdx,
                     int dstIdx)
{
  if (inEdges.find(dstOp) == inEdges.end()) {
    inEdges[dstOp];
  }
  if (outEdges.find(srcOp) == outEdges.end()) {
    outEdges[srcOp];
  }
  Edge e(srcOp, dstOp, srcIdx, dstIdx);
  inEdges[srcOp];
  outEdges[dstOp];
  inEdges[dstOp].insert(e);
  outEdges[srcOp].insert(e);
}

void Graph::add_edge(const Edge& e)
{
  inEdges[e.srcOp];
  outEdges[e.dstOp];

  inEdges[e.dstOp].insert(e);
  outEdges[e.srcOp].insert(e);
}

void Graph::remove_edge(const Edge& e)
{
  assert(outEdges[e.srcOp].find(e) != outEdges[e.srcOp].end());
  assert(inEdges[e.dstOp].find(e) != inEdges[e.dstOp].end());
  assert(outEdges[e.srcOp].erase(e) == 1);
  assert(inEdges[e.dstOp].erase(e) == 1);
  if ((outEdges[e.srcOp].size() == 0) && (inEdges[e.srcOp].size() == 0)) {
    outEdges.erase(e.srcOp);
    inEdges.erase(e.srcOp);
  }
  if ((outEdges[e.dstOp].size() == 0) && (inEdges[e.dstOp].size() == 0)) {
    outEdges.erase(e.dstOp);
    inEdges.erase(e.dstOp);
  }
}

bool Graph::has_edge(const Node& srcOp,
                     const Node& dstOp,
                     int srcIdx,
                     int dstIdx)
{
  Edge e(srcOp, dstOp, srcIdx, dstIdx);
  return (inEdges[dstOp].find(e) != inEdges[dstOp].end());
}

bool Graph::has_edge(const Edge& e)
{
  return (inEdges[e.dstOp].find(e) != inEdges[e.dstOp].end());
}

void Graph::print(void) const
{
  log_graph.print("Printing in-edge graph...");
  for (const auto& it : inEdges) {
    if (it.first.guid == 0) continue;
    log_graph.print("	guid(%zu) type(%s): ", it.first.guid,
                    optype_to_string(it.first.ptr->op_type).data());
    const std::unordered_set<Edge>& list = it.second;
    for (const auto& it2 : list) {
      Edge e = it2;
      log_graph.print("         inEdge(guid(%zu) idx(%d))",
                      e.srcOp.guid, e.srcIdx);
    }
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
  log_graph.print("Printing out-edge graph...");
  for (const auto& it : outEdges) {
    if (it.first.guid == 0) continue;
    log_graph.print("	guid(%zu) type(%d): ", it.first.guid,
                    it.first.ptr->op_type);
    const std::unordered_set<Edge>& list = it.second;
    for (const auto& it2 : list) {
      Edge e = it2;
      log_graph.print("         outEdge(guid(%zu) idx(%d))",
                      e.dstOp.guid, e.dstIdx);
    }
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
  std::unordered_map<Node, int> todos;
  std::vector<Node> opList;
  for (const auto& it : inEdges) {
    const auto& inList = it.second;
    todos[it.first] = (int)inList.size();
    if (todos[it.first] == 0)
      opList.push_back(it.first);
  }
  #ifdef DEADCODE
  for (const auto& it : outEdges) {
    if (inEdges.find(it.first) == inEdges.end()) {
      opList.push_back(it.first);
    }
  }
  #endif
  size_t i = 0;
  while (i < opList.size()) {
    Node op = opList[i++];
    const auto& outList = outEdges[op];
    for (const auto& it2 : outList) {
      todos[it2.dstOp] --;
      if (todos[it2.dstOp] == 0) {
        opList.push_back(it2.dstOp);
      }
    }
  }
  return (opList.size() < inEdges.size());
}

bool Graph::check_correctness(void)
{
  bool okay = true;
  for (auto it = outEdges.begin(); it != outEdges.end(); it++) {
    const auto& list = it->second;
    for (auto it2 = list.begin(); it2 != list.end(); it2++) {
      Edge e = *it2;
      if (!has_edge(e)) assert(false);
      if (e.srcOp.ptr == NULL) continue;
      Tensor srcTensor = e.srcOp.ptr->outputs[e.srcIdx];
      Tensor dstTensor = e.dstOp.ptr->inputs[e.dstIdx];
      if (srcTensor->num_dims != dstTensor->num_dims) assert(false);
      for (int i = 0; i < srcTensor->num_dims; i++) {
        assert(srcTensor->dims[i] == dstTensor->dims[i]);
      }
    }
  }
  return okay;
}

std::vector<MachineView> SearchHelper::get_valid_machine_views(const Op* op, const MachineResource& resource) const
{
  std::vector<MachineView> const *cached_op_views = NULL;
  std::vector<MachineView> valid_views;

  const auto& iter = cached_operator_valid_views.find(op->op_guid);
  if (iter != cached_operator_valid_views.end()) {
    cached_op_views = iter->second.get();
  } else {
    auto to_cache = std::unique_ptr<std::vector<MachineView>>(new std::vector<MachineView>());
    for (size_t i = 0; i < this->model->all_valid_views.size(); i++) {
      bool valid = true;
      for (int j = 0; j < op->numOutputs; j++)
        if (!op->outputs[j]->is_valid_machine_view(this->model->all_valid_views[i]))
          valid = false;
      if (valid)
        to_cache->push_back(this->model->all_valid_views[i]);
    }
    cached_operator_valid_views[op->op_guid] = std::move(to_cache);
    cached_op_views = cached_operator_valid_views.at(op->op_guid).get();
  }
  for (size_t i = 0; i < cached_op_views->size(); i++) {
    MachineView view = (*cached_op_views)[i];
    if (view.device_type == MachineView::GPU)
      view.start_device_id = resource.start_gpu_id;
    else if (view.device_type == MachineView::CPU)
      view.start_device_id = resource.start_cpu_id;
    else
      assert(false);
    if (resource.is_valid_machine_view(view))
      valid_views.push_back(view);
  }
  return valid_views;
}

void FFModel::register_all_machine_views(int num_nodes,
                                         int gpus_per_node,
                                         int cpus_per_node,
                                         std::vector<MachineView>& valid_views)
{
  // Data parallel views
  for (int i = 1; i <= num_nodes * gpus_per_node; i++)
    if (num_nodes * gpus_per_node % i == 0) {
      MachineView view;
      view.device_type = MachineView::GPU;
      view.ndims = 1;
      view.dim[0] = i;
      view.stride[0] = 1;
      view.start_device_id = 0;
      valid_views.push_back(view);
    }
}

Node Graph::find_bottleneck_node(const Node& sink_node, const Node& source_node) const
{
  using ::flexflow::graph::imm_post_dominators;
  using ::flexflow::graph::MultisourceGraphStructure;
  using ::flexflow::graph::GraphStructure;
  using ::flexflow::graph::roots;


  Node source(source_node);
  std::unordered_map<Node, Node> ipd;
  std::unordered_set<Node> graph_roots = roots(*this);
  if (source_node != Node::INVALID_NODE) {
    ipd = imm_post_dominators(*this);
  } else if (graph_roots.size() == 1) {
    ipd = imm_post_dominators(*this);
    source = *graph_roots.begin();
  } else {
    ipd = imm_post_dominators<Graph, MultisourceGraphStructure<Graph>>(*this);
  }

  Node bn_node = ipd.at(source);
  if (bn_node == source || bn_node == sink_node) {
    return Node::INVALID_NODE;
  }

  return bn_node;
}

std::pair<std::unique_ptr<Graph>, std::unique_ptr<Graph>> Graph::split_at_node(Node const &bottleneck) const {
  using ::flexflow::graph::topo_sort;

  auto first_graph = std::unique_ptr<Graph>(new Graph(this->model));
  auto second_graph = std::unique_ptr<Graph>(new Graph(this->model));

  std::unordered_set<Node> used_nodes;
  {
    std::vector<Node> topo_sorted;
    topo_sort(*this, &topo_sorted);

    for (auto const &node : topo_sorted) {
      if (node == bottleneck) {
        break;
      }

      used_nodes.insert(node);
    }
    used_nodes.insert(bottleneck);

    assert (used_nodes.size() < topo_sorted.size());
  }

  for (const auto& it : this->inEdges) {
    const auto& inList = it.second;
    if (used_nodes.find(it.first) != used_nodes.end()) {
      // Add all in-edges of used_nodes in to the first_graph
      for (const auto& it2 : inList) {
        first_graph->add_edge(it2);
      }
    } else {
      // Add all in-edges of not_used_nodes into the second_graph
      for (const auto& it2 : inList) {
        second_graph->add_edge(it2);
      }
    }
  }

  return {
    std::move(first_graph),
    std::move(second_graph)
  };
}

std::pair<std::unique_ptr<Graph>, std::unique_ptr<Graph>>
Graph::split_horizontal(Node const &source_node, Node const &sink_node) const
{
  auto first_graph = std::unique_ptr<Graph>(new Graph(this->model));
  auto second_graph = std::unique_ptr<Graph>(new Graph(this->model));
  Node bn_node = Node::INVALID_NODE;
  // Find sink_node's first input
  {
    const auto& inList = this->inEdges.find(sink_node)->second;
    int minIdx = MAX_NUM_INPUTS;
    for (const auto& it2 : inList) {
      //if (it2.dstIdx != 0) continue;
      //if (it2.weightEdge) continue;
      if (it2.dstIdx < minIdx) {
        minIdx = it2.dstIdx;
        bn_node = it2.srcOp;
      }
    }
  }
  assert(bn_node != Node::INVALID_NODE);
  std::unordered_set<Node> used_nodes;
  std::vector<Node> queue;
  queue.push_back(bn_node);
  used_nodes.insert(bn_node);
  size_t i = 0;
  while (i < queue.size()) {
    Node node = queue[i++];
    const auto& inList = this->inEdges.find(node)->second;
    for (const auto& it2 : inList) {
      if (used_nodes.find(it2.srcOp) == used_nodes.end()) {
        used_nodes.insert(it2.srcOp);
        queue.push_back(it2.srcOp);
      }
    }
  }
  for (const auto& it : this->inEdges) {
    if (it.first == sink_node) continue;
    const auto& inList = it.second;
    if (used_nodes.find(it.first) != used_nodes.end()) {
      // Add all in-edges of used_nodes in to the first_graph
      for (const auto& e : inList) {
        first_graph->add_edge(e);
      }
    } else {
      // Add all in-edges of not_used_nodes into the second_graph
      for (const auto& e : inList) {
        second_graph->add_edge(e);
      }
    }
  }
  // Split sink_node's inedges between the two graphs
  {
    const auto& inList = this->inEdges.find(sink_node)->second;
    for (const auto& e : inList) {
      if (used_nodes.find(e.srcOp) != used_nodes.end()) {
        first_graph->add_edge(e);
      } else {
        second_graph->add_edge(e);
      }
    }
  }
  // Assert there must be at least on sink_source's inEdges in the second graph
  assert(second_graph->inEdges.find(sink_node) != second_graph->inEdges.end());

  return {std::move(first_graph), std::move(second_graph)};
}

float FFModel::graph_cost(const Graph* graph,
                          const Node& sink_node,
                          const MachineView& sink_view,
                          const Node& source_node,
                          const MachineView& source_view,
                          const MachineResource& resources,
                          bool include_sink_compute_time,
                          bool constructing_optimal_view)
{
  assert (!graph->inEdges.empty());

  return this->search->graph_cost<float>(
      graph,
      { source_node, source_view },
      { sink_node, sink_view },
      resources,
      include_sink_compute_time
  );
}

void FFModel::construct_optimal_view(const Graph *graph,
                                     const Node& sink_node,
                                     const MachineView& sink_view,
                                     const Node& source_node,
                                     const MachineView& source_view,
                                     const MachineResource& resources,
                                     bool include_sink_compute_time,
                                     float optimal_cost,
                                     std::unordered_map<Node, MachineView>& optimal_views)
{
  GraphCostResult result = this->search->graph_cost<GraphCostResult>(
      graph,
      { source_node, source_view },
      { sink_node, sink_view },
      resources,
      include_sink_compute_time
  );

  optimal_views.insert(result.views.begin(), result.views.end());
}


GraphCostResult GraphCostResult::invalid() {
  return {std::numeric_limits<float>::infinity(), {}};
}

bool GraphCostResult::operator<(GraphCostResult const &other) const {
  return this->cost < other.cost;
}

template <>
GraphCostResult sequence_cost<GraphCostResult>(GraphCostResult const &first, GraphCostResult const &second) {
  GraphCostResult result;
  result.cost = first.cost + second.cost;
  result.views.insert(first.views.cbegin(), first.views.cend());
  result.views.insert(second.views.cbegin(), second.views.cend());

  return result;
}

template <>
float sequence_cost<float>(float const &first, float const &second) {
  return first + second;
}

template <>
GraphCostResult parallel_cost<GraphCostResult>(GraphCostResult const &first, GraphCostResult const &second) {
  GraphCostResult result;
  result.cost = std::max(first.cost, second.cost);
  result.views.insert(first.views.cbegin(), first.views.cend());
  result.views.insert(second.views.cbegin(), second.views.cend());

  return result;
}

template <>
float parallel_cost<float>(float const &first, float const &second) {
  return std::max(first, second);
}

template <>
bool SearchHelper::is_invalid<float>(float const &cost) const {
  return cost == std::numeric_limits<float>::infinity();
}

template <>
bool SearchHelper::is_invalid<GraphCostResult>(GraphCostResult const &cost) const {
  return cost.cost == std::numeric_limits<float>::infinity();
}

template <>
void SearchHelper::check_matches_graph<GraphCostResult>(Graph const *g, GraphCostResult const &r, Node const &sink) const {
  using ::flexflow::graph::nodes;

  if (this->is_invalid(r)) {
    return;
  }

  std::unordered_set<Node> g_nodes = nodes(*g);
  g_nodes.erase(sink);

  std::unordered_set<Node> r_nodes;
  for (auto const &kv : r.views) {
    r_nodes.insert(kv.first);
  }

  assert( g_nodes == r_nodes );
}

template <>
void SearchHelper::check_matches_graph<float>(Graph const *g, float const &r, Node const &sink) const { }

template <>
std::pair<bool, float> SearchHelper::try_get_cost_from_cache<float>(size_t hash) const {
  if (this->cached_graph_costs.find(hash) == this->cached_graph_costs.end()) {
    return {false, std::numeric_limits<float>::infinity()};
  } else {
    return {true, this->cached_graph_costs.at(hash)};
  }
}

template <>
std::pair<bool, GraphCostResult> SearchHelper::try_get_cost_from_cache<GraphCostResult>(size_t hash) const {
  return {false, GraphCostResult::invalid()};
}

template <>
void SearchHelper::try_cache_result<float>(size_t hash, float const &value) const {
  this->debug() << "cached_graph_costs[" << hash << "=" << value << "]";
  this->cached_graph_costs[hash] = value;
}

template <>
void SearchHelper::try_cache_result<GraphCostResult>(size_t hash, GraphCostResult const &value) const {
  this->debug() << "cached_graph_costs[" << hash << "=" << value.cost << "]";
  this->cached_graph_costs[hash] = value.cost;
}

template <>
float SearchHelper::infinity<float>() const {
  return std::numeric_limits<float>::infinity();
}

template <>
GraphCostResult SearchHelper::infinity<GraphCostResult>() const {
  return { std::numeric_limits<float>::infinity(), {}};
}

template <>
float SearchHelper::empty<float>() const {
  return 0.0f;
}

template <>
GraphCostResult SearchHelper::empty<GraphCostResult>() const {
  return { 0.0f, {} };
}

template <typename T>
T SearchHelper::estimate_xfer_cost(
    Graph const *graph, NodeAssignment const &source, NodeAssignment const &sink) const {
  T result = this->empty<T>();

  if (source.node == Node::INVALID_NODE) {
    // if source is an invalid node, then the source node of the graph is an input node
    Node real_source = Node::INVALID_NODE;
    for (auto const &kv : graph->inEdges) {
      if (kv.first != sink.node) {
        real_source = kv.first;
        break;
      }
    }
    this->add_operator_cost<T>(
        {real_source, MachineView::NO_VIEW}, // the machine view of an input node is currently ignored
        0.0f,
        &result
    );
  } else {
    const auto& inList = graph->inEdges.find(sink.node)->second;
    float op_cost = 0.0f;
    for (const auto& it2 : inList) {
      assert(it2.srcOp == source.node);
      assert(sink.node.ptr->inputs[it2.dstIdx]->is_valid_machine_view(source.view));

      op_cost += this->model->simulator->estimate_xfer_cost(sink.node.ptr, it2.srcIdx, source.view, sink.view);
    }
    this->add_operator_cost<T>(source, op_cost, &result);
  }

  return result;
}

template <>
void SearchHelper::add_operator_cost<float>(NodeAssignment const &node, float node_cost, float *cost) const {
  *cost += node_cost;
}

template <>
void SearchHelper::add_operator_cost<GraphCostResult>(NodeAssignment const &node, float node_cost, GraphCostResult *cost) const {
  cost->cost += node_cost;
  cost->views[node.node] = node.view;
}

template <typename T>
T SearchHelper::graph_cost(const Graph* graph,
                          const NodeAssignment& source,
                          const NodeAssignment& sink,
                          const MachineResource& resources,
                          bool include_sink_compute_time) const
{
  this->depth++;
  this->debug() << "sink(" << sink.node.guid << ") "
                 << "sink.view(" << sink.view.ndims << " " << sink.view.start_device_id << " " << sink.view.dim[0] << ") "
                 << "source(" << source.node.guid << ") "
                 << "source.view(" << source.view.ndims << " " << source.view.start_device_id << " " << source.view.dim[0] << ") "
                 << "resources(" << resources.num_nodes << " " << resources.start_gpu_id << " " << resources.available_gpus_per_node << ")";
  if (this->model->config.profiling) {
    graph->print();
  }

  assert(graph->inEdges.find(sink.node) != graph->inEdges.end());
  if (source.node != Node::INVALID_NODE)
    assert(graph->outEdges.find(source.node) != graph->outEdges.end());

  size_t hash = dp_state_hash(graph, sink.node, sink.view, source.node, source.view, resources);
  log_dp.spew("hash = %zu", hash);

  T result;

  std::pair<bool, T> from_cache = this->try_get_cost_from_cache<T>(hash);
  if (from_cache.first) {
    // cached_graph_costs does not include sink_compute_time
    result = from_cache.second;
  } else {
    if (graph->inEdges.size() <= 2) {
      result = this->estimate_xfer_cost<T>(graph, source, sink);
    } else {
      Node bn_node = graph->find_bottleneck_node(sink.node, source.node);
      if (bn_node != Node::INVALID_NODE) {
        // We found a bottleneck node
        this->debug() << "Found bn_node = " << bn_node.guid;

        result = this->find_optimal_sequence_graph_time<T>(
          graph,
          bn_node,
          { source.node, source.view },
          { sink.node, sink.view },
          resources
        );
      } else {
        // sink node must have multiple branches
        // otherwise we should not be here
        assert(graph->inEdges.find(sink.node)->second.size() > 1);

        result = this->find_optimal_nonsequence_graph_time<T>(
          graph,
          { source.node, source.view },
          { sink.node, sink.view },
          resources
        );
      }
    }

    this->try_cache_result<T>(hash, result);
  }

  check_matches_graph<T>(graph, result, sink.node);

  if (include_sink_compute_time) {
    CostMetrics metrics = this->model->simulator->measure_operator_cost(sink.node.ptr, sink.view);
    this->add_operator_cost<T>(sink, metrics.forward_time + metrics.backward_time + metrics.sync_time, &result);
  }

  this->depth--;
  return result;
}

float Graph::optimal_cost() const {
  return this->generic_optimal_cost<float>();
}

std::unordered_map<Node, MachineView> Graph::optimal_views() const {
  return this->generic_optimal_cost<GraphCostResult>().views;
}

Graph Graph::reduced() const {
  using ::flexflow::graph::BasicGraph;
  using ::flexflow::graph::transitive_reduction;
  using ::flexflow::graph::get_edges;

  BasicGraph<Node> transitive_skeleton = transitive_reduction(*this);

  Graph reduced_graph(this->model);

  for (Edge const &e : get_edges(*this)) {
    if (transitive_skeleton.has_edge(e.srcOp, e.dstOp)) {
      reduced_graph.add_edge(e);
    }
  }

  return reduced_graph;
}

template <typename T>
T Graph::generic_optimal_cost() const
{
  using ::flexflow::graph::leaves;

  Graph reduced_graph = this->reduced();

  // Find sink_nodes
  // i.e., nodes with no out edge
  std::unordered_set<Node> sink_nodes = leaves(reduced_graph);
  assert (sink_nodes.size() == 1);

  Node sink_node = *sink_nodes.cbegin();

  MachineResource resource;
  resource.num_nodes = model->config.numNodes;
  resource.all_cpus_per_node = model->config.cpusPerNode;
  resource.all_gpus_per_node = model->config.workersPerNode;
  resource.available_cpus_per_node = resource.all_cpus_per_node;
  resource.available_gpus_per_node = resource.all_gpus_per_node;
  resource.start_gpu_id = 0;
  resource.start_cpu_id = 0;

  std::vector<MachineView> valid_views = search->get_valid_machine_views(sink_node.ptr, resource);

  T optimal = search->infinity<T>();

  for (MachineView const &sink_view : valid_views) {
    T new_cost = search->graph_cost<T>(
        &reduced_graph,
        {Node::INVALID_NODE, MachineView::NO_VIEW},
        {sink_node, sink_view},
        resource,
        true
    );
    if (new_cost < optimal) {
      optimal = new_cost;
    }
  }

  return optimal;
}

size_t Graph::hash(void) const
{
  // Graph hash should be additive and independent to the ordering of the nodes
  size_t total_hash = 0;
  for (const auto& it : inEdges) {
    const auto& inList = it.second;
    size_t node_hash = std::hash<size_t>()((size_t)it.first.ptr);
    for (const auto& e : inList) {
      size_t edge_hash = 17;
      edge_hash = edge_hash * 31 + std::hash<size_t>()((size_t)e.srcOp.ptr);
      edge_hash = edge_hash * 31 + std::hash<int>()(e.srcIdx);
      edge_hash = edge_hash * 31 + std::hash<int>()(e.dstIdx);
      node_hash *= edge_hash;
    }
    total_hash += node_hash;
  }
  return total_hash;
}

size_t dp_state_hash(const Graph* graph,
                     const Node& sink_node,
                     const MachineView& sink_view,
                     const Node& source_node,
                     const MachineView& source_view,
                     const MachineResource& resource)
{
  size_t key = graph->hash();
  key = key * 31 + std::hash<size_t>()((size_t)sink_node.ptr);
  key = key * 31 + sink_view.hash();
  key = key * 31 + std::hash<size_t>()((size_t)source_node.ptr);
  key = key * 31 + source_view.hash();
  key = key * 31 + resource.hash();
  return key;
}

GraphOptimalViewSerialized Graph::graph_optimize_task(const Task *task,
    const std::vector<PhysicalRegion> &regions,
    Context ctx, Runtime *runtime)
{
  FFModel* model = *((FFModel**) task->args);
  Memory gpu_mem = Machine::MemoryQuery(Machine::get_machine())
         .only_kind(Memory::GPU_FB_MEM).best_affinity_to(task->target_proc).first();
  MachineModel *machine;
  if (model->config.machine_model_version == 0) {
    machine = (MachineModel *) new SimpleMachineModel(model->config.numNodes, model->config.workersPerNode, gpu_mem.capacity());
  }
  else if (model->config.machine_model_version == 1 and !model->config.machine_model_file.empty()) {
    machine = (MachineModel *) new EnhancedMachineModel(model->config.machine_model_file, gpu_mem.capacity());
  }
  else {
    assert(false && "machine model creation error: currently only support machine-model-version = 0 or 1. When machine-model-version = 1, machine-model-file should not be empty.");
  }
  // Assume this task is running on GPU0
  Simulator* simulator = new Simulator(model, model->handlers[0], gpu_mem, machine);
  model->simulator = simulator;
  Graph* best_graph = NULL;
  std::unordered_map<Node, MachineView> optimal_views;
  model->graph_optimize(model->config.search_budget,
                        model->config.only_data_parallel,
                        best_graph, optimal_views);
  Serializer sez;
  // FIrst serialize graph
  sez.serialize(best_graph->inEdges.size());
  std::unordered_map<Node, int> todos;
  std::vector<Node> opList;
  for (const auto& it : best_graph->inEdges) {
    const auto& inList = it.second;
    todos[it.first] = (int)inList.size();
    if (todos[it.first] == 0)
      opList.push_back(it.first);
  }
  size_t node_idx = 0;
  while (node_idx < opList.size()) {
    Node cur_node = opList[node_idx++];
    const auto& outList = best_graph->outEdges[cur_node];
    for (const auto& e : outList) {
      todos[e.dstOp] --;
      if (todos[e.dstOp] == 0) {
        opList.push_back(e.dstOp);
      }
    }
    const auto& inList = best_graph->inEdges[cur_node];
    sez.serialize(inList.size());
    for (const auto& e : inList) {
      sez.serialize(e.srcOp.guid);
      assert(e.dstOp.guid == cur_node.guid);
      sez.serialize(e.srcIdx);
      sez.serialize(e.dstIdx);
    }
    sez.serialize((size_t)10101010); // safe guard for the end of inedges
    const Op* op = cur_node.ptr;
    assert(op != NULL);
    sez.serialize(cur_node.guid);
    sez.serialize(op->op_type);
    switch (op->op_type) {
      case OP_INPUT:
      {
        assert(op->numOutputs == 1);
        sez.serialize(op->op_type);
        sez.serialize(op->outputs[0]->data_type);
        sez.serialize(op->outputs[0]->num_dims);
        for (int i = 0; i < op->outputs[0]->num_dims; i++)
          sez.serialize(op->outputs[0]->dims[i].size);
        break;
      }
      case OP_NOOP:
      {
        break;
      }
      case OP_CONCAT:
      {
        Concat* concat = (Concat*) op;
        sez.serialize(concat->axis);
        break;
      }
      case OP_EMBEDDING:
      {
        Embedding* embed = (Embedding*) op;
        sez.serialize(embed->num_entries);
        sez.serialize(embed->out_channels);
        sez.serialize(embed->aggr);
        break;
      }
      case OP_EW_ADD:
      {
        sez.serialize(op->op_type);
        break;
      }
      case OP_MULTIHEAD_ATTENTION:
      {
        MultiHeadAttention* attn = (MultiHeadAttention*) op;
        sez.serialize(attn->oProjSize);
        sez.serialize(attn->num_heads);
        sez.serialize(attn->qProjSize);
        sez.serialize(attn->vProjSize);
        sez.serialize(attn->dropout);
        sez.serialize(attn->bias);
        sez.serialize(attn->add_bias_kv);
        sez.serialize(attn->add_zero_attn);
        break;
      }
      case OP_SOFTMAX:
      {
        Softmax* softmax = (Softmax*) op;
        sez.serialize(softmax->dim);
        break;
      }
      case OP_REPARTITION:
      {
        Repartition* repart = (Repartition*) op;
        sez.serialize(repart->repartition_dim);
        sez.serialize(repart->repartition_degree);
        break;
      }
      case OP_REPLICATE:
      {
        Replicate* replicate = (Replicate*) op;
        sez.serialize(replicate->replicate_dim);
        sez.serialize(replicate->replicate_degree);
        break;
      }
      case OP_REDUCTION:
      {
        Reduction* reduction = (Reduction*) op;
        sez.serialize(reduction->reduction_dim);
        sez.serialize(reduction->reduction_degree);
        break;
      }
      case OP_COMBINE:
      {
        Combine* combine = (Combine*) op;
        sez.serialize(combine->combine_dim);
        sez.serialize(combine->combine_degree);
        break;
      }
      case OP_FUSED_PARALLEL:
      {
        FusedParallelOp* fused = (FusedParallelOp*) op;
        sez.serialize(fused->num_parallel_ops);
        for (int i = 0; i < fused->num_parallel_ops; i++)
          sez.serialize(fused->parallel_ops[i]);
        break;
      }
      default:
      {
        op->serialize(sez);
      }
    }
    sez.serialize((size_t)12345678); // safe guard for the end of an op
  }
  assert(node_idx == best_graph->inEdges.size());
  // Second, serialize optimal machine view
  sez.serialize(optimal_views.size());
  for (const auto & it : optimal_views) {
    sez.serialize((size_t) 98765432); // safe guard
    sez.serialize(it.first.guid);
    sez.serialize(it.second);
  }
  assert(sez.get_used_bytes() < GraphOptimalViewSerialized::buffer_size);
  GraphOptimalViewSerialized ret;
  ret.total_bytes = sez.get_used_bytes();
  memcpy(ret.data, sez.get_buffer(), ret.total_bytes);
  // Deallocate best_graph
  delete best_graph;
  return ret;
}

void FFModel::deserialize_graph_optimal_view(Deserializer& dez,
                                             Graph* graph,
                                             std::unordered_map<Node, MachineView>& optimal_views)
{
  //Deserializer dez(serialized.data, serialized.total_bytes);
  std::unordered_map<size_t, Node> guid_to_nodes;
  size_t num_nodes;
  dez.deserialize(num_nodes);
  //best_graph = new Graph(this);
  for (size_t node_idx = 0; node_idx < num_nodes; node_idx++) {
    Edge inedges[MAX_NUM_INPUTS];
    Tensor inputs[MAX_NUM_INPUTS];
    size_t num_inputs;
    dez.deserialize(num_inputs);
    for (size_t j = 0; j < num_inputs; j++) {
      size_t src_guid;
      int src_idx, dst_idx;
      dez.deserialize(src_guid);
      assert(guid_to_nodes.find(src_guid) != guid_to_nodes.end());
      dez.deserialize(src_idx);
      dez.deserialize(dst_idx);
      assert(dst_idx < (int)num_inputs);
      inedges[dst_idx].srcOp = guid_to_nodes[src_guid];
      inedges[dst_idx].srcIdx = src_idx;
      inedges[dst_idx].dstIdx = dst_idx;
      inputs[dst_idx] = inedges[dst_idx].srcOp.ptr->outputs[src_idx];
    }
    {
      size_t safecode;
      dez.deserialize(safecode);
      assert(safecode == 10101010);
    }
    Node node = Node::INVALID_NODE;
    size_t guid;
    OperatorType op_type;
    dez.deserialize(guid);
    dez.deserialize(op_type);
    switch(op_type) {
      case OP_INPUT:
      {
        assert(num_inputs == 0);
        int num_dims, dims[MAX_TENSOR_DIM];
        OperatorType op_type;
        dez.deserialize(op_type);
        DataType data_type;
        dez.deserialize(data_type);
        dez.deserialize(num_dims);
        for (int i = 0; i < num_dims; i++)
          dez.deserialize(dims[i]);
        Tensor t = create_tensor_legion_ordering(num_dims, dims, data_type);
        node.ptr = t->owner_op;
        node.guid = node_global_guid ++;
        break;
      }
      case OP_NOOP:
      {
        assert(num_inputs == 1);
        node = get_or_create_noop_node(inputs[0]);
        break;
      }
      case OP_CONCAT:
      {
        int axis;
        dez.deserialize(axis);
        node = get_or_create_concat_node(num_inputs, inputs, axis);
        break;
      }
      case OP_EMBEDDING:
      {
        assert(num_inputs == 1);
        AggrMode aggr;
        int num_entries, out_channels;
        dez.deserialize(num_entries);
        dez.deserialize(out_channels);
        dez.deserialize(aggr);
        node = get_or_create_embedding_node(inputs[0], num_entries, out_channels, aggr);
        break;
      }
      case OP_EW_ADD:
      {
        assert(num_inputs == 2);
        OperatorType op_type;
        dez.deserialize(op_type);
        node = get_or_create_element_binary_node(inputs[0], inputs[1], op_type);
        break;
      }
      case OP_CONV2D:
      { 
        node = Conv2D::deserialize(*this, dez, inputs, num_inputs);
        break;
      }
      case OP_POOL2D:
      {
        node = Pool2D::deserialize(*this, dez, inputs, num_inputs);
        break;
      }
      case OP_LINEAR:
      {
        node = Linear::deserialize(*this, dez, inputs, num_inputs);
        break;
      }
      case OP_EXP:
      case OP_SCALAR_MULTIPLY:
      case OP_RELU:
      case OP_SIGMOID:
      case OP_TANH:
      case OP_IDENTITY:
      case OP_GELU:
      case OP_ELU:
      {
        node = ElementUnary::deserialize(*this, dez, inputs, num_inputs);
        break;
      }
      case OP_FLAT:
      {
        node = Flat::deserialize(*this, dez, inputs, num_inputs);
        break;
      }
      case OP_MULTIHEAD_ATTENTION:
      {
        assert(num_inputs == 3);
        int embed_dim, num_heads, k_dim, v_dim;
        float dropout;
        bool bias, add_bias_kv, add_zero_attn;
        dez.deserialize(embed_dim);
        dez.deserialize(num_heads);
        dez.deserialize(k_dim);
        dez.deserialize(v_dim);
        dez.deserialize(dropout);
        dez.deserialize(bias);
        dez.deserialize(add_bias_kv);
        dez.deserialize(add_zero_attn);
        node = get_or_create_multihead_attn_node(inputs[0], inputs[1], inputs[2],
                                                 embed_dim, num_heads,
                                                 k_dim, v_dim, dropout,
                                                 bias, add_bias_kv, add_zero_attn);
        break;
      }
      case OP_SOFTMAX:
      {
        assert(num_inputs == 1);
        int softmax_dim;
        dez.deserialize(softmax_dim);
        node = get_or_create_softmax_node(inputs[0], softmax_dim);
        break;
      }
      case OP_COMBINE:
      {
        assert(num_inputs == 1);
        int combine_dim, combine_degree;
        dez.deserialize(combine_dim);
        dez.deserialize(combine_degree);
        node = get_or_create_combine_node(inputs[0], combine_dim,
                                          combine_degree);
        break;
      }
      case OP_REPARTITION:
      {
        assert(num_inputs == 1);
        int repartition_dim, repartition_degree;
        dez.deserialize(repartition_dim);
        dez.deserialize(repartition_degree);
        node = get_or_create_repartition_node(inputs[0], repartition_dim,
                                              repartition_degree);
        break;
      }
      case OP_REPLICATE:
      {
        assert(num_inputs == 1);
        int replicate_dim, replicate_degree;
        dez.deserialize(replicate_dim);
        dez.deserialize(replicate_degree);
        node = get_or_create_replicate_node(inputs[0], replicate_dim,
                                            replicate_degree);
        break;
      }
      case OP_REDUCTION:
      {
        assert(num_inputs == 1);
        int reduction_dim, reduction_degree;
        dez.deserialize(reduction_dim);
        dez.deserialize(reduction_degree);
        node = get_or_create_reduction_node(inputs[0], reduction_dim,
                                            reduction_degree);
        break;
      }
      case OP_FUSED_PARALLEL:
      {
        assert(num_inputs == 1);
        std::vector<ParallelOpInfo> parallel_ops;
        int num_parallel_ops;
        dez.deserialize(num_parallel_ops);
        for (int i = 0; i < num_parallel_ops; i++) {
          ParallelOpInfo info;
          dez.deserialize(info);
          parallel_ops.push_back(info);
        }
        node = get_or_create_fused_parallel_node(inputs[0], parallel_ops);
        break;
      }
      default:
      {
        fprintf(stderr, "The following operator type is currently not supported"
                " for graph deserialization: %s\n"
                "Report the issue to the FlexFlow developers",
                optype_to_string(op_type).c_str());
        assert(false && "Unsupported operator type");
      }
    }
    {
      size_t safecode;
      dez.deserialize(safecode);
      assert(safecode == 12345678);
    }
    guid_to_nodes[guid] = node;
    for (size_t i = 0; i < num_inputs; i++) {
      inedges[i].dstOp = node;
      graph->add_edge(inedges[i]);
    }
  }
  // Second, deserialize optimal machine view
  size_t num_views;
  dez.deserialize(num_views);
  for (size_t i = 0; i < num_views; i++) {
    size_t safecode, guid;
    MachineView view;
    dez.deserialize(safecode);
    assert(safecode == 98765432);
    dez.deserialize(guid);
    assert(guid_to_nodes.find(guid) != guid_to_nodes.end());
    dez.deserialize(view);
    optimal_views[guid_to_nodes[guid]] = view;
  }
  assert(dez.get_remaining_bytes() == 0);
  printf("Deserialized Views...\n");
  for (const auto& it : optimal_views) {
    printf("node[%zu]: type(%s) view(%d %d %d) ", it.first.guid,
           it.first.to_string().c_str(),
           it.second.ndims,
           it.second.dim[0],
           it.second.start_device_id);
    const auto& list = graph->inEdges.at(it.first);
    for (const auto& it2 : list) {
      Edge e = it2;
      printf(" inEdge(node(%zu) idx(%d))", e.srcOp.guid, e.srcIdx);
    }
    printf("\n");
  }
}

