/* Copyright 2023 CMU, Facebook, LANL, MIT, NVIDIA, and Stanford (alphabetical)
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
#include "flexflow/graph.h"
#include "flexflow/dominators.h"
#include "flexflow/ffconst_utils.h"
#include "flexflow/ops/aggregate.h"
#include "flexflow/ops/attention.h"
#include "flexflow/ops/batch_matmul.h"
#include "flexflow/ops/cast.h"
#include "flexflow/ops/concat.h"
#include "flexflow/ops/conv_2d.h"
#include "flexflow/ops/dropout.h"
#include "flexflow/ops/element_binary.h"
#include "flexflow/ops/element_unary.h"
#include "flexflow/ops/embedding.h"
#include "flexflow/ops/flat.h"
#include "flexflow/ops/gather.h"
#include "flexflow/ops/groupby.h"
#include "flexflow/ops/layer_norm.h"
#include "flexflow/ops/linear.h"
#include "flexflow/ops/noop.h"
#include "flexflow/ops/pool_2d.h"
#include "flexflow/ops/reduce.h"
#include "flexflow/ops/reshape.h"
#include "flexflow/ops/softmax.h"
#include "flexflow/ops/split.h"
#include "flexflow/ops/topk.h"
#include "flexflow/ops/transpose.h"
#include "flexflow/parallel_ops/combine.h"
#include "flexflow/parallel_ops/fused_parallel_op.h"
#include "flexflow/parallel_ops/partition.h"
#include "flexflow/parallel_ops/reduction.h"
#include "flexflow/parallel_ops/replicate.h"
#include "flexflow/utils/disjoint_set.h"
#include "legion.h"
#include "legion/legion_utilities.h"

namespace FlexFlow::PCG {

using namespace Legion;
using FlexFlow::MachineView;

LegionRuntime::Logger::Category log_graph("graph");
LegionRuntime::Logger::Category log_simplify("graph_simplify");

const Node Node::INVALID_NODE = Node();

Node::Node(void) : guid(0), ptr(NULL) {}

std::string Node::op_to_string(Op const *op) const {
  return get_operator_type_name(op->op_type);
}

Edge::Edge(void)
    : srcOp(Node::INVALID_NODE), dstOp(Node::INVALID_NODE), srcIdx(-1),
      dstIdx(-1) {}

Edge::Edge(Node const &_srcOp, Node const &_dstOp, int _srcIdx, int _dstIdx)
    : srcOp(_srcOp), dstOp(_dstOp), srcIdx(_srcIdx), dstIdx(_dstIdx) {}

bool Edge::operator==(Edge const &rhs) const {
  if (srcOp != rhs.srcOp) {
    return false;
  }
  if (dstOp != rhs.dstOp) {
    return false;
  }
  if (srcIdx != rhs.srcIdx) {
    return false;
  }
  if (dstIdx != rhs.dstIdx) {
    return false;
  }
  return true;
}

SearchHelper::SearchHelper(FFModel *model) : model(model) {
  this->logger = std::unique_ptr<RecursiveLogger>(new RecursiveLogger("DP"));
}

/**
 * @brief Combine results from sequential sub-problems.
 */
template <typename T>
T SearchHelper::execute_sequence_split(std::unique_ptr<Graph> const &pre_graph,
                                       std::unique_ptr<Graph> const &post_graph,
                                       NodeAssignment const &source,
                                       NodeAssignment const &sink,
                                       MachineResource const &resources,
                                       SequenceSplit const &bn) const {
  return sequence_cost<T>(
      this->graph_cost<T>(pre_graph.get(), source, bn, resources, true),
      this->graph_cost<T>(post_graph.get(), bn, sink, resources, false));
}

/**
 * @brief Starting point to get sequential split time cost.
 *
 * @tparam T float or GraphCostResult (or GraphCostResultWithMemory in memory
 * optimization)
 */
template <typename T>
T SearchHelper::find_optimal_sequence_graph_time(
    Graph const *g,
    Node const &bn_node,
    NodeAssignment const &source,
    NodeAssignment const &sink,
    MachineResource const &resources) const {
  std::unique_ptr<Graph> pre_graph;
  std::unique_ptr<Graph> post_graph;
  std::tie(pre_graph, post_graph) = g->split_at_node(bn_node);

  T optimal = this->infinity<T>();

  std::vector<MachineView> valid_views =
      this->get_valid_machine_views(bn_node.ptr, resources);
  // A Corner Case:
  // If bn_node is a parallel_op and an input to sink_node,
  // Add sink_node's view to the list, since sink_node's view
  // may not be a valid view for resources, but UniFlow support
  // this case since parallel_op does not trigger computation
  if (bn_node.ptr->is_parallel_op()) {
    bool found = false;
    auto const &inList = g->inEdges.find(sink.node)->second;
    for (auto const &e : inList) {
      if (e.srcOp == bn_node) {
        found = true;
        break;
      }
    }
    if (found) {
      for (int j = 0; j < bn_node.ptr->numOutputs; j++) {
        if (!bn_node.ptr->outputs[j]->is_valid_machine_view(sink.view)) {
          found = false;
        }
      }
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
        pre_graph, post_graph, source, sink, resources, {bn_node, bn_view});

    if (cost < optimal_cost) {
      best_view = bn_view;
      optimal_cost = cost;
    }
  }

  if (optimal_cost != std::numeric_limits<float>::infinity()) {
    optimal = this->execute_sequence_split<T>(
        pre_graph, post_graph, source, sink, resources, {bn_node, best_view});
  }

  check_matches_graph<T>(g, optimal, sink.node);

  return optimal;
}

void SearchHelper::clear_cache() {
  cached_graph_costs.clear();
  cached_operator_valid_views.clear();
}

template <typename T>
T SearchHelper::execute_nonsequence_split(
    std::unique_ptr<Graph> const &first_graph,
    std::unique_ptr<Graph> const &second_graph,
    NodeAssignment const &source,
    NodeAssignment const &sink,
    MachineResource const &resources,
    NonsequenceSplit const &split) const {
  Graph const *first = first_graph.get();
  Graph const *second = second_graph.get();
  if (split.flip_graphs) {
    std::swap(first, second);
  }
  switch (split.type) {
    case SplitType::SEQUENTIAL:
      this->logger->debug() << "Exploring sequential nonsequence split";
      return sequence_cost<T>(
          this->graph_cost<T>(first, source, sink, resources, false),
          this->graph_cost<T>(second, source, sink, resources, false));
    case SplitType::VERTICAL: {
      this->logger->debug() << "Exploring vertical nonsequence split ("
                            << split.param << ", " << split.flip_graphs << ")";
      MachineResource firstRes = resources, secondRes = resources;
      firstRes.num_nodes = split.param;
      secondRes.num_nodes = resources.num_nodes - split.param;
      secondRes.start_gpu_id =
          resources.start_gpu_id + resources.all_gpus_per_node * split.param;

      return parallel_cost<T>(
          this->graph_cost<T>(first, source, sink, firstRes, false),
          this->graph_cost<T>(second, source, sink, secondRes, false));
    }
    case SplitType::HORIZONTAL: {
      this->logger->debug() << "Exploring horizontal nonsequence split ("
                            << split.param << ", " << split.flip_graphs << ")";
      MachineResource firstRes = resources, secondRes = resources;
      firstRes.available_gpus_per_node = split.param;
      secondRes.available_gpus_per_node =
          resources.available_gpus_per_node - split.param;
      secondRes.start_gpu_id = resources.start_gpu_id + split.param;

      return parallel_cost<T>(
          this->graph_cost<T>(first, source, sink, firstRes, false),
          this->graph_cost<T>(second, source, sink, secondRes, false));
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
    MachineResource const &resources) const {
  std::unique_ptr<Graph> first_graph;
  std::unique_ptr<Graph> second_graph;
  std::tie(first_graph, second_graph) =
      g->split_horizontal(source.node, sink.node);

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
      first_graph, second_graph, source, sink, resources, best_split);
  for (NonsequenceSplit const &split : potential_splits) {
    float cost = this->execute_nonsequence_split<float>(
        first_graph, second_graph, source, sink, resources, split);
    this->logger->debug() << "Found cost: " << cost;

    if (cost < best_cost) {
      best_cost = cost;
      best_split = split;
    }
  }

  switch (best_split.type) {
    case SplitType::SEQUENTIAL:
      this->logger->debug() << "Best split: SEQUENTIAL";
      break;
    case SplitType::VERTICAL:
      this->logger->debug() << "Best split: VERTICAL(" << best_split.param
                            << ", " << best_split.flip_graphs << ")";
      break;
    case SplitType::HORIZONTAL:
      this->logger->debug() << "Best split: HORIZONTAL(" << best_split.param
                            << ", " << best_split.flip_graphs << ")";
      break;
  }
  T optimal = this->execute_nonsequence_split<T>(
      first_graph, second_graph, source, sink, resources, best_split);

  check_matches_graph<T>(g, optimal, sink.node);

  return optimal;
}

Graph::Graph(FFModel *_model) : model(_model), search(_model->search) {}

void Graph::add_edge(Node const &srcOp,
                     Node const &dstOp,
                     int srcIdx,
                     int dstIdx) {
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

void Graph::add_node(Node const &node) {
  inEdges[node];
  outEdges[node];
}

void Graph::add_edge(Edge const &e) {
  inEdges[e.srcOp];
  outEdges[e.dstOp];

  inEdges[e.dstOp].insert(e);
  outEdges[e.srcOp].insert(e);
}

void Graph::remove_edge(Edge const &e, bool remove_node_if_unused) {
  assert(outEdges[e.srcOp].find(e) != outEdges[e.srcOp].end());
  assert(inEdges[e.dstOp].find(e) != inEdges[e.dstOp].end());
  assert(outEdges[e.srcOp].erase(e) == 1);
  assert(inEdges[e.dstOp].erase(e) == 1);
  if (remove_node_if_unused) {
    if ((outEdges[e.srcOp].size() == 0) && (inEdges[e.srcOp].size() == 0)) {
      outEdges.erase(e.srcOp);
      inEdges.erase(e.srcOp);
    }
    if ((outEdges[e.dstOp].size() == 0) && (inEdges[e.dstOp].size() == 0)) {
      outEdges.erase(e.dstOp);
      inEdges.erase(e.dstOp);
    }
  }
}

bool Graph::has_edge(Node const &srcOp,
                     Node const &dstOp,
                     int srcIdx,
                     int dstIdx) const {
  Edge e(srcOp, dstOp, srcIdx, dstIdx);
  return this->has_edge(e);
}

bool Graph::has_edge(Edge const &e) const {
  if (inEdges.find(e.dstOp) == inEdges.end()) {
    return false;
  }
  if (inEdges.at(e.dstOp).find(e) == inEdges.at(e.dstOp).end()) {
    return false;
  }
  return true;
}

void Graph::print(void) const {
  log_graph.print("Printing in-edge graph...");
  for (auto const &it : inEdges) {
    if (it.first.guid == 0) {
      continue;
    }
    log_graph.print("	guid(%zu) type(%s): ",
                    it.first.guid,
                    get_operator_type_name(it.first.ptr->op_type).data());
    std::unordered_set<Edge> const &list = it.second;
    for (auto const &it2 : list) {
      Edge e = it2;
      log_graph.print(
          "         inEdge(guid(%zu) idx(%d))", e.srcOp.guid, e.srcIdx);
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
  for (auto const &it : outEdges) {
    if (it.first.guid == 0) {
      continue;
    }
    log_graph.print(
        "	guid(%zu) type(%d): ", it.first.guid, it.first.ptr->op_type);
    std::unordered_set<Edge> const &list = it.second;
    for (auto const &it2 : list) {
      Edge e = it2;
      log_graph.print(
          "         outEdge(guid(%zu) idx(%d))", e.dstOp.guid, e.dstIdx);
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

void Graph::print_dot() const {
  this->print_dot(std::cout);
}

void Graph::print_dot(std::ostream &s) const {
  using FlexFlow::PCG::Utils::export_as_dot;

  DotFile<Node> dot(s);

  export_as_dot(dot, *this, [](Node const &node) -> RecordFormatter {
    RecordFormatter rf;
    rf << node.to_string();
    tl::optional<RecordFormatter> sub_rf = node.ptr->as_dot();
    if (sub_rf.has_value()) {
      rf << sub_rf.value();
    }

    return rf;
  });
  s << std::endl;
}

bool Graph::has_loop(void) {
  std::unordered_map<Node, int> todos;
  std::vector<Node> opList;
  for (auto const &it : inEdges) {
    auto const &inList = it.second;
    todos[it.first] = (int)inList.size();
    if (todos[it.first] == 0) {
      opList.push_back(it.first);
    }
  }
  size_t i = 0;
  while (i < opList.size()) {
    Node op = opList[i++];
    auto const &outList = outEdges[op];
    for (auto const &it2 : outList) {
      todos[it2.dstOp]--;
      if (todos[it2.dstOp] == 0) {
        opList.push_back(it2.dstOp);
      }
    }
  }
  return (opList.size() < inEdges.size());
}

bool Graph::check_correctness(void) {
  bool okay = true;
  for (auto it = outEdges.begin(); it != outEdges.end(); it++) {
    auto const &list = it->second;
    for (auto it2 = list.begin(); it2 != list.end(); it2++) {
      Edge e = *it2;
      if (!has_edge(e)) {
        assert(false);
      }
      if (e.srcOp.ptr == NULL) {
        continue;
      }
      assert(e.srcOp != e.dstOp);
      ParallelTensor srcTensor = e.srcOp.ptr->outputs[e.srcIdx];
      ParallelTensor dstTensor = e.dstOp.ptr->inputs[e.dstIdx];
      if (srcTensor->num_dims != dstTensor->num_dims) {
        assert(false);
      }
      for (int i = 0; i < srcTensor->num_dims; i++) {
        assert(srcTensor->dims[i] == dstTensor->dims[i]);
      }
    }
  }
  return okay;
}

std::vector<MachineView> SearchHelper::get_valid_machine_views(
    Node const &node, MachineResource const &resource, bool log) const {
  this->logger->info() << "Getting valid machine views for "
                       << node.to_string();
  return this->get_valid_machine_views(node.ptr, resource, log);
}

std::vector<MachineView> SearchHelper::get_valid_machine_views(
    Op const *op, MachineResource const &resource, bool log) const {
  std::vector<MachineView> const *cached_op_views = NULL;
  std::vector<MachineView> valid_views;

  auto const &iter = cached_operator_valid_views.find(op->op_guid);
  if (iter != cached_operator_valid_views.end()) {
    cached_op_views = iter->second.get();
  } else {
    auto to_cache = std::unique_ptr<std::vector<MachineView>>(
        new std::vector<MachineView>());
    if (log) {
      this->logger->info() << "Considering a total of "
                           << this->model->all_valid_views.size()
                           << " potential valid views";
    }
    for (size_t i = 0; i < this->model->all_valid_views.size(); i++) {
      bool valid = true;
      for (int j = 0; j < op->numOutputs; j++) {
        if (!op->outputs[j]->is_valid_machine_view(
                this->model->all_valid_views[i])) {
          valid = false;
          {
            MachineView const &view = this->model->all_valid_views[i];
            std::ostringstream oss;
            oss << "[" << view.ndims << "](";
            for (int i = 0; i < view.ndims; i++) {
              oss << view.dim[i] << "/" << view.stride[i];
              if (i != view.ndims - 1) {
                oss << " ";
              }
            }
            oss << ")";
            if (log) {
              this->logger->info() << "Rejecting machine view: " << oss.str();
            }
          }
          break;
        }
      }
      if (valid) {
        {
          MachineView const &view = this->model->all_valid_views[i];
          std::ostringstream oss;
          oss << "[" << view.ndims << "](";
          for (int i = 0; i < view.ndims; i++) {
            oss << view.dim[i] << "/" << view.stride[i];
            if (i != view.ndims - 1) {
              oss << " ";
            }
          }
          oss << ")";
          if (log) {
            this->logger->info() << "Accepting machine view: " << oss.str();
          }
        }
        to_cache->push_back(this->model->all_valid_views[i]);
      }
    }
    cached_operator_valid_views[op->op_guid] = std::move(to_cache);
    cached_op_views = cached_operator_valid_views.at(op->op_guid).get();
  }
  if (log) {
    this->logger->info() << "Found " << cached_op_views->size()
                         << " cached op views";
  }
  for (size_t i = 0; i < cached_op_views->size(); i++) {
    MachineView view = (*cached_op_views)[i];
    if (view.device_type == MachineView::GPU) {
      view.start_device_id = resource.start_gpu_id;
    } else if (view.device_type == MachineView::CPU) {
      view.start_device_id = resource.start_cpu_id;
    } else {
      assert(false);
    }
    if (resource.is_valid_machine_view(view)) {
      valid_views.push_back(view);
    }
  }
  return valid_views;
}

Node Graph::find_bottleneck_node(Node const &sink_node,
                                 Node const &source_node) const {
  using FlexFlow::PCG::Utils::GraphStructure;
  using FlexFlow::PCG::Utils::imm_post_dominators;
  using FlexFlow::PCG::Utils::MultisourceGraphStructure;
  using FlexFlow::PCG::Utils::roots;

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

void Edge::replace_node(Node const &currentOp, Node const &replaceWith) {
  if (this->srcOp == currentOp) {
    this->srcOp = replaceWith;
  }
  if (this->dstOp == currentOp) {
    this->dstOp = replaceWith;
  }
}

Graph Graph::subgraph(std::unordered_set<Node> const &ns) const {
  using FlexFlow::PCG::Utils::nodes;

  Graph sub(this->model);

  std::unordered_set<Node> all_nodes = nodes(*this);

  for (Node const &node : ns) {
    assert(all_nodes.find(node) != all_nodes.end());
    sub.add_node(node);
  }
  for (auto const &kv : this->inEdges) {
    for (Edge const &in_edge : kv.second) {
      if (ns.find(in_edge.srcOp) != ns.end() &&
          ns.find(in_edge.dstOp) != ns.end()) {
        sub.add_edge(in_edge);
      }
    }
  }

  return sub;
}

void Graph::remove_node(Node const &node, bool purge_edges) {
  if (purge_edges) {
    std::unordered_set<Edge> out_edges = this->outEdges.at(node);
    for (auto const &e : out_edges) {
      this->remove_edge(e, false /*remove_node_if_unused*/);
    }
    std::unordered_set<Edge> in_edges = this->outEdges.at(node);
    for (auto const &e : in_edges) {
      this->remove_edge(e, false /*remove_node_if_unused*/);
    }
  } else {
    assert(this->inEdges.at(node).empty());
    assert(this->outEdges.at(node).empty());
  }
  this->inEdges.erase(node);
  this->outEdges.erase(node);
}

/*static*/
Graph Graph::singleton(FFModel *model, Node const &node) {
  Graph g(model);
  g.add_node(node);
  return g;
}

bool Graph::empty() const {
  bool inEdges_empty = this->inEdges.empty();
  bool outEdges_empty = this->outEdges.empty();
  assert(inEdges_empty == outEdges_empty);
  return inEdges_empty;
}

void Graph::replace_subgraph(std::unordered_set<Node> const &currentNodes,
                             Graph const &replaceWith) {
  assert(currentNodes.size() > 0);
  if (replaceWith.empty()) {
    Graph subgraph = this->subgraph(currentNodes);
    assert(!subgraph.empty());
    Node source_node = subgraph.find_source_node();
    Node noop =
        this->model->get_or_create_noop_node(source_node.ptr->inputs[0]);
    this->replace_subgraph_with_nonempty(currentNodes,
                                         Graph::singleton(this->model, noop));
    this->contract_out_node(noop);
  } else {
    this->replace_subgraph_with_nonempty(currentNodes, replaceWith);
  }
}

void Graph::replace_subgraph_with_nonempty(
    std::unordered_set<Node> const &currentNodes, Graph const &replaceWith) {
  using FlexFlow::PCG::Utils::get_edges;
  using FlexFlow::PCG::Utils::nodes;

  Node new_sink_node = replaceWith.find_sink_node();

  Graph old_subgraph = this->subgraph(currentNodes);
  Node old_sink_node = old_subgraph.find_sink_node();
  Node old_source_node = old_subgraph.find_source_node();

  std::unordered_set<Node> all_nodes = nodes(*this);

  for (Edge const &old_inner_edge : get_edges(old_subgraph)) {
    this->remove_edge(old_inner_edge, false);
  }
  for (Edge const &new_inner_edge : get_edges(replaceWith)) {
    this->add_edge(new_inner_edge);
  }

  std::unordered_set<Edge> old_in_edges = this->inEdges.at(old_source_node);
  if (!old_in_edges.empty()) {
    Node new_source_node = replaceWith.find_source_node();
    for (Edge const &old_in_edge : old_in_edges) {
      Edge new_in_edge(old_in_edge);
      new_in_edge.dstOp = new_source_node;
      this->remove_edge(old_in_edge, false);
      this->add_edge(new_in_edge);
    }
  }

  std::unordered_set<Edge> old_out_edges = this->outEdges.at(old_sink_node);
  for (Edge const &old_out_edge : old_out_edges) {
    Edge new_out_edge(old_out_edge);
    new_out_edge.srcOp = new_sink_node;
    this->remove_edge(old_out_edge, false);
    this->add_edge(new_out_edge);
  }

  for (Node const &node : currentNodes) {
    this->remove_node(node);
  }

  assert(this->check_correctness());
}

void Graph::contract_out_node(Node const &node) {
  using FlexFlow::PCG::Utils::successors;

  assert(node.ptr->numOutputs == 1);
  assert(node.ptr->numInputs == 1);

  std::unordered_set<Edge> in_edges = this->inEdges.at(node);
  assert(in_edges.size() == 1);
  std::unordered_set<Edge> out_edges = this->outEdges.at(node);

  for (auto const &in_edge : in_edges) {
    this->remove_edge(in_edge);
    for (auto const &out_edge : out_edges) {
      this->remove_edge(out_edge);
      this->add_edge(
          in_edge.srcOp, out_edge.dstOp, in_edge.srcIdx, out_edge.dstIdx);
    }
  }
}

void Graph::simplify_parallel_ops() {
  log_simplify.debug() << "Trying to simplify parallel ops";
  using FlexFlow::PCG::Utils::nodes;
  using FlexFlow::PCG::Utils::predecessor;
  using FlexFlow::PCG::Utils::predecessors;
  using FlexFlow::PCG::Utils::successor;

  std::queue<Node> work_queue;
  for (Node const &node : nodes(*this)) {
    if (node.ptr->is_parallel_op()) {
      work_queue.push(node);
    }
  }

  while (!work_queue.empty()) {
    Node node = work_queue.front();
    log_simplify.debug() << "Trying to simplify starting from "
                         << node.to_string();
    work_queue.pop();

    auto opt_succ = successor(*this, node);
    if (!opt_succ.has_value()) {
      log_simplify.debug() << "Skipping because does not have single successor";
      continue;
    }
    Node succ = opt_succ.value();
    if (!succ.ptr->is_parallel_op()) {
      log_simplify.debug() << "Skipping because successor is not a parallel op";
      continue;
    }

    std::vector<ParallelOpInfo> node_parallel_op_info,
        successor_parallel_op_info;
    ((ParallelOp *)node.ptr)->append_parallel_op_info(node_parallel_op_info);
    ((ParallelOp *)succ.ptr)
        ->append_parallel_op_info(successor_parallel_op_info);
    ParallelOpJoinResult result = try_join_parallel_ops(
        node_parallel_op_info.front(), successor_parallel_op_info.front());

    if (!result.join_did_succeed) {
      log_simplify.debug() << "Skipping because join did not succeed";
      continue;
    }
    log_simplify.debug() << "Did join nodes";
    log_simplify.debug() << "  " << node.to_string();
    log_simplify.debug() << "  " << succ.to_string();

    for (Node const &p : predecessors(*this, node)) {
      if (p.ptr->is_parallel_op()) {
        work_queue.push(p);
      }
    }

    Graph new_g(this->model);
    if (result.op.has_value()) {
      Node new_op = this->model->get_or_create_parallel_op_node(
          node.ptr->inputs[0], result.op.value());
      work_queue.push(new_op);
      new_g.add_node(new_op);
    }
    this->replace_subgraph({node, succ}, new_g);
  }
  log_simplify.debug() << "Finished simplifying parallel ops";
}

void Graph::simplify(SimplificationSettings const &settings) {
  // Simplify the graph by eliminating reverse parallel ops
  // and fusing multiple parallel ops
  // old graph: e1->n1->e2->n2->en
  // new graph: e1->new_node->en
  // TODO: temporarily disabled graph simplification
  if (settings.simplify_parallel_ops) {
    this->simplify_parallel_ops();
  }
  if (settings.fuse_parallel_ops) {
    bool simplify = true;
    while (simplify) {
      simplify = false;
      for (auto const &it : this->inEdges) {
        if (it.first.ptr == NULL) {
          continue;
        }
        if (it.first.ptr->is_parallel_op()) {
          Node n2 = it.first;
          assert(it.second.size() == 1);
          Edge e2 = *it.second.begin();
          Node n1 = e2.srcOp;
          // Check that n1 is a parallel op
          // Check that n1 must have a single out edge
          if (n1.ptr->is_parallel_op() &&
              this->outEdges.find(n1)->second.size() == 1) {
            // merge n1 and n2
            std::vector<ParallelOpInfo> parallel_ops;
            ((ParallelOp *)n1.ptr)->append_parallel_op_info(parallel_ops);
            ((ParallelOp *)n2.ptr)->append_parallel_op_info(parallel_ops);
            Node new_node = model->get_or_create_fused_parallel_node(
                n1.ptr->inputs[0], parallel_ops);
            auto const &inList = this->inEdges.find(n1)->second;
            assert(inList.size() == 1);
            Edge e1 = *inList.begin();
            // Update graph by adding edges
            this->add_edge(e1.srcOp, new_node, e1.srcIdx, 0);
            this->remove_edge(e1);
            this->remove_edge(e2);
            // make a copy of outList
            if (this->outEdges.find(n2) != this->outEdges.end()) {
              auto const outList = this->outEdges.find(n2)->second;
              for (auto const &e : outList) {
                this->add_edge(new_node, e.dstOp, 0, e.dstIdx);
                this->remove_edge(e);
              }
            }
            simplify = true;
          }
        }
        if (simplify) {
          break;
        }
      }
    }
  }

  if (settings.remove_trailing_parallel_ops) {
    // Remove final parallel ops
    std::vector<Node> candidates;
    for (auto const &it : this->outEdges) {
      if (it.second.size() == 0 && it.first.ptr->op_type != OP_REDUCTION &&
          it.first.ptr->op_type != OP_FUSED_PARALLEL &&
          it.first.ptr->is_parallel_op()) {
        candidates.push_back(it.first);
      }
    }
    size_t index = 0;
    while (index < candidates.size()) {
      Node parallel_op = candidates[index++];
      auto const &inList = this->inEdges.find(parallel_op)->second;
      assert(inList.size() == 1);
      Edge e = *inList.begin();
      this->remove_edge(e);
      if (this->outEdges.find(e.srcOp)->second.size() == 0 &&
          e.srcOp.ptr->is_parallel_op()) {
        candidates.push_back(e.srcOp);
      }
    }
  }

  if (settings.remove_noops) {
    // Remove NoOps
    std::vector<Node> noop_nodes;
    for (auto const &it : this->inEdges) {
      if (it.first.ptr == NULL) {
        continue;
      }
      if (it.first.ptr->op_type == OP_NOOP) {
        noop_nodes.push_back(it.first);
      }
    }
    size_t index = 0;
    while (index < noop_nodes.size()) {
      Node noop = noop_nodes[index++];
      auto const &inList = this->inEdges.find(noop)->second;
      assert(inList.size() == 1);
      Edge in_edge = *inList.begin();
      // make a copy of outList
      if (this->outEdges.find(noop) != this->outEdges.end()) {
        auto const outList = this->outEdges.find(noop)->second;
        for (auto const &e : outList) {
          this->add_edge(in_edge.srcOp, e.dstOp, in_edge.srcIdx, e.dstIdx);
          this->remove_edge(e);
        }
      }
      this->remove_edge(in_edge);
    }
  }
}

std::pair<std::unique_ptr<Graph>, std::unique_ptr<Graph>>
    Graph::split_at_node(Node const &bottleneck) const {
  using FlexFlow::PCG::Utils::topo_sort;

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

    assert(used_nodes.size() < topo_sorted.size());
  }

  for (auto const &it : this->inEdges) {
    auto const &inList = it.second;
    if (used_nodes.find(it.first) != used_nodes.end()) {
      // Add all in-edges of used_nodes in to the first_graph
      for (auto const &it2 : inList) {
        first_graph->add_edge(it2);
      }
    } else {
      // Add all in-edges of not_used_nodes into the second_graph
      for (auto const &it2 : inList) {
        second_graph->add_edge(it2);
      }
    }
  }

  return {std::move(first_graph), std::move(second_graph)};
}

void Graph::remove_input_nodes() {
  using FlexFlow::PCG::Utils::nodes;

  for (auto const &n : nodes(*this)) {
    if (n.ptr->op_type == OP_INPUT) {
      this->remove_node(n, true /*purge_edges*/);
    }
  }
}

Node Graph::clone_node(Node const &n) {
  Node cloned = n;
  cloned.original_guid = n.guid;
  cloned.guid = this->model->node_global_guid++;
  this->add_node(cloned);
  return cloned;
}

Node Graph::declone_node(Node const &n) {
  assert(n.original_guid.has_value());
  Node decloned = n;
  decloned.guid = n.original_guid.value();
  decloned.original_guid = tl::nullopt;
  this->add_node(decloned);
  return decloned;
}

std::pair<Node, std::unordered_set<Node>>
    Graph::deduplicate_input_node(Node const &n) {
  using FlexFlow::PCG::Utils::nodes;
  using FlexFlow::PCG::Utils::outgoing_edges;

  assert(n.original_guid.has_value());
  std::unordered_set<Node> old_all_nodes = nodes(*this);
  Node decloned = this->declone_node(n);

  std::unordered_set<Node> old_nodes;
  std::unordered_set<Edge> new_edges;
  for (Node const &nn : old_all_nodes) {
    if (nn.original_guid == n.original_guid) {
      old_nodes.insert(nn);
      for (Edge const &e : outgoing_edges(*this, nn)) {
        Edge decloned_edge(e);
        decloned_edge.replace_node(nn, decloned);
        new_edges.insert(decloned_edge);
      }
      this->remove_node(nn, true /*purge_edges*/);
    }
  }

  for (Edge const &e : new_edges) {
    this->add_edge(e);
  }

  return {decloned, old_nodes};
}

std::unordered_map<Node, Node> Graph::deduplicate_input_nodes() {
  using FlexFlow::PCG::Utils::nodes;

  std::unordered_map<Node, Node> deduplication_map;

  bool done;
  while (true) {
    done = true;
    for (Node const &n : nodes(*this)) {
      if (n.original_guid.has_value()) {
        done = false;
        auto kv = this->deduplicate_input_node(n);
        for (auto const &r : kv.second) {
          deduplication_map[r] = kv.first;
        }
        break;
      }
    }
    if (done) {
      break;
    }
  }

  return deduplication_map;
}

void Graph::duplicate_input_node(Node const &n) {
  using FlexFlow::PCG::Utils::outgoing_edges;
  using FlexFlow::PCG::Utils::successors;

  assert(n.ptr->op_type == OP_INPUT);

  std::unordered_map<Node, Node> clones;

  for (auto const &s : successors(*this, n)) {
    clones[s] = this->clone_node(n);
  }

  for (auto const &e : outgoing_edges(*this, n)) {
    Edge cloned(e);
    cloned.srcOp = clones.at(e.dstOp);
    this->add_edge(cloned);
  }
  this->remove_node(n, true /*purge_edges*/);
}

void Graph::duplicate_input_nodes() {
  using FlexFlow::PCG::Utils::nodes;

  for (auto const &n : nodes(*this)) {
    if (n.ptr->op_type == OP_INPUT) {
      this->duplicate_input_node(n);
    }
  }
}

std::pair<std::unique_ptr<Graph>, std::unique_ptr<Graph>>
    Graph::split_horizontal(Node const &source_node,
                            Node const &sink_node) const {
  using FlexFlow::PCG::Utils::weakly_connected_components;

  Graph trimmed_graph(*this);
  assert(sink_node !=
         Node::INVALID_NODE); // sink node should never be invalid node
  if (source_node != Node::INVALID_NODE) {
    trimmed_graph.remove_node(source_node, true /*purge_edges*/);
  }
  trimmed_graph.remove_node(sink_node, true /*purge_edges*/);
  std::vector<std::unordered_set<Node>> wccs =
      weakly_connected_components(trimmed_graph);
  assert(wccs.size() >= 2);
  std::unordered_set<Node> first_branch = wccs.back();
  wccs.pop_back();
  std::unordered_set<Node> rest;
  for (auto const &wcc : wccs) {
    rest.insert(wcc.begin(), wcc.end());
  }
  if (source_node != Node::INVALID_NODE) {
    first_branch.insert(source_node);
    rest.insert(source_node);
  }
  first_branch.insert(sink_node);
  rest.insert(sink_node);

  auto first_graph =
      std::unique_ptr<Graph>(new Graph(this->subgraph(first_branch)));
  auto second_graph = std::unique_ptr<Graph>(new Graph(this->subgraph(rest)));

  return {std::move(first_graph), std::move(second_graph)};
}

GraphCostResult GraphCostResult::invalid() {
  return {std::numeric_limits<float>::infinity(), {}};
}

GraphCostResultWithMemory GraphCostResultWithMemory::invalid() {
  return {std::numeric_limits<float>::infinity(), MemoryUsage{}, {}};
}

float GraphCostResultWithMemory::get_multi_obj_cost() const {
  return this->cost;
}

bool GraphCostResult::operator<(GraphCostResult const &other) const {
  return this->cost < other.cost;
}

bool GraphCostResultWithMemory::operator<(
    GraphCostResultWithMemory const &other) const {
  return this->get_multi_obj_cost() < other.get_multi_obj_cost();
}

std::ostream &operator<<(std::ostream &s, GraphCostResult const &r) {
  s << "GraphCostResult{cost=" << r.cost << "}";
  return s;
}

std::ostream &operator<<(std::ostream &s, GraphCostResultWithMemory const &r) {
  s << "GraphCostResultWithMemory{run_time_cost=" << r.cost
    << ", memory_cost=" << r.mem_cost << "}";
  return s;
}

std::ostream &operator<<(std::ostream &s, GraphOptimizeResult const &r) {
  s << "GraphOptimizeResult{cost=" << r.cost << "}";
  return s;
}

std::ostream &operator<<(std::ostream &s,
                         GraphOptimizeResultWithMemory const &r) {
  s << "GraphOptimizeResultWithMemory{run_time_cost=" << r.cost
    << ", memory_cost=" << r.mem_cost << "}";
  return s;
}

template <>
GraphCostResult sequence_cost<GraphCostResult>(GraphCostResult const &first,
                                               GraphCostResult const &second) {
  GraphCostResult result(first);
  result.cost += second.cost;
  result.views.insert(second.views.cbegin(), second.views.cend());
  return result;
}

template <>
GraphCostResultWithMemory sequence_cost<GraphCostResultWithMemory>(
    GraphCostResultWithMemory const &first,
    GraphCostResultWithMemory const &second) {
  GraphCostResultWithMemory result{first};
  result.cost += second.cost;
  result.mem_cost += second.mem_cost;
  result.views.insert(second.views.cbegin(), second.views.cend());
  return result;
}

template <>
float sequence_cost<float>(float const &first, float const &second) {
  return first + second;
}

template <>
GraphOptimizeResult
    sequence_cost<GraphOptimizeResult>(GraphOptimizeResult const &first,
                                       GraphOptimizeResult const &second) {
  GraphOptimizeResult result;
  result.cost = first.cost + second.cost;
  result.views.insert(first.views.cbegin(), first.views.cend());
  result.views.insert(second.views.cbegin(), second.views.cend());

  result.graph = second.graph;
  Node second_src = result.graph.value().find_source_node();
  result.graph.value().replace_subgraph({second_src}, first.graph.value());
  return result;
}

/**
 * @brief Specialization of sequence_cost<T> to combine two
 * GraphOptimizeResultWithMemory. This reuses the parts of combining run time
 * costs. This should be merged with other versions of sequence_cost<T>.
 */
template <>
GraphOptimizeResultWithMemory sequence_cost<GraphOptimizeResultWithMemory>(
    GraphOptimizeResultWithMemory const &first,
    GraphOptimizeResultWithMemory const &second) {
  GraphOptimizeResultWithMemory result;
  result.cost = first.cost + second.cost;
  result.views.insert(first.views.cbegin(), first.views.cend());
  result.views.insert(second.views.cbegin(), second.views.cend());

  result.graph = second.graph;
  Node second_src = result.graph.value().find_source_node();
  result.graph.value().replace_subgraph({second_src}, first.graph.value());

  // New: Combine memory cost
  result.mem_cost = first.mem_cost + second.mem_cost;

  return result;
}

template <>
GraphCostResult parallel_cost<GraphCostResult>(GraphCostResult const &first,
                                               GraphCostResult const &second) {
  GraphCostResult result;
  result.cost = std::max(first.cost, second.cost);
  result.views.insert(first.views.cbegin(), first.views.cend());
  result.views.insert(second.views.cbegin(), second.views.cend());

  return result;
}

template <>
GraphCostResultWithMemory parallel_cost<GraphCostResultWithMemory>(
    GraphCostResultWithMemory const &first,
    GraphCostResultWithMemory const &second) {
  GraphCostResultWithMemory result;
  result.cost = std::max(first.cost, second.cost);
  result.mem_cost = first.mem_cost + second.mem_cost;
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
bool SearchHelper::is_invalid<GraphCostResult>(
    GraphCostResult const &cost) const {
  return cost.cost == std::numeric_limits<float>::infinity();
}

template <>
bool SearchHelper::is_invalid<GraphCostResultWithMemory>(
    GraphCostResultWithMemory const &cost) const {
  return cost.cost == std::numeric_limits<float>::infinity();
}

/**
 * @brief Asserts that the results of graph optimization are valid for the graph
 *
 * @param g the graph to check against
 * @param r the results to check
 * @param sink the sink node of the graph g
 * @param include_sink whether or not to include the sink node
 */
template <>
void SearchHelper::check_matches_graph<GraphCostResult>(
    Graph const *g, GraphCostResult const &r, Node const &sink) const {
  using FlexFlow::PCG::Utils::nodes;

  if (this->is_invalid(r)) {
    return;
  }

  std::unordered_set<Node> g_nodes = nodes(*g);
  g_nodes.erase(sink);

  std::unordered_set<Node> r_nodes;
  for (auto const &kv : r.views) {
    r_nodes.insert(kv.first);
  }

  assert(g_nodes == r_nodes);
}

template <>
void SearchHelper::check_matches_graph<GraphCostResultWithMemory>(
    Graph const *g,
    GraphCostResultWithMemory const &r,
    Node const &sink) const {
  using FlexFlow::PCG::Utils::nodes;

  if (this->is_invalid(r)) {
    return;
  }

  std::unordered_set<Node> g_nodes = nodes(*g);
  g_nodes.erase(sink);

  std::unordered_set<Node> r_nodes;
  for (auto const &kv : r.views) {
    r_nodes.insert(kv.first);
  }

  assert(g_nodes == r_nodes);
}

template <>
void SearchHelper::check_matches_graph<float>(Graph const *g,
                                              float const &r,
                                              Node const &sink) const {}

template <>
std::pair<bool, float>
    SearchHelper::try_get_cost_from_cache<float>(size_t hash) const {
  if (this->cached_graph_costs.find(hash) == this->cached_graph_costs.end()) {
    return {false, std::numeric_limits<float>::infinity()};
  } else {
    return {true, this->cached_graph_costs.at(hash)};
  }
}

template <>
std::pair<bool, GraphCostResult>
    SearchHelper::try_get_cost_from_cache<GraphCostResult>(size_t hash) const {
  return {false, GraphCostResult::invalid()};
}

template <>
std::pair<bool, GraphCostResultWithMemory>
    SearchHelper::try_get_cost_from_cache<GraphCostResultWithMemory>(
        size_t hash) const {
  return {false, GraphCostResultWithMemory::invalid()};
}

template <>
void SearchHelper::try_cache_result<float>(size_t hash,
                                           float const &value) const {
  this->logger->debug() << "cached_graph_costs[" << hash << "] = " << value;
  this->cached_graph_costs[hash] = value;
}

template <>
void SearchHelper::try_cache_result<GraphCostResult>(
    size_t hash, GraphCostResult const &value) const {
  this->logger->debug() << "cached_graph_costs[" << hash << "=" << value.cost
                        << "]";
  this->cached_graph_costs[hash] = value.cost;
}

template <>
void SearchHelper::try_cache_result<GraphCostResultWithMemory>(
    size_t hash, GraphCostResultWithMemory const &value) const {
  this->logger->debug() << "cached_graph_costs[" << hash << "="
                        << value.get_multi_obj_cost() << "]";
  this->cached_graph_costs[hash] = value.get_multi_obj_cost();
}

template <>
float SearchHelper::infinity<float>() const {
  return std::numeric_limits<float>::infinity();
}

template <>
GraphCostResult SearchHelper::infinity<GraphCostResult>() const {
  return {std::numeric_limits<float>::infinity(), {}};
}

template <>
GraphCostResultWithMemory
    SearchHelper::infinity<GraphCostResultWithMemory>() const {
  return {std::numeric_limits<float>::infinity(),
          MemoryUsage(MemoryUsageType::GLOBAL,
                      std::numeric_limits<float>::infinity()),
          {}};
}

template <>
float SearchHelper::empty<float>() const {
  return 0.0f;
}

template <>
GraphCostResult SearchHelper::empty<GraphCostResult>() const {
  return {0.0f, {}};
}

template <>
GraphCostResultWithMemory
    SearchHelper::empty<GraphCostResultWithMemory>() const {
  return {0.0f, MemoryUsage{}, {}};
}

template <typename T>
T SearchHelper::estimate_xfer_cost(Graph const *graph,
                                   NodeAssignment const &source,
                                   NodeAssignment const &sink) const {
  T result = this->empty<T>();

  if (source.node != Node::INVALID_NODE) {
    auto const &inList = graph->inEdges.find(sink.node)->second;
    float op_cost = 0.0f;
    for (auto const &it2 : inList) {
      assert(it2.srcOp == source.node);
      assert(sink.node.ptr->inputs[it2.dstIdx]->is_valid_machine_view(
          source.view));

      float estimated_xfer_cost = this->model->simulator->estimate_xfer_cost(
          sink.node.ptr, it2.dstIdx, source.view, sink.view);
      // printf("Estimated xfer cost from %s to %s: %fms\n",
      // source.node.ptr->name, sink.node.ptr->name, estimated_xfer_cost);
      op_cost += estimated_xfer_cost;
    }
    this->add_operator_cost<T>(source, op_cost, &result);
  } else {
    Node real_source = graph->find_source_node();
    assert(real_source.ptr->op_type == OP_INPUT);
    this->add_operator_cost({real_source, MachineView::NO_VIEW}, 0.0f, &result);
  }

  return result;
}

/**
 * @brief Specialization to avoid changing many calls.
 * @details Note that this function is only called when the graph has no more
 * than 2 nodes
 */
template <>
GraphCostResultWithMemory
    SearchHelper::estimate_xfer_cost<GraphCostResultWithMemory>(
        Graph const *graph,
        NodeAssignment const &source,
        NodeAssignment const &sink) const {
  GraphCostResultWithMemory result = this->empty<GraphCostResultWithMemory>();

  if (source.node != Node::INVALID_NODE) {
    // Get the in-edges of the sink node
    auto const &inList = graph->inEdges.find(sink.node)->second;
    float op_cost = 0.0f; // run time cost
    for (auto const &it2 : inList) {
      // For all edges between source node and sink node
      assert(it2.srcOp == source.node);
      assert(sink.node.ptr->inputs[it2.dstIdx]->is_valid_machine_view(
          source.view));

      float estimated_xfer_cost = this->model->simulator->estimate_xfer_cost(
          sink.node.ptr, it2.dstIdx, source.view, sink.view);
      op_cost += estimated_xfer_cost;
    }
    this->add_operator_cost_with_memory(
        source, op_cost, MemoryUsage{}, &result);
  } else {
    // The real source must be an input operator
    Node real_source = graph->find_source_node();
    assert(real_source.ptr->op_type == OP_INPUT);
    this->add_operator_cost_with_memory(
        {real_source, MachineView::NO_VIEW}, 0.0f, MemoryUsage{}, &result);
  }

  return result;
}

template <>
void SearchHelper::add_operator_cost<float>(NodeAssignment const &node,
                                            float node_cost,
                                            float *cost) const {
  *cost += node_cost;
}

template <>
void SearchHelper::add_operator_cost<GraphCostResult>(
    NodeAssignment const &node, float node_cost, GraphCostResult *cost) const {
  cost->cost += node_cost;
  cost->views[node.node] = node.view;
}

/**
 * @brief Add an operator's run time and memory cost to the graph cost.
 * "cost" is updated within this function.
 */
void SearchHelper::add_operator_cost_with_memory(
    NodeAssignment const &node,
    float node_run_time_cost,
    MemoryUsage node_mem_cost,
    GraphCostResultWithMemory *cost) const {
  cost->cost += node_run_time_cost;
  cost->mem_cost += node_mem_cost;
  cost->views[node.node] = node.view;
}

template <>
float SearchHelper::get_cost<float>(float const &f) const {
  return f;
}

template <>
float SearchHelper::get_cost<GraphCostResult>(
    GraphCostResult const &gcr) const {
  return gcr.cost;
}

template <>
float SearchHelper::get_cost<GraphCostResultWithMemory>(
    GraphCostResultWithMemory const &gcr) const {
  return gcr.get_multi_obj_cost();
}

template <typename T>
void SearchHelper::add_sink_node_costs(NodeAssignment const &sink,
                                       CostMetrics metrics,
                                       T *result) const {
  this->add_operator_cost<T>(sink,
                             metrics.forward_time + metrics.backward_time +
                                 metrics.sync_time,
                             result);
}

/**
 * @brief Specialization of add_sink_node_costs to handle
 * GraphCostResultWithMemory
 */
template <>
void SearchHelper::add_sink_node_costs<GraphCostResultWithMemory>(
    NodeAssignment const &sink,
    CostMetrics metrics,
    GraphCostResultWithMemory *result) const {
  float op_total_mem_mb = ((float)(metrics.op_total_mem / 1e4)) / 1e2;
  this->add_operator_cost_with_memory(
      sink,
      metrics.forward_time + metrics.backward_time + metrics.sync_time,
      MemoryUsage{MemoryUsageType::GLOBAL, op_total_mem_mb},
      result);
}

/**
 * @brief Core function to analyze the cost of a graph.
 *
 * @tparam T float or GraphCostResult (or GraphCostResultWithMemory in memory
 * optimization)
 */
template <typename T>
T SearchHelper::graph_cost(Graph const *graph,
                           NodeAssignment const &source,
                           NodeAssignment const &sink,
                           MachineResource const &resources,
                           bool include_sink_compute_time) const {
  TAG_ENTER(this->logger);
  this->logger->debug() << "PCG::SearchHelper::graph_cost: sink("
                        << sink.node.guid << ") "
                        << "sink.view(" << sink.view.ndims << " "
                        << sink.view.start_device_id << " " << sink.view.dim[0]
                        << ") "
                        << "source(" << source.node.guid << ") "
                        << "source.view(" << source.view.ndims << " "
                        << source.view.start_device_id << " "
                        << source.view.dim[0] << ") "
                        << "resources(" << resources.num_nodes << " "
                        << resources.start_gpu_id << " "
                        << resources.available_gpus_per_node << ")";
  if (this->model->config.profiling) {
    graph->print_dot();
  }

  assert(graph->inEdges.find(sink.node) != graph->inEdges.end());
  if (source.node != Node::INVALID_NODE) {
    assert(graph->outEdges.find(source.node) != graph->outEdges.end());
  }

  size_t hash = dp_state_hash(
      graph, sink.node, sink.view, source.node, source.view, resources);
  this->logger->spew() << "hash = " << hash;

  T result;

  std::pair<bool, T> from_cache = this->try_get_cost_from_cache<T>(hash);
  if (from_cache.first) {
    // cached_graph_costs does not include sink_compute_time
    result = from_cache.second;
  } else {
    if (graph->inEdges.size() <= 2) {
      // When there are no more than 2 nodes in the graph
      result = this->estimate_xfer_cost<T>(graph, source, sink);
      this->logger->debug()
          << "[PCG::SearchHelper::graph_cost] Estimated xfer cost is "
          << this->get_cost(result);
    } else {
      Node bn_node = graph->find_bottleneck_node(sink.node, source.node);
      if (bn_node != Node::INVALID_NODE) {
        // We found a bottleneck node
        this->logger->debug() << "Found bn_node = " << bn_node.guid;

        result = this->find_optimal_sequence_graph_time<T>(
            graph,
            bn_node,
            {source.node, source.view},
            {sink.node, sink.view},
            resources);
      } else {
        // sink node must have multiple branches
        // otherwise we should not be here
        assert(graph->inEdges.find(sink.node)->second.size() > 1);

        result = this->find_optimal_nonsequence_graph_time<T>(
            graph,
            {source.node, source.view},
            {sink.node, sink.view},
            resources);
      }
    }

    this->try_cache_result<T>(hash, result);
  }

  check_matches_graph<T>(graph, result, sink.node);

  // This is where we really add the costs of an operator
  if (include_sink_compute_time) {
    // Sink node costs
    CostMetrics metrics =
        this->model->simulator->measure_operator_cost(sink.node.ptr, sink.view);

    // Adjust operator memory usage
    this->logger->spew()
        << "[PCG::SearchHelper::graph_cost] Analyzing sink op memory cost ["
        << sink.node.to_string() << "]:";

    int input_num_parts = 0;
    int output_num_parts = 0;
    int weight_num_parts = 0;
    auto op = sink.node.ptr;
    this->logger->spew() << "  input ParallelTensor shape|num_replicas:";
    for (int i = 0; i < op->numInputs; i++) {
      auto shape = op->inputs[i]->get_shape();
      this->logger->spew() << shape << "|" << shape.get_num_replicas() << "; ";
      if (input_num_parts == 0) {
        input_num_parts = op->inputs[i]->get_total_num_parts();
      }
    }
    this->logger->spew() << "  output ParallelTensor shape|num_replicas:";
    for (int i = 0; i < op->numOutputs; i++) {
      auto shape = op->outputs[i]->get_shape();
      this->logger->spew() << shape << "|" << shape.get_num_replicas() << "; ";
      if (output_num_parts == 0) {
        output_num_parts = op->outputs[i]->get_total_num_parts();
      }
    }
    this->logger->spew() << "  weight ParallelTensor shape|num_replicas:";
    for (int i = 0; i < op->numWeights; i++) {
      if (op->weights[i] == nullptr) {
        continue;
      }
      auto shape = op->weights[i]->get_shape();
      this->logger->spew() << shape << "|" << shape.get_num_replicas() << "; ";
      if (weight_num_parts == 0) {
        weight_num_parts = op->weights[i]->get_total_num_parts();
      }
    }
    input_num_parts = std::max(input_num_parts, 1);
    output_num_parts = std::max(output_num_parts, 1);
    weight_num_parts = std::max(weight_num_parts, 1);
    if (input_num_parts > weight_num_parts) {
      weight_num_parts = input_num_parts;
    }
    this->logger->spew()
        << "  Total number of parts of inputs|outputs|weights: "
        << input_num_parts << "|" << output_num_parts << "|"
        << weight_num_parts;

    // Real memory usage of this Op* considering parallelization over devices
    this->logger->spew() << "  cost_metrics input|output|weight memory: "
                         << metrics.inputs_memory << "|"
                         << metrics.outputs_memory << "|"
                         << metrics.weights_memory;

    metrics.op_total_mem = input_num_parts * metrics.inputs_memory +
                           output_num_parts * metrics.outputs_memory +
                           weight_num_parts * metrics.weights_memory;

    this->logger->spew() << "  op_total_mem: " << metrics.op_total_mem;
    float op_total_mem_mb = (float)((metrics.op_total_mem) / 1e4) / 1e2;
    this->logger->debug() << "[PCG::SearchHelper::graph_cost] Sink node cost ["
                          << sink.node.to_string() << "]: "
                          << "forward(" << metrics.forward_time << ") "
                          << "backward(" << metrics.backward_time << ") "
                          << "sync(" << metrics.sync_time << ") "
                          << "memory(" << op_total_mem_mb << " MB)";
    this->add_sink_node_costs<T>(sink, metrics, &result);
  }

  return result;
}

/**
 * @brief Get the optimal run time cost of a PCG.
 * @details This is the current metric used to decide which PCG is better
 * in Unity's search algorithm.
 */
float Graph::optimal_cost() const {
  return this->generic_optimal_cost<float>();
}

/**
 * @brief Get a single number to represent the multi-objective cost of a PCG.
 */
float Graph::optimal_cost_with_memory(float run_time_cost_factor) const {
  auto optimal = this->generic_optimal_cost<GraphCostResultWithMemory>();
  float run_time_cost = optimal.cost;
  float mem_cost = optimal.mem_cost.num;
  // This is where we combine two costs to get the multi-objective cost
  auto combined_cost = (run_time_cost_factor * run_time_cost +
                        (1 - run_time_cost_factor) * mem_cost);
  std::string output_str =
      "Multi-objective cost in Graph::optimal_cost_with_memory:"
      "run time cost: " +
      std::to_string(run_time_cost) +
      ", mem cost: " + std::to_string(mem_cost) +
      ", combined cost: " + std::to_string(combined_cost) +
      " (with run time cost factor: " + std::to_string(run_time_cost_factor) +
      ")";
  this->search->logger->spew() << output_str;
  return combined_cost;
}

std::unordered_map<Node, MachineView> Graph::optimal_views() const {
  return this->generic_optimal_cost<GraphCostResult>().views;
}

Graph Graph::reduced() const {
  using FlexFlow::PCG::Utils::BasicGraph;
  using FlexFlow::PCG::Utils::get_edges;
  using FlexFlow::PCG::Utils::transitive_reduction;

  BasicGraph<Node> transitive_skeleton = transitive_reduction(*this);

  Graph reduced_graph(this->model);

  for (Edge const &e : get_edges(*this)) {
    if (transitive_skeleton.has_edge(e.srcOp, e.dstOp)) {
      reduced_graph.add_edge(e);
    }
  }

  return reduced_graph;
}

/**
 * @brief A generic cost function for a graph capable of finding both the cost
 * and the optimal views
 *
 * @note A templated function is used here because while the caching behaviors
 * of the cost and the optimal views are different, much of the code between the
 * two versions is almost identical. By using a few template specializations we
 * can avoid duplicating all this code.
 *
 * @tparam T Result type (float, GraphCostResult, or GraphCostResultWithMemory)
 * @return T Cost of the graph (along with any additional data)
 */
template <typename T>
T Graph::generic_optimal_cost() const {
  using FlexFlow::PCG::Utils::GraphStructure;

  Graph reduced_graph = this->reduced();
  // GraphStructure<Graph> s;
  // if (source_node.ptr->op_type == OP_INPUT) {
  //   for (auto const &e : s.get_outgoing_edges(reduced_graph, source_node)) {
  //     reduced_graph.remove_edge(e, false/*remove_node_if_unused*/);
  //   }
  //   reduced_graph.remove_node(source_node);
  // }

  Node sink_node = reduced_graph.find_sink_node();
  this->search->logger->info()
      << "Graph::generic_optimal_cost: Found sink node: "
      << sink_node.to_string();

  MachineResource resource(model->config);

  std::vector<MachineView> valid_views =
      search->get_valid_machine_views(sink_node, resource, true);

  T optimal = search->infinity<T>();

  this->search->logger->info()
      << "Exploring " << valid_views.size() << " valid views";
  for (MachineView const &sink_view : valid_views) {
    this->search->logger->info() << "  Exploring valid view " << sink_view;
    T new_cost =
        search->graph_cost<T>(&reduced_graph,
                              {Node::INVALID_NODE, MachineView::NO_VIEW},
                              {sink_node, sink_view},
                              resource,
                              true);
    if (new_cost < optimal) {
      optimal = new_cost;
    }
  }

  return optimal;
}

size_t Graph::hash(void) const {
  // Graph hash should be additive and independent to the ordering of the nodes
  size_t total_hash = 0;
  for (auto const &it : inEdges) {
    auto const &inList = it.second;
    size_t node_hash = std::hash<size_t>()((size_t)it.first.ptr);
    for (auto const &e : inList) {
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

size_t dp_state_hash(Graph const *graph,
                     Node const &sink_node,
                     MachineView const &sink_view,
                     Node const &source_node,
                     MachineView const &source_view,
                     MachineResource const &resource) {
  size_t key = graph->hash();
  hash_combine(key, sink_node.ptr);
  hash_combine(key, sink_view.hash());
  hash_combine(key, source_node.ptr);
  hash_combine(key, resource.hash());
  return key;
}

namespace {

/**
 * @brief Given a lambda value, perform the search and return the optimized PCG
 * and corresponding MachineView.
 */
std::pair<std::unique_ptr<Graph>, std::unordered_map<Node, MachineView>>
    try_one_lambda(std::pair<float, MemorySearchResult> &lambda,
                   Task const *task,
                   std::shared_ptr<Simulator> &cached_simulator,
                   bool perform_memory_search) {
  // Create a new fresh model
  FFModel *model = *((FFModel **)task->args);
  model->clear_graph_search_cache();

  if (model->config.search_num_nodes.has_value()) {
    model->config.numNodes = model->config.search_num_nodes.value();
  }
  if (model->config.search_num_workers.has_value()) {
    model->config.workersPerNode = model->config.search_num_workers.value();
  }
  model->all_valid_views.clear();
  model->register_all_machine_views(model->config.numNodes,
                                    model->config.workersPerNode,
                                    model->config.cpusPerNode,
                                    model->all_valid_views);
  Memory gpu_mem = Machine::MemoryQuery(Machine::get_machine())
                       .only_kind(Memory::GPU_FB_MEM)
                       .best_affinity_to(task->target_proc)
                       .first();
  MachineModel *machine;
  if (model->config.machine_model_version == 0) {
    machine =
        (MachineModel *)new SimpleMachineModel(model->config.numNodes,
                                               model->config.workersPerNode,
                                               gpu_mem.capacity());
  } else if (model->config.machine_model_version == 1 and
             !model->config.machine_model_file.empty()) {
    machine = (MachineModel *)new EnhancedMachineModel(
        model->config.machine_model_file, gpu_mem.capacity());
  } else {
    assert(false &&
           "machine model creation error: currently only support "
           "machine-model-version = 0 or 1. When machine-model-version = 1, "
           "machine-model-file should not be empty.");
  }
  // Assume this task is running on GPU0
  if (!cached_simulator) {
    cached_simulator = std::make_shared<Simulator>(
        model, model->handlers[0], gpu_mem, machine);
  } else {
    // Update simulator with the new stuff
    cached_simulator->handler = model->handlers[0];
    cached_simulator->memory = gpu_mem;
    cached_simulator->machine = machine;
  }
  model->simulator = cached_simulator.get();

  // Perform the search
  std::unique_ptr<Graph> curr_best_graph;
  std::unordered_map<Node, MachineView> curr_optimal_views;

  if (model->config.only_data_parallel) {
    Graph *graph = new Graph(model);
    std::unordered_map<FlexFlow::Op const *, Node> op_to_node_map;
    for (FlexFlow::Op const *dstOp : model->operators) {
      Node dstNode;
      dstNode.ptr = dstOp;
      dstNode.guid = model->node_global_guid++;
      op_to_node_map[dstOp] = dstNode;
      for (int j = 0; j < dstOp->numInputs; j++) {
        FlexFlow::Op const *srcOp = dstOp->inputs[j]->owner_op;
        assert(op_to_node_map.find(srcOp) != op_to_node_map.end());
        Node srcNode = op_to_node_map[srcOp];
        graph->add_edge(srcNode, dstNode, dstOp->inputs[j]->owner_idx, j);
      }
    }
    curr_best_graph = std::unique_ptr<Graph>(graph);
    MachineView data_parallel_view;
    data_parallel_view.device_type = MachineView::GPU;
    data_parallel_view.ndims = 1;
    data_parallel_view.dim[0] =
        model->config.numNodes * model->config.workersPerNode;
    data_parallel_view.stride[0] = 1;
    data_parallel_view.start_device_id = 0;
    for (auto const &node : curr_best_graph->inEdges) {
      curr_optimal_views[node.first] = data_parallel_view;
    }
  } else {
    // Main step to optimize the PCG of an FFModel
    model->graph_optimize(model->config.search_budget,
                          model->config.only_data_parallel,
                          curr_best_graph,
                          curr_optimal_views,
                          perform_memory_search,
                          MemoryOptimConfig{lambda.first},
                          lambda.second);
  }
  // Return the best result of the current search
  return std::make_pair(std::move(curr_best_graph), curr_optimal_views);
};

/**
 * @brief Analyze the per-device memory cost and compare with the memory
 * threshold of each device.
 */
bool is_valid_strategy(
    std::vector<std::pair<float, MemorySearchResult>> &lambdas_results,
    Graph *curr_graph,
    std::unordered_map<Node, MachineView> &curr_views,
    std::shared_ptr<Simulator> const cached_simulator,
    float memory_threshold) {
  std::cout << "try to check valid for lambda " << lambdas_results.back().first
            << std::endl;
  assert(cached_simulator.get() != nullptr &&
         "cached_simulator cannot be nullptr");

  // Analyze the strategy and update max_per_device_mem_all_deivces in the
  // lambda_result.
  std::unordered_map<int, float> device_to_mem{};
  for (auto const &view : curr_views) {
    CostMetrics op_cost =
        cached_simulator->measure_operator_cost(view.first.ptr, view.second);
    float node_mem_as_mb = op_cost.total_memory_in_mb();

    for (auto const d_id : view.second.device_ids()) {
      if (device_to_mem.find(d_id) == device_to_mem.end()) {
        device_to_mem.emplace(std::make_pair(d_id, node_mem_as_mb));
      } else {
        device_to_mem[d_id] += node_mem_as_mb;
      }
    }
  }

  float max_per_device_mem = 0.0;
  float total_device_mem = 0.0;
  for (auto const &d : device_to_mem) {
    std::cout << "d_id: " << d.first << ", mem: " << d.second << std::endl;
    total_device_mem += d.second;
    if (d.second > max_per_device_mem) {
      max_per_device_mem = d.second;
    }
  }

  lambdas_results.back().second.max_per_device_mem_all_deivces =
      max_per_device_mem;

  std::cout << "max_per_device_mem: "
            << lambdas_results.back().second.max_per_device_mem_all_deivces
            << ", total_device_mem: " << total_device_mem << std::endl;

  if (max_per_device_mem >= memory_threshold) {
    return false;
  }
  return true;
};

}; // namespace

/**
 * @brief Starting point of Unity search procedure. Registered on Legion
 * runtime. Legion task to launch as one step of model.compile().
 *
 * @param task Legion task to get FFModel and other configs
 * @param regions Not used
 * @param ctx Not used
 * @param runtime Not used
 * @return GraphOptimalViewSerialized Serialized optimal PCG
 */
GraphOptimalViewSerialized
    Graph::graph_optimize_task(Task const *task,
                               std::vector<PhysicalRegion> const &regions,
                               Context ctx,
                               Runtime *runtime) {
  auto model_config = (*((FFModel **)task->args))->config;
  bool perform_memory_search = model_config.perform_memory_search;
  float memory_threshold = model_config.device_mem;
  bool only_data_parallel = model_config.only_data_parallel;

  std::vector<std::pair<float, MemorySearchResult>> lambdas{};

  std::shared_ptr<Simulator> cached_simulator{};

  // Optimized graph from the search
  std::unique_ptr<Graph> best_graph;
  std::unordered_map<Node, MachineView> optimal_views;

  // Be optimistic
  lambdas.emplace_back(std::make_pair(1.0, MemorySearchResult{}));
  auto try_result = try_one_lambda(
      lambdas.back(), task, cached_simulator, perform_memory_search);
  best_graph = std::move(try_result.first);
  optimal_views = try_result.second;

  bool has_valid_strategy = false;
  int best_lambda_index = -1;
  int binary_search_budget = 10;

  if (perform_memory_search && !is_valid_strategy(lambdas,
                                                  best_graph.get(),
                                                  optimal_views,
                                                  cached_simulator,
                                                  memory_threshold)) {
    // Not found the strategy; need to do binary search
    lambdas.emplace_back(std::make_pair(0.0, MemorySearchResult{}));
    try_result = try_one_lambda(
        lambdas.back(), task, cached_simulator, perform_memory_search);
    best_graph = std::move(try_result.first);
    optimal_views = try_result.second;

    if (!is_valid_strategy(lambdas,
                           best_graph.get(),
                           optimal_views,
                           cached_simulator,
                           memory_threshold)) {
      // Cannot find a valid strategy
      has_valid_strategy = false;
    } else {
      has_valid_strategy = true;
      best_lambda_index = 1;

      // Do a binary search between 0 and 1 for the best lambda
      int bianry_search_num = 0;
      float lower = 0.0;
      float upper = 1.0;

      while (bianry_search_num < binary_search_budget) {
        bianry_search_num++;

        float mid = (lower + upper) * 0.5;

        lambdas.emplace_back(std::make_pair(mid, MemorySearchResult{}));
        try_result = try_one_lambda(
            lambdas.back(), task, cached_simulator, perform_memory_search);

        if (!is_valid_strategy(lambdas,
                               try_result.first.get(),
                               try_result.second,
                               cached_simulator,
                               memory_threshold)) {
          upper = mid;
        } else {
          // Found a better and valid strategy
          best_graph = std::move(try_result.first);
          optimal_views = try_result.second;

          lower = mid;
          best_lambda_index = 1 + bianry_search_num;
        }
      }
    }
  } else {
    has_valid_strategy = true;
    best_lambda_index = 0;
  }

  // Print out the results
  if (perform_memory_search) {
    if (has_valid_strategy) {
      auto &best_l = lambdas[best_lambda_index];
      std::cout << "Found valid strategy with memory_threshold: "
                << memory_threshold << " | lambda index: " << best_lambda_index
                << ", lambda value: " << best_l.first
                << ", result: run time cost: " << best_l.second.run_time_cost
                << ", memory cost: " << best_l.second.memory_cost
                << ", search time: " << best_l.second.search_time
                << ", per-device max memory: "
                << best_l.second.max_per_device_mem_all_deivces << std::endl;
    } else {
      std::cout << "Failed to find a valid strategy" << std::endl;
    }

    std::cout << "All lambda results:" << std::endl;
    for (auto l : lambdas) {
      std::cout << "lambda: " << l.first
                << ", run time cost: " << l.second.run_time_cost
                << ", memory cost: " << l.second.memory_cost
                << ", search time: " << l.second.search_time
                << ", per-device max memory: "
                << l.second.max_per_device_mem_all_deivces << std::endl;
    }
  } else if (!only_data_parallel) {
    std::cout << "\nNot doing memory search" << std::endl;
  }

  // Following lines are to serialize the optimized PCG.
  // Only need best_graph and optimal_views below.
  Serializer sez;
  // First serialize graph
  sez.serialize(best_graph->inEdges.size());
  std::unordered_map<Node, int> todos;
  std::vector<Node> opList;
  for (auto const &it : best_graph->inEdges) {
    auto const &inList = it.second;
    todos[it.first] = (int)inList.size();
    if (todos[it.first] == 0) {
      opList.push_back(it.first);
    }
  }
  size_t node_idx = 0;
  while (node_idx < opList.size()) {
    Node cur_node = opList[node_idx++];
    auto const &outList = best_graph->outEdges[cur_node];
    for (auto const &e : outList) {
      todos[e.dstOp]--;
      if (todos[e.dstOp] == 0) {
        opList.push_back(e.dstOp);
      }
    }
    auto const &inList = best_graph->inEdges[cur_node];
    sez.serialize(inList.size());
    for (auto const &e : inList) {
      sez.serialize(e.srcOp.guid);
      assert(e.dstOp.guid == cur_node.guid);
      sez.serialize(e.srcIdx);
      sez.serialize(e.dstIdx);
    }
    sez.serialize((size_t)10101010); // safe guard for the end of inedges
    Op const *op = cur_node.ptr;
    assert(op != NULL);
    sez.serialize(cur_node.guid);
    sez.serialize(op->op_type);
    switch (op->op_type) {
      case OP_INPUT: {
        assert(op->numOutputs == 1);
        NoOp *noop = (NoOp *)op;
        sez.serialize(noop->op_type);
        sez.serialize(noop->input_tensor_guid);
        sez.serialize(noop->outputs[0]->data_type);
        sez.serialize(noop->outputs[0]->num_dims);
        for (int i = 0; i < noop->outputs[0]->num_dims; i++) {
          sez.serialize(noop->outputs[0]->dims[i]);
        }
        break;
      }
      case OP_NOOP: {
        break;
      }
      case OP_CONCAT: {
        Concat *concat = (Concat *)op;
        sez.serialize(concat->legion_axis);
        break;
      }
      case OP_SPLIT: {
        Split *split = (Split *)op;
        sez.serialize(split->legion_axis);
        sez.serialize(split->numOutputs);
        for (int i = 0; i < split->numOutputs; i++) {
          sez.serialize(split->outputs[i]->dims[split->legion_axis].size);
        }
        break;
      }
      case OP_EMBEDDING: {
        Embedding *embed = (Embedding *)op;
        sez.serialize(embed->layer_guid.id);
        sez.serialize(embed->num_entries);
        sez.serialize(embed->out_channels);
        sez.serialize(embed->aggr);
        sez.serialize(embed->data_type);
        break;
      }
      case OP_EW_ADD:
      case OP_EW_SUB:
      case OP_EW_MUL:
      case OP_EW_MAX:
      case OP_EW_MIN: {
        sez.serialize(op->op_type);
        break;
      }
      case OP_MULTIHEAD_ATTENTION: {
        MultiHeadAttention *attn = (MultiHeadAttention *)op;
        sez.serialize(attn->layer_guid.id);
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
      case OP_SOFTMAX: {
        Softmax *softmax = (Softmax *)op;
        sez.serialize(softmax->dim);
        break;
      }
      case OP_REPARTITION: {
        Repartition *repart = (Repartition *)op;
        sez.serialize(repart->repartition_dim);
        sez.serialize(repart->repartition_degree);
        break;
      }
      case OP_REPLICATE: {
        Replicate *replicate = (Replicate *)op;
        sez.serialize(replicate->replicate_dim);
        sez.serialize(replicate->replicate_degree);
        break;
      }
      case OP_REDUCTION: {
        Reduction *reduction = (Reduction *)op;
        sez.serialize(reduction->reduction_dim);
        sez.serialize(reduction->reduction_degree);
        break;
      }
      case OP_COMBINE: {
        Combine *combine = (Combine *)op;
        sez.serialize(combine->combine_dim);
        sez.serialize(combine->combine_degree);
        break;
      }
      case OP_FUSED_PARALLEL: {
        FusedParallelOp *fused = (FusedParallelOp *)op;
        sez.serialize(fused->num_parallel_ops);
        for (int i = 0; i < fused->num_parallel_ops; i++) {
          sez.serialize(fused->parallel_ops[i]);
        }
        break;
      }
      default: {
        op->serialize(sez);
      }
    }
    sez.serialize((size_t)12345678); // safe guard for the end of an op
  }
  assert(node_idx == best_graph->inEdges.size());
  // Second, serialize optimal machine view
  printf("optimal_views.size = %zu\n", optimal_views.size());
  sez.serialize(optimal_views.size());
  for (auto const &it : optimal_views) {
    sez.serialize((size_t)98765432); // safe guard
    sez.serialize(it.first.guid);
    sez.serialize(it.second);
  }
  assert(sez.get_used_bytes() < GraphOptimalViewSerialized::buffer_size);
  GraphOptimalViewSerialized ret;
  ret.total_bytes = sez.get_used_bytes();
  memcpy(ret.data, sez.get_buffer(), ret.total_bytes);
  // Deallocate best_graph
  // delete best_graph;
  return ret;
}

}; // namespace FlexFlow::PCG

namespace FlexFlow {

using PCG::Edge;
using PCG::Graph;
using PCG::GraphCostResult;
using PCG::Node;

void FFModel::register_all_machine_views(
    int num_nodes,
    int gpus_per_node,
    int cpus_per_node,
    std::vector<MachineView> &valid_views) {
  // Single-parallelism-dimension views
  for (int i = 1; i <= num_nodes * gpus_per_node; i++) {
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
  // Two-dimensional views
  /* for (int i = 1; i <= num_nodes; i++) { */
  /*   for (int j = 1; j <= gpus_per_node; j++) { */
  /*     MachineView view; */
  /*     view.device_type = MachineView::GPU; */
  /*     view.ndims = 2; */
  /*     view.dim[0] = i; */
  /*     view.stride[0] = 1; */
  /*     view.dim[1] = j; */
  /*     view.stride[1] = 1; */
  /*     view.start_device_id = 0; */
  /*     valid_views.push_back(view); */
  /*   } */
  /* } */
}

float FFModel::graph_cost(Graph const *graph,
                          Node const &sink_node,
                          MachineView const &sink_view,
                          Node const &source_node,
                          MachineView const &source_view,
                          MachineResource const &resources,
                          bool include_sink_compute_time,
                          bool constructing_optimal_view) {
  assert(!graph->inEdges.empty());

  return this->search->graph_cost<float>(graph,
                                         {source_node, source_view},
                                         {sink_node, sink_view},
                                         resources,
                                         include_sink_compute_time);
}

void FFModel::construct_optimal_view(
    Graph const *graph,
    Node const &sink_node,
    MachineView const &sink_view,
    Node const &source_node,
    MachineView const &source_view,
    MachineResource const &resources,
    bool include_sink_compute_time,
    float optimal_cost,
    std::unordered_map<Node, MachineView> &optimal_views) {
  GraphCostResult result =
      this->search->graph_cost<GraphCostResult>(graph,
                                                {source_node, source_view},
                                                {sink_node, sink_view},
                                                resources,
                                                include_sink_compute_time);

  optimal_views.insert(result.views.begin(), result.views.end());
}

void FFModel::deserialize_graph_optimal_view(
    Legion::Deserializer &dez,
    Graph *graph,
    std::unordered_map<Node, MachineView> &optimal_views) {
  // Deserializer dez(serialized.data, serialized.total_bytes);
  std::unordered_map<size_t, Node> guid_to_nodes;
  size_t num_nodes;
  dez.deserialize(num_nodes);
  // best_graph = new Graph(this);
  for (size_t node_idx = 0; node_idx < num_nodes; node_idx++) {
    Edge inedges[MAX_NUM_INPUTS];
    ParallelTensor inputs[MAX_NUM_INPUTS];
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
    switch (op_type) {
      case OP_INPUT: {
        assert(num_inputs == 0);
        int num_dims;
        ParallelDim dims[MAX_TENSOR_DIM];
        OperatorType op_type;
        dez.deserialize(op_type);
        size_t input_tensor_guid;
        dez.deserialize(input_tensor_guid);
        DataType data_type;
        dez.deserialize(data_type);
        dez.deserialize(num_dims);
        for (int i = 0; i < num_dims; i++) {
          dez.deserialize(dims[i]);
        }
        ParallelTensor t =
            create_parallel_tensor_legion_ordering(num_dims,
                                                   dims,
                                                   data_type,
                                                   nullptr,
                                                   0,
                                                   true /*create_grad*/,
                                                   input_tensor_guid);
        node.ptr = t->owner_op;
        node.guid = node_global_guid++;
        break;
      }
      case OP_NOOP: {
        assert(num_inputs == 1);
        node = get_or_create_noop_node(inputs[0]);
        break;
      }
      case OP_BATCHMATMUL: {
        node = BatchMatmul::deserialize(*this, dez, inputs, num_inputs);
        break;
      }
      case OP_CAST: {
        node = Cast::deserialize(*this, dez, inputs, num_inputs);
        break;
      }
      case OP_CONCAT: {
        int legion_axis;
        dez.deserialize(legion_axis);
        node = get_or_create_node<Concat>(
            {std::begin(inputs), std::begin(inputs) + num_inputs},
            {legion_axis});
        break;
      }
      case OP_SPLIT: {
        int legion_axis;
        dez.deserialize(legion_axis);
        int num_outputs;
        dez.deserialize(num_outputs);
        std::vector<int> splits;
        for (int i = 0; i < num_outputs; i++) {
          int dim_size;
          dez.deserialize(dim_size);
          splits.push_back(dim_size);
        }
        node = get_or_create_node<Split>(inputs[0], {splits, legion_axis});
        break;
      }
      case OP_EMBEDDING: {
        assert(num_inputs == 1);
        AggrMode aggr;
        int num_entries, out_channels;
        size_t id;
        DataType data_type;
        dez.deserialize(id);
        LayerID layer_guid(id);
        dez.deserialize(num_entries);
        dez.deserialize(out_channels);
        dez.deserialize(aggr);
        dez.deserialize(data_type);

        EmbeddingParams params;
        params.aggr = aggr;
        params.num_entries = num_entries;
        params.out_channels = out_channels;
        params.layer_guid = layer_guid;
        params.data_type = data_type;
        node = get_or_create_node<Embedding>(inputs[0], params);
        break;
      }
      case OP_EW_ADD:
      case OP_EW_SUB:
      case OP_EW_MUL:
      case OP_EW_MAX:
      case OP_EW_MIN: {
        assert(num_inputs == 2);
        OperatorType op_type;
        dez.deserialize(op_type);
        node = get_or_create_node<ElementBinary>({inputs[0], inputs[1]},
                                                 {op_type});
        break;
      }
      case OP_CONV2D: {
        node = Conv2D::deserialize(*this, dez, inputs, num_inputs);
        break;
      }
      case OP_DROPOUT: {
        node = Dropout::deserialize(*this, dez, inputs, num_inputs);
        break;
      }
      case OP_EXP:
      case OP_SIN:
      case OP_COS:
      case OP_SCALAR_MULTIPLY:
      case OP_SCALAR_FLOOR_DIV:
      case OP_SCALAR_TRUE_DIV:
      case OP_SCALAR_ADD:
      case OP_SCALAR_SUB:
      case OP_RELU:
      case OP_SIGMOID:
      case OP_TANH:
      case OP_POW:
      case OP_IDENTITY:
      case OP_GELU:
      case OP_ELU: {
        node = ElementUnary::deserialize(*this, dez, inputs, num_inputs);
        break;
      }
      case OP_FLAT: {
        node = Flat::deserialize(*this, dez, inputs, num_inputs);
        break;
      }
      case OP_GATHER: {
        node = Gather::deserialize(*this, dez, inputs, num_inputs);
        break;
      }
      case OP_LAYERNORM: {
        node = LayerNorm::deserialize(*this, dez, inputs, num_inputs);
        break;
      }
      case OP_LINEAR: {
        node = Linear::deserialize(*this, dez, inputs, num_inputs);
        break;
      }
      case OP_MULTIHEAD_ATTENTION: {
        assert(num_inputs == 3);
        int embed_dim, num_heads, k_dim, v_dim;
        float dropout;
        bool bias, add_bias_kv, add_zero_attn;
        size_t id;
        dez.deserialize(id);
        LayerID layer_guid(id);
        dez.deserialize(embed_dim);
        dez.deserialize(num_heads);
        dez.deserialize(k_dim);
        dez.deserialize(v_dim);
        dez.deserialize(dropout);
        dez.deserialize(bias);
        dez.deserialize(add_bias_kv);
        dez.deserialize(add_zero_attn);

        MultiHeadAttentionParams params;
        params.embed_dim = embed_dim;
        params.num_heads = num_heads;
        params.kdim = k_dim;
        params.vdim = v_dim;
        params.dropout = dropout;
        params.bias = bias;
        params.add_bias_kv = add_bias_kv;
        params.add_zero_attn = add_zero_attn;
        params.layer_guid = layer_guid;
        node = get_or_create_node<MultiHeadAttention>(
            {inputs[0], inputs[1], inputs[2]}, params);
        break;
      }
      case OP_TOPK: {
        node = TopK::deserialize(*this, dez, inputs, num_inputs);
        break;
      }
      case OP_GROUP_BY: {
        node = Group_by::deserialize(*this, dez, inputs, num_inputs);
        break;
      }
      case OP_AGGREGATE: {
        // node = Aggregate::deserialize(*this, dez, inputs, num_inputs);
        int n;
        float lambda_bal;
        dez.deserialize(n);
        dez.deserialize(lambda_bal);
        assert(num_inputs == n + 4);
        AggregateParams params;
        params.n = n;
        params.lambda_bal = lambda_bal;
        node = get_or_create_node<Aggregate>(
            {std::begin(inputs), std::begin(inputs) + num_inputs}, params);
        break;
      }
      case OP_POOL2D: {
        node = Pool2D::deserialize(*this, dez, inputs, num_inputs);
        break;
      }
      case OP_REDUCE_SUM: {
        node = Reduce::deserialize(*this, dez, inputs, num_inputs);
        break;
      }
      case OP_RESHAPE: {
        node = Reshape::deserialize(*this, dez, inputs, num_inputs);
        break;
      }
      case OP_SOFTMAX: {
        assert(num_inputs == 1);
        int softmax_dim;
        dez.deserialize(softmax_dim);
        node = get_or_create_node<Softmax>(inputs[0], {softmax_dim});
        break;
      }
      case OP_TRANSPOSE: {
        node = Transpose::deserialize(*this, dez, inputs, num_inputs);
        break;
      }
      case OP_COMBINE: {
        assert(num_inputs == 1);
        int combine_dim, combine_degree;
        dez.deserialize(combine_dim);
        dez.deserialize(combine_degree);
        node = get_or_create_node<Combine>(inputs[0],
                                           {combine_dim, combine_degree});
        break;
      }
      case OP_REPARTITION: {
        assert(num_inputs == 1);
        int repartition_dim, repartition_degree;
        dez.deserialize(repartition_dim);
        dez.deserialize(repartition_degree);
        node = get_or_create_node<Repartition>(
            inputs[0], {repartition_dim, repartition_degree});
        break;
      }
      case OP_REPLICATE: {
        assert(num_inputs == 1);
        int replicate_dim, replicate_degree;
        dez.deserialize(replicate_dim);
        dez.deserialize(replicate_degree);
        node = get_or_create_node<Replicate>(inputs[0],
                                             {replicate_dim, replicate_degree});
        break;
      }
      case OP_REDUCTION: {
        assert(num_inputs == 1);
        int reduction_dim, reduction_degree;
        dez.deserialize(reduction_dim);
        dez.deserialize(reduction_degree);
        node = get_or_create_node<Reduction>(inputs[0],
                                             {reduction_dim, reduction_degree});
        break;
      }
      case OP_FUSED_PARALLEL: {
        assert(num_inputs == 1);
        std::vector<ParallelOpInfo> parallel_ops;
        int num_parallel_ops;
        dez.deserialize(num_parallel_ops);
        for (int i = 0; i < num_parallel_ops; i++) {
          ParallelOpInfo info;
          dez.deserialize(info);
          parallel_ops.push_back(info);
        }
        node = get_or_create_node<FusedParallelOp>(inputs[0], {parallel_ops});
        break;
      }
      default: {
        fprintf(stderr,
                "The following operator type is currently not supported"
                " for graph deserialization: %s\n"
                "Report the issue to the FlexFlow developers\n",
                get_operator_type_name(op_type).c_str());
        assert(false && "Unsupported operator type");
      }
    }
    {
      size_t safecode;
      dez.deserialize(safecode);
      assert(safecode == 12345678);
    }
    assert(node.ptr != nullptr);
    guid_to_nodes[guid] = node;
    for (size_t i = 0; i < num_inputs; i++) {
      inedges[i].dstOp = node;
      graph->add_edge(inedges[i]);
    }
  }
  // Second, deserialize optimal machine view
  size_t num_views;
  dez.deserialize(num_views);
  printf("views.size() = %zu\n", num_views);
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
  for (auto const &it : optimal_views) {
    printf("node[%zu]: type(%s) view(%d %d %d) ",
           it.first.guid,
           it.first.to_string().c_str(),
           it.second.ndims,
           it.second.dim[0],
           it.second.start_device_id);
    auto const &list = graph->inEdges.at(it.first);
    for (auto const &it2 : list) {
      Edge e = it2;
      printf(" inEdge(node(%zu) idx(%d))", e.srcOp.guid, e.srcIdx);
    }
    printf("\n");
  }
}

}; // namespace FlexFlow
