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
#include "graph.h"
#include "dominators.h"
#include "op-attrs/op-attrs.h"
#include "utils/disjoint_set.h"
#include "utils/unique.h"
#include <iostream>

// using FlexFlow::utils::Node;
// using FlexFlow::opmeta::OperatorParameters;

namespace FlexFlow {

ParallelComputationGraph::Graph(std::string const &logger_name)
    : Graph(spdlog::get(logger_name)) {}

ParallelComputationGraph::Graph(std::shared_ptr<spdlog::logger> const &logger)
    : logger(logger) {}

Graph::Graph(utils::AdjacencyMultiDiGraph const &g,
             utils::bidict<Node, PCGOperatorAttrs> const &nodeMap,
             std::shared_ptr<spdlog::logger> const &logger)
    : g(g), nodeMap(nodeMap), logger(logger) {}

/* using namespace Legion; */
/* using FlexFlow::MachineView; */

/* LegionRuntime::Logger::Category log_graph("graph"); */
/* LegionRuntime::Logger::Category log_simplify("graph_simplify"); */

void Graph::add_edge(Node const &srcOp,
                     Node const &dstOp,
                     int srcIdx,
                     int dstIdx) {
  this->g.add_edge({srcOp, dstOp, (std::size_t)srcIdx, (std::size_t)dstIdx});
}

Node Graph::add_node(PCGOperatorAttrs const &params) {
  Node n = this->g.add_node();
  this->nodeMap.equate(n, params);
  return n;
}

void Graph::add_edge(utils::MultiDiEdge const &e) {
  this->g.add_edge(e);
}

void Graph::remove_edge(utils::MultiDiEdge const &e,
                        bool remove_node_if_unused) {
  this->g.remove_edge(e);
  utils::remove_node_if_unused(this->g, e.src);
  utils::remove_node_if_unused(this->g, e.dst);
}

bool Graph::has_edge(utils::MultiDiEdge const &e) const {
  return utils::contains_edge(this->g, e);
}

void Graph::print_dot() const {
  this->print_dot(std::cout);
}

void Graph::print_dot(std::ostream &s) const {
  auto directed = unsafe_view_as_digraph(this->g);

  DotFile<Node> dot(s);

  export_as_dot(dot, directed, [&](utils::Node const &node) -> RecordFormatter {
    RecordFormatter rf;
    rf << node.to_string();
    tl::optional<RecordFormatter> sub_rf = as_dot(this->nodeMap.at_l(node));
    if (sub_rf.has_value()) {
      rf << sub_rf.value();
    }

    return rf;
  });
  s << std::endl;
}

bool Graph::has_loop() {
  return !utils::is_acyclic(this->g).value_or(true);
}

/* Node Graph::find_bottleneck_node(Node const &sink_node, */
/*                                  Node const &source_node) const { */
/*   using FlexFlow::PCG::Utils::GraphStructure; */
/*   using FlexFlow::PCG::Utils::imm_post_dominators; */
/*   using FlexFlow::PCG::Utils::MultisourceGraphStructure; */
/*   using FlexFlow::PCG::Utils::roots; */

/*   Node source(source_node); */
/*   std::unordered_map<Node, Node> ipd; */
/*   std::unordered_set<Node> graph_roots = roots(*this); */
/*   if (source_node != Node::INVALID_NODE) { */
/*     ipd = imm_post_dominators(*this); */
/*   } else if (graph_roots.size() == 1) { */
/*     ipd = imm_post_dominators(*this); */
/*     source = *graph_roots.begin(); */
/*   } else { */
/*     ipd = imm_post_dominators<Graph,
 * MultisourceGraphStructure<Graph>>(*this); */
/*   } */

/*   Node bn_node = ipd.at(source); */
/*   if (bn_node == source || bn_node == sink_node) { */
/*     return Node::INVALID_NODE; */
/*   } */

/*   return bn_node; */
/* } */

Graph Graph::subgraph(std::unordered_set<Node> const &nodes) const {
  AdjacencyMultiDiGraph sub_g = subgraph<AdjacencyMultiDiGraph>(this->g, nodes);

  bidict<Node, opmeta::PCGOperatorAttrs> sub_nodeMap;
  for (auto const &kv : this->nodeMap) {
    if (contains(nodes, kv.first)) {
      sub_nodeMap.equate(kv.first, kv.second);
    }
  }

  return {sub_g, sub_nodeMap, this->logger};
}

void Graph::remove_node(Node const &node, bool purge_edges) {
  assert(purge_edges == true);
  utils::remove_node(this->g, node);
  this->nodeMap.erase_l(node);
}

/*static*/
Graph Graph::singleton(PCGOperatorAttrs const &params) {
  Graph g;
  g.add_node(params);
  return g;
}

bool Graph::empty() const {
  return utils::empty(this->g);
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
  contract_node(this->g, node);
  this->nodeMap.erase_l(node);
}

/* std::pair<std::unique_ptr<Graph>, std::unique_ptr<Graph>> */
/*     Graph::split_at_node(Node const &bottleneck) const { */
/*   using FlexFlow::PCGe:Utils::topo_sort; */

/*   auto first_graph = std::unique_ptr<Graph>(new Graph(this->model)); */
/*   auto second_graph = std::unique_ptr<Graph>(new Graph(this->model)); */

/*   std::unordered_set<Node> used_nodes; */
/*   { */
/*     std::vector<Node> topo_sorted; */
/*     topo_sort(*this, &topo_sorted); */

/*     for (auto const &node : topo_sorted) { */
/*       if (node == bottleneck) { */
/*         break; */
/*       } */

/*       used_nodes.insert(node); */
/*     } */
/*     used_nodes.insert(bottleneck); */

/*     assert(used_nodes.size() < topo_sorted.size()); */
/*   } */

/*   for (auto const &it : this->inEdges) { */
/*     auto const &inList = it.second; */
/*     if (used_nodes.find(it.first) != used_nodes.end()) { */
/*       // Add all in-edges of used_nodes in to the first_graph */
/*       for (auto const &it2 : inList) { */
/*         first_graph->add_edge(it2); */
/*       } */
/*     } else { */
/*       // Add all in-edges of not_used_nodes into the second_graph */
/*       for (auto const &it2 : inList) { */
/*         second_graph->add_edge(it2); */
/*       } */
/*     } */
/*   } */

/*   return {std::move(first_graph), std::move(second_graph)}; */
/* } */

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

bool GraphCostResult::operator<(GraphCostResult const &other) const {
  return this->cost < other.cost;
}

std::ostream &operator<<(std::ostream &s, GraphCostResult const &r) {
  s << "GraphCostResult{cost=" << r.cost << "}";
  return s;
}

std::ostream &operator<<(std::ostream &s, GraphOptimizeResult const &r) {
  s << "GraphOptimizeResult{cost=" << r.cost << "}";
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
float parallel_cost<float>(float const &first, float const &second) {
  return std::max(first, second);
}

float Graph::optimal_cost() const {
  return this->generic_optimal_cost<float>();
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
 * @tparam T the result type (can be either float or GraphCostResult)
 * @return T the cost of the graph (along with any additional data in the return
 * type)
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
  this->search->logger->info() << "Found sink node: " << sink_node.to_string();

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

GraphOptimalViewSerialized
    Graph::graph_optimize_task(Task const *task,
                               std::vector<PhysicalRegion> const &regions,
                               Context ctx,
                               Runtime *runtime) {
  FFModel *model = *((FFModel **)task->args);
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
  model->simulator =
      make_unique<Simulator>(model, model->handlers[0], gpu_mem, machine);
  std::unique_ptr<Graph> best_graph;
  std::unordered_map<Node, MachineView> optimal_views;
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
    best_graph = std::unique_ptr<Graph>(graph);
    MachineView data_parallel_view;
    data_parallel_view.device_type = MachineView::GPU;
    data_parallel_view.ndims = 1;
    data_parallel_view.dim[0] =
        model->config.numNodes * model->config.workersPerNode;
    data_parallel_view.stride[0] = 1;
    data_parallel_view.start_device_id = 0;
    for (auto const &node : best_graph->inEdges) {
      optimal_views[node.first] = data_parallel_view;
    }
  } else {
    model->graph_optimize(model->config.search_budget,
                          model->config.only_data_parallel,
                          best_graph,
                          optimal_views);
  }
  /* Serializer sez; */
  /* // First serialize graph */
  /* sez.serialize(best_graph->inEdges.size()); */
  /* std::unordered_map<Node, int> todos; */
  /* std::vector<Node> opList; */
  /* for (auto const &it : best_graph->inEdges) { */
  /*   auto const &inList = it.second; */
  /*   todos[it.first] = (int)inList.size(); */
  /*   if (todos[it.first] == 0) { */
  /*     opList.push_back(it.first); */
  /*   } */
  /* } */
  /* size_t node_idx = 0; */
  /* while (node_idx < opList.size()) { */
  /*   Node cur_node = opList[node_idx++]; */
  /*   auto const &outList = best_graph->outEdges[cur_node]; */
  /*   for (auto const &e : outList) { */
  /*     todos[e.dstOp]--; */
  /*     if (todos[e.dstOp] == 0) { */
  /*       opList.push_back(e.dstOp); */
  /*     } */
  /*   } */
  /*   auto const &inList = best_graph->inEdges[cur_node]; */
  /*   sez.serialize(inList.size()); */
  /*   for (auto const &e : inList) { */
  /*     sez.serialize(e.srcOp.guid); */
  /*     assert(e.dstOp.guid == cur_node.guid); */
  /*     sez.serialize(e.srcIdx); */
  /*     sez.serialize(e.dstIdx); */
  /*   } */
  /*   sez.serialize((size_t)10101010); // safe guard for the end of inedges */
  /*   Op const *op = cur_node.ptr; */
  /*   assert(op != NULL); */
  /*   sez.serialize(cur_node.guid); */
  /*   sez.serialize(op->op_type); */
  /*   switch (op->op_type) { */
  /*     case OP_INPUT: { */
  /*       assert(op->numOutputs == 1); */
  /*       NoOp *noop = (NoOp *)op; */
  /*       sez.serialize(noop->op_type); */
  /*       sez.serialize(noop->input_tensor_guid); */
  /*       sez.serialize(noop->outputs[0]->data_type); */
  /*       sez.serialize(noop->outputs[0]->num_dims); */
  /*       for (int i = 0; i < noop->outputs[0]->num_dims; i++) { */
  /*         sez.serialize(noop->outputs[0]->dims[i]); */
  /*       } */
  /*       break; */
  /*     } */
  /*     case OP_NOOP: { */
  /*       break; */
  /*     } */
  /*     case OP_CONCAT: { */
  /*       Concat *concat = (Concat *)op; */
  /*       sez.serialize(concat->legion_axis); */
  /*       break; */
  /*     } */
  /*     case OP_SPLIT: { */
  /*       Split *split = (Split *)op; */
  /*       sez.serialize(split->legion_axis); */
  /*       sez.serialize(split->numOutputs); */
  /*       for (int i = 0; i < split->numOutputs; i++) { */
  /*         sez.serialize(split->outputs[i]->dims[split->legion_axis].size); */
  /*       } */
  /*       break; */
  /*     } */
  /*     case OP_EMBEDDING: { */
  /*       Embedding *embed = (Embedding *)op; */
  /*       sez.serialize(embed->layer_guid.id); */
  /*       sez.serialize(embed->num_entries); */
  /*       sez.serialize(embed->out_channels); */
  /*       sez.serialize(embed->aggr); */
  /*       sez.serialize(embed->data_type); */
  /*       break; */
  /*     } */
  /*     case OP_EW_ADD: */
  /*     case OP_EW_SUB: */
  /*     case OP_EW_MUL: */
  /*     case OP_EW_MAX: */
  /*     case OP_EW_MIN: { */
  /*       sez.serialize(op->op_type); */
  /*       break; */
  /*     } */
  /*     case OP_MULTIHEAD_ATTENTION: { */
  /*       MultiHeadAttention *attn = (MultiHeadAttention *)op; */
  /*       sez.serialize(attn->layer_guid.id); */
  /*       sez.serialize(attn->oProjSize); */
  /*       sez.serialize(attn->num_heads); */
  /*       sez.serialize(attn->qProjSize); */
  /*       sez.serialize(attn->vProjSize); */
  /*       sez.serialize(attn->dropout); */
  /*       sez.serialize(attn->bias); */
  /*       sez.serialize(attn->add_bias_kv); */
  /*       sez.serialize(attn->add_zero_attn); */
  /*       break; */
  /*     } */
  /*     case OP_SOFTMAX: { */
  /*       Softmax *softmax = (Softmax *)op; */
  /*       sez.serialize(softmax->dim); */
  /*       break; */
  /*     } */
  /*     case OP_REPARTITION: { */
  /*       Repartition *repart = (Repartition *)op; */
  /*       sez.serialize(repart->repartition_dim); */
  /*       sez.serialize(repart->repartition_degree); */
  /*       break; */
  /*     } */
  /*     case OP_REPLICATE: { */
  /*       Replicate *replicate = (Replicate *)op; */
  /*       sez.serialize(replicate->replicate_dim); */
  /*       sez.serialize(replicate->replicate_degree); */
  /*       break; */
  /*     } */
  /*     case OP_REDUCTION: { */
  /*       Reduction *reduction = (Reduction *)op; */
  /*       sez.serialize(reduction->reduction_dim); */
  /*       sez.serialize(reduction->reduction_degree); */
  /*       break; */
  /*     } */
  /*     case OP_COMBINE: { */
  /*       Combine *combine = (Combine *)op; */
  /*       sez.serialize(combine->combine_dim); */
  /*       sez.serialize(combine->combine_degree); */
  /*       break; */
  /*     } */
  /*     case OP_FUSED_PARALLEL: { */
  /*       FusedParallelOp *fused = (FusedParallelOp *)op; */
  /*       sez.serialize(fused->num_parallel_ops); */
  /*       for (int i = 0; i < fused->num_parallel_ops; i++) { */
  /*         sez.serialize(fused->parallel_ops[i]); */
  /*       } */
  /*       break; */
  /*     } */
  /*     default: { */
  /*       op->serialize(sez); */
  /*     } */
  /*   } */
  /*   sez.serialize((size_t)12345678); // safe guard for the end of an op */
  /* } */
  /* assert(node_idx == best_graph->inEdges.size()); */
  /* // Second, serialize optimal machine view */
  /* printf("opotimal_views.size = %zu\n", optimal_views.size()); */
  /* sez.serialize(optimal_views.size()); */
  /* for (auto const &it : optimal_views) { */
  /*   sez.serialize((size_t)98765432); // safe guard */
  /*   sez.serialize(it.first.guid); */
  /*   sez.serialize(it.second); */
  /* } */
  /* assert(sez.get_used_bytes() < GraphOptimalViewSerialized::buffer_size); */
  /* GraphOptimalViewSerialized ret; */
  /* ret.total_bytes = sez.get_used_bytes(); */
  /* memcpy(ret.data, sez.get_buffer(), ret.total_bytes); */
  /* // Deallocate best_graph */
  /* // delete best_graph; */
  /* return ret; */
}

}; // namespace FlexFlow

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

/* void FFModel::deserialize_graph_optimal_view( */
/*     Legion::Deserializer &dez, */
/*     Graph *graph, */
/*     std::unordered_map<Node, MachineView> &optimal_views) { */
/*   // Deserializer dez(serialized.data, serialized.total_bytes); */
/*   std::unordered_map<size_t, Node> guid_to_nodes; */
/*   size_t num_nodes; */
/*   dez.deserialize(num_nodes); */
/*   // best_graph = new Graph(this); */
/*   for (size_t node_idx = 0; node_idx < num_nodes; node_idx++) { */
/*     Edge inedges[MAX_NUM_INPUTS]; */
/*     ParallelTensor inputs[MAX_NUM_INPUTS]; */
/*     size_t num_inputs; */
/*     dez.deserialize(num_inputs); */
/*     for (size_t j = 0; j < num_inputs; j++) { */
/*       size_t src_guid; */
/*       int src_idx, dst_idx; */
/*       dez.deserialize(src_guid); */
/*       assert(guid_to_nodes.find(src_guid) != guid_to_nodes.end()); */
/*       dez.deserialize(src_idx); */
/*       dez.deserialize(dst_idx); */
/*       assert(dst_idx < (int)num_inputs); */
/*       inedges[dst_idx].srcOp = guid_to_nodes[src_guid]; */
/*       inedges[dst_idx].srcIdx = src_idx; */
/*       inedges[dst_idx].dstIdx = dst_idx; */
/*       inputs[dst_idx] = inedges[dst_idx].srcOp.ptr->outputs[src_idx]; */
/*     } */
/*     { */
/*       size_t safecode; */
/*       dez.deserialize(safecode); */
/*       assert(safecode == 10101010); */
/*     } */
/*     Node node = Node::INVALID_NODE; */
/*     size_t guid; */
/*     OperatorType op_type; */
/*     dez.deserialize(guid); */
/*     dez.deserialize(op_type); */
/*     switch (op_type) { */
/*       case OP_INPUT: { */
/*         assert(num_inputs == 0); */
/*         int num_dims; */
/*         ParallelDim dims[MAX_TENSOR_DIM]; */
/*         OperatorType op_type; */
/*         dez.deserialize(op_type); */
/*         size_t input_tensor_guid; */
/*         dez.deserialize(input_tensor_guid); */
/*         DataType data_type; */
/*         dez.deserialize(data_type); */
/*         dez.deserialize(num_dims); */
/*         for (int i = 0; i < num_dims; i++) { */
/*           dez.deserialize(dims[i]); */
/*         } */
/*         ParallelTensor t = */
/*             create_parallel_tensor_legion_ordering(num_dims, */
/*                                                    dims, */
/*                                                    data_type, */
/*                                                    nullptr, */
/*                                                    0, */
/*                                                    true create_grad, */
/*                                                    input_tensor_guid); */
/*         node.ptr = t->owner_op; */
/*         node.guid = node_global_guid++; */
/*         break; */
/*       } */
/*       case OP_NOOP: { */
/*         assert(num_inputs == 1); */
/*         node = get_or_create_noop_node(inputs[0]); */
/*         break; */
/*       } */
/*       case OP_BATCHMATMUL: { */
/*         node = BatchMatmul::deserialize(*this, dez, inputs, num_inputs); */
/*         break; */
/*       } */
/*       case OP_CAST: { */
/*         node = Cast::deserialize(*this, dez, inputs, num_inputs); */
/*         break; */
/*       } */
/*       case OP_CONCAT: { */
/*         int legion_axis; */
/*         dez.deserialize(legion_axis); */
/*         node = get_or_create_node<Concat>( */
/*             {std::begin(inputs), std::begin(inputs) + num_inputs}, */
/*             {legion_axis}); */
/*         break; */
/*       } */
/*       case OP_SPLIT: { */
/*         int legion_axis; */
/*         dez.deserialize(legion_axis); */
/*         int num_outputs; */
/*         dez.deserialize(num_outputs); */
/*         std::vector<int> splits; */
/*         for (int i = 0; i < num_outputs; i++) { */
/*           int dim_size; */
/*           dez.deserialize(dim_size); */
/*           splits.push_back(dim_size); */
/*         } */
/*         node = get_or_create_node<Split>(inputs[0], {splits, legion_axis});
 */
/*         break; */
/*       } */
/*       case OP_EMBEDDING: { */
/*         assert(num_inputs == 1); */
/*         AggrMode aggr; */
/*         int num_entries, out_channels; */
/*         size_t id; */
/*         DataType data_type; */
/*         dez.deserialize(id); */
/*         LayerID layer_guid(id); */
/*         dez.deserialize(num_entries); */
/*         dez.deserialize(out_channels); */
/*         dez.deserialize(aggr); */
/*         dez.deserialize(data_type); */

/*         EmbeddingParams params; */
/*         params.aggr = aggr; */
/*         params.num_entries = num_entries; */
/*         params.out_channels = out_channels; */
/*         params.layer_guid = layer_guid; */
/*         params.data_type = data_type; */
/*         node = get_or_create_node<Embedding>(inputs[0], params); */
/*         break; */
/*       } */
/*       case OP_EW_ADD: */
/*       case OP_EW_SUB: */
/*       case OP_EW_MUL: */
/*       case OP_EW_MAX: */
/*       case OP_EW_MIN: { */
/*         assert(num_inputs == 2); */
/*         OperatorType op_type; */
/*         dez.deserialize(op_type); */
/*         node = get_or_create_node<ElementBinary>({inputs[0], inputs[1]}, */
/*                                                  {op_type}); */
/*         break; */
/*       } */
/*       case OP_CONV2D: { */
/*         node = Conv2D::deserialize(*this, dez, inputs, num_inputs); */
/*         break; */
/*       } */
/*       case OP_DROPOUT: { */
/*         node = Dropout::deserialize(*this, dez, inputs, num_inputs); */
/*         break; */
/*       } */
/*       case OP_EXP: */
/*       case OP_SIN: */
/*       case OP_COS: */
/*       case OP_SCALAR_MULTIPLY: */
/*       case OP_SCALAR_FLOOR_DIV: */
/*       case OP_SCALAR_TRUE_DIV: */
/*       case OP_SCALAR_ADD: */
/*       case OP_SCALAR_SUB: */
/*       case OP_RELU: */
/*       case OP_SIGMOID: */
/*       case OP_TANH: */
/*       case OP_POW: */
/*       case OP_IDENTITY: */
/*       case OP_GELU: */
/*       case OP_ELU: { */
/*         node = ElementUnary::deserialize(*this, dez, inputs, num_inputs); */
/*         break; */
/*       } */
/*       case OP_FLAT: { */
/*         node = Flat::deserialize(*this, dez, inputs, num_inputs); */
/*         break; */
/*       } */
/*       case OP_GATHER: { */
/*         node = Gather::deserialize(*this, dez, inputs, num_inputs); */
/*         break; */
/*       } */
/*       case OP_LAYERNORM: { */
/*         node = LayerNorm::deserialize(*this, dez, inputs, num_inputs); */
/*         break; */
/*       } */
/*       case OP_LINEAR: { */
/*         node = Linear::deserialize(*this, dez, inputs, num_inputs); */
/*         break; */
/*       } */
/*       case OP_MULTIHEAD_ATTENTION: { */
/*         assert(num_inputs == 3); */
/*         int embed_dim, num_heads, k_dim, v_dim; */
/*         float dropout; */
/*         bool bias, add_bias_kv, add_zero_attn; */
/*         size_t id; */
/*         dez.deserialize(id); */
/*         LayerID layer_guid(id); */
/*         dez.deserialize(embed_dim); */
/*         dez.deserialize(num_heads); */
/*         dez.deserialize(k_dim); */
/*         dez.deserialize(v_dim); */
/*         dez.deserialize(dropout); */
/*         dez.deserialize(bias); */
/*         dez.deserialize(add_bias_kv); */
/*         dez.deserialize(add_zero_attn); */

/*         MultiHeadAttentionParams params; */
/*         params.embed_dim = embed_dim; */
/*         params.num_heads = num_heads; */
/*         params.kdim = k_dim; */
/*         params.vdim = v_dim; */
/*         params.dropout = dropout; */
/*         params.bias = bias; */
/*         params.add_bias_kv = add_bias_kv; */
/*         params.add_zero_attn = add_zero_attn; */
/*         params.layer_guid = layer_guid; */
/*         node = get_or_create_node<MultiHeadAttention>( */
/*             {inputs[0], inputs[1], inputs[2]}, params); */
/*         break; */
/*       } */
/*       case OP_TOPK: { */
/*         node = TopK::deserialize(*this, dez, inputs, num_inputs); */
/*         break; */
/*       } */
/*       case OP_GROUP_BY: { */
/*         node = Group_by::deserialize(*this, dez, inputs, num_inputs); */
/*         break; */
/*       } */
/*       case OP_AGGREGATE: { */
/*         // node = Aggregate::deserialize(*this, dez, inputs, num_inputs); */
/*         int n; */
/*         float lambda_bal; */
/*         dez.deserialize(n); */
/*         dez.deserialize(lambda_bal); */
/*         assert(num_inputs == n + 4); */
/*         AggregateParams params; */
/*         params.n = n; */
/*         params.lambda_bal = lambda_bal; */
/*         node = get_or_create_node<Aggregate>( */
/*             {std::begin(inputs), std::begin(inputs) + num_inputs}, params);
 */
/*         break; */
/*       } */
/*       case OP_POOL2D: { */
/*         node = Pool2D::deserialize(*this, dez, inputs, num_inputs); */
/*         break; */
/*       } */
/*       case OP_REDUCE_SUM: { */
/*         node = Reduce::deserialize(*this, dez, inputs, num_inputs); */
/*         break; */
/*       } */
/*       case OP_RESHAPE: { */
/*         node = Reshape::deserialize(*this, dez, inputs, num_inputs); */
/*         break; */
/*       } */
/*       case OP_SOFTMAX: { */
/*         assert(num_inputs == 1); */
/*         int softmax_dim; */
/*         dez.deserialize(softmax_dim); */
/*         node = get_or_create_node<Softmax>(inputs[0], {softmax_dim}); */
/*         break; */
/*       } */
/*       case OP_TRANSPOSE: { */
/*         node = Transpose::deserialize(*this, dez, inputs, num_inputs); */
/*         break; */
/*       } */
/*       case OP_COMBINE: { */
/*         assert(num_inputs == 1); */
/*         int combine_dim, combine_degree; */
/*         dez.deserialize(combine_dim); */
/*         dez.deserialize(combine_degree); */
/*         node = get_or_create_node<Combine>(inputs[0], */
/*                                            {combine_dim, combine_degree}); */
/*         break; */
/*       } */
/*       case OP_REPARTITION: { */
/*         assert(num_inputs == 1); */
/*         int repartition_dim, repartition_degree; */
/*         dez.deserialize(repartition_dim); */
/*         dez.deserialize(repartition_degree); */
/*         node = get_or_create_node<Repartition>( */
/*             inputs[0], {repartition_dim, repartition_degree}); */
/*         break; */
/*       } */
/*       case OP_REPLICATE: { */
/*         assert(num_inputs == 1); */
/*         int replicate_dim, replicate_degree; */
/*         dez.deserialize(replicate_dim); */
/*         dez.deserialize(replicate_degree); */
/*         node = get_or_create_node<Replicate>(inputs[0], */
/*                                              {replicate_dim,
 * replicate_degree}); */
/*         break; */
/*       } */
/*       case OP_REDUCTION: { */
/*         assert(num_inputs == 1); */
/*         int reduction_dim, reduction_degree; */
/*         dez.deserialize(reduction_dim); */
/*         dez.deserialize(reduction_degree); */
/*         node = get_or_create_node<Reduction>(inputs[0], */
/*                                              {reduction_dim,
 * reduction_degree}); */
/*         break; */
/*       } */
/*       case OP_FUSED_PARALLEL: { */
/*         assert(num_inputs == 1); */
/*         std::vector<ParallelOpInfo> parallel_ops; */
/*         int num_parallel_ops; */
/*         dez.deserialize(num_parallel_ops); */
/*         for (int i = 0; i < num_parallel_ops; i++) { */
/*           ParallelOpInfo info; */
/*           dez.deserialize(info); */
/*           parallel_ops.push_back(info); */
/*         } */
/*         node = get_or_create_node<FusedParallelOp>(inputs[0],
 * {parallel_ops}); */
/*         break; */
/*       } */
/*       default: { */
/*         fprintf(stderr, */
/*                 "The following operator type is currently not supported" */
/*                 " for graph deserialization: %s\n" */
/*                 "Report the issue to the FlexFlow developers\n", */
/*                 get_operator_type_name(op_type).c_str()); */
/*         assert(false && "Unsupported operator type"); */
/*       } */
/*     } */
/*     { */
/*       size_t safecode; */
/*       dez.deserialize(safecode); */
/*       assert(safecode == 12345678); */
/*     } */
/*     assert(node.ptr != nullptr); */
/*     guid_to_nodes[guid] = node; */
/*     for (size_t i = 0; i < num_inputs; i++) { */
/*       inedges[i].dstOp = node; */
/*       graph->add_edge(inedges[i]); */
/*     } */
/*   } */
/*   // Second, deserialize optimal machine view */
/*   size_t num_views; */
/*   dez.deserialize(num_views); */
/*   printf("views.size() = %zu\n", num_views); */
/*   for (size_t i = 0; i < num_views; i++) { */
/*     size_t safecode, guid; */
/*     MachineView view; */
/*     dez.deserialize(safecode); */
/*     assert(safecode == 98765432); */
/*     dez.deserialize(guid); */
/*     assert(guid_to_nodes.find(guid) != guid_to_nodes.end()); */
/*     dez.deserialize(view); */
/*     optimal_views[guid_to_nodes[guid]] = view; */
/*   } */
/*   assert(dez.get_remaining_bytes() == 0); */
/*   printf("Deserialized Views...\n"); */
/*   for (auto const &it : optimal_views) { */
/*     printf("node[%zu]: type(%s) view(%d %d %d) ", */
/*            it.first.guid, */
/*            it.first.to_string().c_str(), */
/*            it.second.ndims, */
/*            it.second.dim[0], */
/*            it.second.start_device_id); */
/*     auto const &list = graph->inEdges.at(it.first); */
/*     for (auto const &it2 : list) { */
/*       Edge e = it2; */
/*       printf(" inEdge(node(%zu) idx(%d))", e.srcOp.guid, e.srcIdx); */
/*     } */
/*     printf("\n"); */
/*   } */
/* } */

} // namespace FlexFlow
