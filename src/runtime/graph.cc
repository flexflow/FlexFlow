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

const MachineView MachineView::NO_VIEW = MachineView();

const Node Node::INVALID_NODE = Node();

MachineView::MachineView()
: device_type(MachineView::GPU), ndims(0), start_device_id(0)
{
  for (int i = 0; i < MAX_TENSOR_DIM; i++) {
    dim[i] = stride[i] = 0;
  }
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
  ret = ret * 31 + std::hash<int>()(gpus_per_node);
  ret = ret * 31 + std::hash<int>()(cpus_per_node);
  return ret;
}

Node::Node(void)
: guid(0), ptr(NULL)
{}

std::string Node::op_to_string(const Op* ptr)
{
  switch (ptr->op_type) {
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
    default:
      return "Unknown_" + std::to_string(ptr->op_type);
  }
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

Graph::Graph(FFModel* _model)
: model(_model)
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
  for (const auto& it : inEdges) {
    if (it.first.guid == 0) continue;
    printf("	guid(%zu) type(%d): ", it.first.guid, it.first.ptr->op_type);
    const std::unordered_set<Edge>& list = it.second;
    for (const auto& it2 : list) {
      Edge e = it2;
      printf(" inEdge(guid(%zu) idx(%d))", e.srcOp.guid, e.srcIdx);
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

std::vector<MachineView>* FFModel::get_valid_machine_views(const Op* op)
{
  const auto& iter = cached_operator_valid_views.find(op->op_guid);
  if (iter != cached_operator_valid_views.end()) {
    return iter->second;
  } else {
    std::vector<MachineView>* valid_views = new std::vector<MachineView>();
    for (size_t i = 0; i < all_valid_views.size(); i++) {
      if (op->outputs[0]->is_valid_machine_view(all_valid_views[i]))
        valid_views->push_back(all_valid_views[i]);
    }
    cached_operator_valid_views[op->op_guid] = valid_views;
    return valid_views;
  }
}

void FFModel::register_machine_views()
{
  // Data parallel views
  for (int i = 1; i <= config.numNodes * config.workersPerNode; i++)
    if (config.numNodes * config.workersPerNode % i == 0) {
      MachineView view;
      view.device_type = MachineView::GPU;
      view.ndims = 1;
      view.dim[0] = i;
      view.stride[0] = 1;
      view.start_device_id = 0;
      all_valid_views.push_back(view);
    }
}

Node Graph::find_bottleneck_node(const Node& sink_node,
                                 const Node& source_node,
                                std::unordered_set<Node>& used_nodes) const
{
  std::unordered_map<Node, int> todos;
  std::unordered_map<Node, size_t> indices;
  // Need two queues
  std::vector<Node> queue;
  // Use queue2 to calculate the todos for each operator
  // Only input_edges are considered
  {
    std::vector<Node> queue2;
    std::unordered_set<Node> visited;
    size_t index = 0;
    queue2.push_back(sink_node);
    visited.insert(sink_node);
    while (index < queue2.size()) {
      Node node = queue2[index++];
      int cnt = 0;
      const auto& it = inEdges.find(node);
      if (it != inEdges.end()) {
        const auto& inList = it->second;
        for (const auto& e : inList) {
          // Ignore weight edge
          //if (e.weightEdge) continue;
          if (e.srcOp == source_node) continue;
          if (visited.find(e.srcOp) == visited.end()) {
            visited.insert(e.srcOp);
            queue2.push_back(e.srcOp);
          }
          cnt ++;
        }
      }
      todos[node] = cnt;
      if (cnt == 0) {
        queue.push_back(node);
      }
    }
  }
  size_t index = 0;
  while (index < queue.size()) {
    Node op = queue[index++];
    const auto& it = outEdges.find(op);
    if (it == outEdges.end())
      continue;
    const auto& outList = it->second;
    for (const auto& it2 : outList) {
      //if (it2.weightEdge) continue;
      todos[it2.dstOp] --;
      assert(todos[it2.dstOp] >= 0);
      if (todos[it2.dstOp] == 0) {
        indices[it2.dstOp] = queue.size();
        queue.push_back(it2.dstOp);
      }
    }
  }
  // assert(queue.size() == inEdges.size());
  // assert that sink_node must be the last one
  assert(indices[sink_node] == queue.size() - 1);
  std::vector<bool> available(queue.size(), true);
  for (const auto& it : inEdges) {
    const auto& inList = it.second;
    for (const auto& it2 : inList) {
      if (indices.find(it2.srcOp) == indices.end()) continue;
      if (indices.find(it2.dstOp) == indices.end()) continue;
      size_t start_idx = indices[it2.srcOp];
      size_t end_idx = indices[it2.dstOp];
      for (size_t i = start_idx+1; i < end_idx; i++)
        available[i] = false;
    }
  }
  Node bn_node = Node::INVALID_NODE;
  for (size_t i = 0; i < queue.size(); i++) {
    if (!available[i]) continue;
    if (queue[i].ptr->op_type == OP_INPUT) continue;
    if (queue[i].ptr->op_type == OP_WEIGHT) continue;
    bn_node = queue[i];
    break;
  }
  if (bn_node == Node::INVALID_NODE)
    return bn_node;
  used_nodes.insert(bn_node);
  queue.clear();
  queue.push_back(bn_node);
  index = 0;
  while (index < queue.size()) {
    Node op = queue[index++];
    const auto& it = inEdges.find(op);
    if (it == inEdges.end()) continue;
    const auto& inList = it->second;
    for (const auto& e : inList) {
      if (used_nodes.find(e.srcOp) == used_nodes.end()) {
        used_nodes.insert(e.srcOp);
        queue.push_back(e.srcOp);
      }
    }
  }
  return bn_node;
}

float FFModel::graph_cost(const Graph* graph,
                          const Node& sink_node,
                          const MachineView& sink_view,
                          const Node& source_node,
                          const MachineView& source_view,
                          const MachineResource& resources,
                          bool include_sink_compute_time)
{
  fprintf(stderr, "[DP] sink(%zu) sink_view(%d %d) source(%zu) source_view(%d %d)\n",
          sink_node.guid, sink_view.ndims, sink_view.dim[0],
          source_node.guid, source_view.ndims, source_view.dim[0]);
  graph->print();
  size_t hash = dp_state_hash(graph, sink_node, sink_view,
                              source_node, source_view, resources);
  assert(graph->inEdges.find(sink_node) != graph->inEdges.end());
  if (source_node != Node::INVALID_NODE)
    assert(graph->outEdges.find(source_node) != graph->outEdges.end());
  if (cached_graph_costs.find(hash) != cached_graph_costs.end()) {
    // cached_graph_costs does not include sink_compute_time
    if (include_sink_compute_time) {
      CostMetrics metrics = simulator->measure_operator_cost(sink_node.ptr, sink_view);
      return cached_graph_costs[hash]+metrics.forward_time+metrics.backward_time;
    } else 
      return cached_graph_costs[hash];
  }
  float cost = 1e7;
  if (graph->inEdges.size() <= 2) {
    if (source_node == Node::INVALID_NODE)
      cost = 0.0f;
    else {
      cost = 0.0f;
      const auto& inList = graph->inEdges.find(sink_node)->second;
      for (const auto& it2 : inList) {
        assert(it2.srcOp == source_node);
        assert(sink_node.ptr->inputs[it2.dstIdx]->is_valid_machine_view(source_view));
        cost += simulator->estimate_xfer_cost(source_node.ptr->outputs[it2.srcIdx],
                                              source_view, sink_view);
      }
    }
  } else {
    std::unordered_set<Node> used_nodes;
    Node bn_node = graph->find_bottleneck_node(sink_node, source_node, used_nodes);
    if (bn_node != Node::INVALID_NODE) {
      // We found a bottleneck node
      fprintf(stderr, " found bn_node = %zu\n", bn_node.guid);
      Graph* first_graph = new Graph(this);
      Graph* second_graph = new Graph(this);
      for (const auto& it : graph->inEdges) {
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
      std::vector<MachineView>* valid_views = get_valid_machine_views(bn_node.ptr);
      for (size_t i = 0; i < valid_views->size(); i++) {
        bool valid = true;
        MachineView bn_view = (*valid_views)[i];
        for (int j = 0; j < bn_node.ptr->numOutputs; j++) {
          if (!bn_node.ptr->outputs[j]->is_valid_machine_view(bn_view))
            valid = false;
        }
        if (!valid) continue;
        if (!resources.is_valid_machine_view(bn_view)) continue;
        fprintf(stderr, "       explore view(%d %d)\n", bn_view.ndims, bn_view.dim[0]);
        fprintf(stderr, "       First Graph\n");
        float first_cost = graph_cost(first_graph, bn_node, bn_view,
                                      source_node, source_view, resources, true);
        fprintf(stderr, "       Second Graph\n");
        float second_cost = graph_cost(second_graph, sink_node, sink_view, 
                                       bn_node, bn_view, resources, false);
        if (first_cost + second_cost < cost)
          cost = first_cost + second_cost;
      }
      delete first_graph;
      delete second_graph;
    } else {
      // sink node must have multiple branches
      // otherwise we should not be here
      assert(graph->inEdges.find(sink_node)->second.size() > 1);
      Graph* first_graph = new Graph(this);
      Graph* second_graph = new Graph(this);
      Node bn_node = Node::INVALID_NODE;
      // Find sink_node's first input
      {
        const auto& inList = graph->inEdges.find(sink_node)->second;
        for (const auto& it2 : inList) {
          if (it2.dstIdx != 0) continue;
          //if (it2.weightEdge) continue;
          bn_node = it2.srcOp;
        }
      }
      assert(bn_node != Node::INVALID_NODE);
      std::unordered_set<Node> used_node;
      std::vector<Node> queue;
      queue.push_back(bn_node);
      used_node.insert(bn_node);
      size_t i = 0;
      while (i < queue.size()) {
        Node node = queue[i++];
        const auto& inList = graph->inEdges.find(node)->second;
        for (const auto& it2 : inList) {
          if (used_nodes.find(it2.srcOp) == used_nodes.end()) {
            used_nodes.insert(it2.srcOp);
            queue.push_back(it2.srcOp);
          }
        }
      }
      for (const auto& it : graph->inEdges) {
        if (it.first == sink_node) continue;
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
      // Split sink_node's inedges between the two graphs
      {
        const auto& inList = graph->inEdges.find(sink_node)->second;
        for (const auto& it2 : inList) {
          if (used_nodes.find(it2.srcOp) != used_nodes.end()) {
            first_graph->add_edge(it2);
          } else {
            second_graph->add_edge(it2);
          }
        }
      }
      // Assert there must be at least on sink_source's inEdges in the second graph
      assert(second_graph->inEdges.find(sink_node) != second_graph->inEdges.end());
      // Run the two sequentially
      cost = graph_cost(first_graph, sink_node, sink_view, 
                        source_node, source_view, resources, false)
           + graph_cost(second_graph, sink_node, sink_view,
                        source_node, source_view, resources, false);
      // Consider run the two in parallel
      // Split resources vertically
      for (int i = 1; i < resources.num_nodes; i++) {
        MachineResource firstRes, secondRes;
        firstRes = resources; secondRes = resources;
        firstRes.num_nodes = i;
        secondRes.num_nodes = resources.num_nodes - i;
        float new_cost;
        new_cost = std::max(graph_cost(first_graph, sink_node, sink_view,
                                       source_node, source_view, firstRes, false),
                            graph_cost(second_graph, sink_node, sink_view,
                                       source_node, source_view, secondRes, false));
        if (new_cost < cost)
          cost = new_cost;
      }
      // Split resources horizontally
      for (int i = 1; i < resources.gpus_per_node; i++) {
        MachineResource firstRes, secondRes;
        firstRes = resources; secondRes = resources;
        firstRes.gpus_per_node = i;
        secondRes.gpus_per_node = resources.gpus_per_node - i;
        float new_cost;
        new_cost = std::max(graph_cost(first_graph, sink_node, sink_view,
                                       source_node, source_view, firstRes, false),
                            graph_cost(second_graph, sink_node, sink_view,
                                       source_node, source_view, secondRes, false));
        if (new_cost < cost)
          cost = new_cost;
      }
      delete first_graph;
      delete second_graph;
    }
  }
  cached_graph_costs[hash] = cost;
  if (include_sink_compute_time) {
    CostMetrics metrics = simulator->measure_operator_cost(sink_node.ptr, sink_view);
    cost += metrics.forward_time + metrics.backward_time;
  }
  return cost;
}

float Graph::total_cost(void)
{
  // Find sink_nodes
  // i.e., nodes with no out edge
  print();
  Node sink_node = Node::INVALID_NODE;
  for (const auto& it : outEdges) {
    const auto& outList = it.second;
    if (outList.size() == 0) {
      assert(sink_node == Node::INVALID_NODE);
      sink_node = it.first;
    }
  }
  assert(sink_node != Node::INVALID_NODE);
  MachineResource resource;
  resource.num_nodes = model->config.numNodes;
  resource.cpus_per_node = model->config.cpusPerNode;
  resource.gpus_per_node = model->config.workersPerNode;
  std::vector<MachineView>* valid_views = model->get_valid_machine_views(sink_node.ptr);
  float total_cost = 1e7;
  for (size_t i = 0; i < valid_views->size(); i++) {
    total_cost = std::min(total_cost,
                          model->graph_cost(this, sink_node, (*valid_views)[i],
                                            Node::INVALID_NODE, MachineView::NO_VIEW,
                                            resource, true));
  }
  return total_cost;
}

size_t Graph::hash(void) const
{
  // Graph hash should be additive and independent to the ordering of the nodes
  size_t total_hash = 0;
  for (const auto& it : inEdges) {
    const auto& inList = it.second;
    size_t node_hash = std::hash<size_t>()(it.first.guid);
    for (const auto& e : inList) {
      size_t edge_hash = 17;
      edge_hash = edge_hash * 31 + std::hash<size_t>()(e.srcOp.guid);
      edge_hash = edge_hash * 31 + std::hash<int>()(e.srcIdx);
      edge_hash = edge_hash * 31 + std::hash<int>()(e.dstIdx);
      node_hash *= edge_hash;
    }
    total_hash += node_hash;
  }
  return total_hash;
}

size_t FFModel::dp_state_hash(const Graph* graph,
                              const Node& sink_node,
                              const MachineView& sink_view,
                              const Node& source_node,
                              const MachineView& source_view,
                              const MachineResource& resource)
{
  size_t key = graph->hash();
  key = key * 31 + std::hash<size_t>()(sink_node.guid);
  key = key * 31 + sink_view.hash();
  key = key * 31 + std::hash<size_t>()(source_node.guid);
  key = key * 31 + source_view.hash();
  key = key * 31 + resource.hash();
  return key;
}

