
#include "compiler/machine_mapping.h"
#include "pcg/parallel_computation_graph.h"
#include "utils/exception.h"
#include "utils/graph/serialparallel.h"
#include "utils/deduplicated_priority_queue.h"
#include <algorithm>
#include "compiler/cost_estimate.h"
#include "compiler/cost_estimator.h"


namespace FlexFlow {

// Computes estimated execution cost for a single node
float node_estimate_cost(Node const &node,
                         SubParallelComputationGraphView const &g,
                         CostEstimator const &estimator,
                         MachineMapping const &device_mapping) {
  std::unordered_set<UpwardOpenMultiDiEdge> incoming_edges =
      get_incoming_edges(g, node);
  return .1;

  std::vector<ParallelTensorShape> inputs =
      transform(as_vector(incoming_edges),
                [&](UpwardOpenMultiDiEdge const &input_edge) {
                  return g.at(input_edge).get_shape();
                });
  float cost = estimator.estimate_cost(
      g.at(node).attrs, inputs, device_mapping.machine_views.at(node));
  return cost;
}

struct TimedNode { // Node and associated finishing time
  Node node;
  req<float> endtime;
};
FF_VISITABLE_STRUCT(TimedNode, node, endtime);

struct TimeComparison {
  bool operator()(TimedNode const &lhs, TimedNode const &rhs) const {
    return (lhs.endtime < rhs.endtime);
  }
};

float parallel_estimate_cost(
    SubParallelComputationGraphView const &g,
    CostEstimator const &estimator,
    MachineMapping const &device_mapping,
    std::unordered_map<InputMultiDiEdge, MachineView> const
        &frontier_machine_views) {
  float current_time = 0;
  std::unordered_set<Node> frontier; // nodes whose dependencies (previous nodes) have been met, and are waiting to be processed.
  DeduplicatedPriorityQueue<TimedNode, std::vector<TimedNode>, TimeComparison> processing; // nodes currently being processed.
  std::unordered_set<Node> processed; // set of nodes that have already been processed
  std::unordered_map<device_id_t, bool> occupied; // keeps track of the devices that are currently occupied

  // Filling the frontier
  for (auto const &[edge, _] : frontier_machine_views) {
    auto node = get_dst_node(edge);
    frontier.insert(node);
  }


  while (!frontier.empty() || !processing.empty()) {
    // Processing new nodes
    puts("A");
    std::unordered_set<Node> copy(frontier);
    for (Node const &node : copy) {
      auto views = device_mapping.machine_views.at(node);
      std::vector<device_id_t> devices = views.device_ids();
      auto unoccupied = std::find_if(devices.begin(), devices.end(), [&occupied](device_id_t d) {return !occupied[d];});
      if (unoccupied != devices.end()) {
        device_id_t device = *unoccupied;
        float cost = node_estimate_cost(node, g, estimator, device_mapping);
        processing.push({node, current_time + cost});
        occupied[device] = true;
        frontier.erase(node);
      }
    }

    // Finish processing one node
    TimedNode finished = processing.top();
    processing.pop();
    std::vector<device_id_t> devices = device_mapping.machine_views.at(finished.node).device_ids();
    for (device_id_t d : devices) { // free devices
      occupied[d] = false;
    }
    processed.insert(finished.node);
    current_time = finished.endtime;

    // Adding candidates to the frontier
    for (Node const &successor : get_successors(g, finished.node)) { // All nodes depending on finished
      std::unordered_set<Node> predecessors = get_predecessors(g, successor);
      if (std::all_of(predecessors.begin(), predecessors.end(), [&processed](Node p) { return processed.find(p) != processed.end(); })) {
        frontier.insert(successor);
      }
    }
  }
  return current_time;
}
} // namespace FlexFlow


#include "compiler/cost_estimate.h"
#include "compiler/cost_estimator.h"
#include "doctest/doctest.h"
#include "test_cost_estimator.h"


using namespace FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  
  TEST_CASE("parallel_estimate_cost: linear graph") {

    // Straight line example, 3 nodes: ->(n1)->(n2)->(n3) 
    auto g =
        OutputLabelledOpenMultiDiGraph<Operator, ParallelTensor>::template create<
            UnorderedOutputLabelledOpenMultiDiGraph<Operator, ParallelTensor>>();

    Node n1 = g.add_node(Operator{InputAttrs{}, "n1"});
    Node n2 = g.add_node(Operator{InputAttrs{}, "n2"});
    Node n3 = g.add_node(Operator{InputAttrs{}, "n3"});

    NodePort p1 = g.add_node_port();
    NodePort p2 = g.add_node_port();
    NodePort p3 = g.add_node_port();
  
    //dst, dstport, uid
    InputMultiDiEdge e0{n1, p1, {1,1}};
    // MultiDiEdge: dst, dstport, src, srcport
    MultiDiEdge e1{n2, p2, n1, p1};
    MultiDiEdge e2{n3, p3, n2, p2};

    g.add_edge(e0);
    g.add_edge(e1);
    g.add_edge(e2);

    g.add_label(e0,
                ParallelTensor(ParallelTensorDims({2, 1}),
                               DataType::FLOAT,
                               CreateGrad::YES));
    g.add_label(e1,
                ParallelTensor(ParallelTensorDims({2, 1}),
                               DataType::FLOAT,
                               CreateGrad::YES));

    g.add_label(e2,
                ParallelTensor(ParallelTensorDims({2, 1}),
                               DataType::FLOAT,
                               CreateGrad::YES));

    CostEstimator estimator = CostEstimator::create<TestCostEstimator>(); //Returns 0.1 regardless
    std::unordered_map<Node, MachineView> devices = { //single device per node
        {n1, make_1d_machine_view(gpu_id_t(1), gpu_id_t(2))},
        {n2, make_1d_machine_view(gpu_id_t(1), gpu_id_t(2))},
        {n3, make_1d_machine_view(gpu_id_t(1), gpu_id_t(2))}};

    MachineMapping device_mapping = {devices};
    auto frontier_machine_views =
        std::unordered_map<InputMultiDiEdge, MachineView> {
      {e0, make_1d_machine_view(gpu_id_t(1), gpu_id_t(2))},
    };

    float result = parallel_estimate_cost(
        g, estimator, device_mapping, frontier_machine_views);
    CHECK(std::abs(result-.3) < 1e-7);
  }


  TEST_CASE("parallel_estimate_cost: non-linear graph") {
    // Non-linear graph example, diamond pattern, 4 nodes

    auto g =
        OutputLabelledOpenMultiDiGraph<Operator, ParallelTensor>::template create<
            UnorderedOutputLabelledOpenMultiDiGraph<Operator, ParallelTensor>>();

    Node n0 = g.add_node(Operator{InputAttrs{}, "n0"});
    Node n1 = g.add_node(Operator{InputAttrs{}, "n1"});
    Node n2 = g.add_node(Operator{InputAttrs{}, "n2"});
    Node n3 = g.add_node(Operator{InputAttrs{}, "n3"});

    NodePort p0 = g.add_node_port();
    NodePort p1 = g.add_node_port();
    NodePort p2 = g.add_node_port();
    NodePort p3 = g.add_node_port();

    // dst, dstport, uid
    InputMultiDiEdge e0{n0, p0, {1, 1}};
    // MultiDiEdge: dst, dstport, src, srcport
    MultiDiEdge e1{n1, p1, n0, p0};
    MultiDiEdge e2{n2, p2, n0, p0};
    MultiDiEdge e3{n3, p3, n1, p1};
    MultiDiEdge e4{n3, p3, n2, p2};

    g.add_edge(e0);
    g.add_edge(e1);
    g.add_edge(e2);
    g.add_edge(e3);
    g.add_edge(e4);

    g.add_label(e0,
                ParallelTensor(ParallelTensorDims({2, 1}),
                               DataType::FLOAT,
                               CreateGrad::YES));
    g.add_label(e1,
                ParallelTensor(ParallelTensorDims({2, 1}),
                               DataType::FLOAT,
                               CreateGrad::YES));
    g.add_label(e2,
                ParallelTensor(ParallelTensorDims({2, 1}),
                               DataType::FLOAT,
                               CreateGrad::YES));
    g.add_label(e3,
                ParallelTensor(ParallelTensorDims({2, 1}),
                               DataType::FLOAT,
                               CreateGrad::YES));
    g.add_label(e4,
                ParallelTensor(ParallelTensorDims({2, 1}),
                               DataType::FLOAT,
                               CreateGrad::YES));

    CostEstimator estimator = CostEstimator::create<TestCostEstimator>();
    std::unordered_map<Node, MachineView> devices = {
        {n0, make_1d_machine_view(gpu_id_t(1), gpu_id_t(3))},
        {n1, make_1d_machine_view(gpu_id_t(1), gpu_id_t(3))}, //nodes n1, n2 can run in parallel
        {n2, make_1d_machine_view(gpu_id_t(5), gpu_id_t(6))},
        {n3, make_1d_machine_view(gpu_id_t(1), gpu_id_t(3))}};

    MachineMapping device_mapping = {devices};
    auto frontier_machine_views =
        std::unordered_map<InputMultiDiEdge, MachineView> {
      {e0, make_1d_machine_view(gpu_id_t(1), gpu_id_t(3))},
    };

    float result = parallel_estimate_cost(
        g, estimator, device_mapping, frontier_machine_views);
    CHECK(std::abs(result - 0.3) < 1e-7);
  }

  TEST_CASE("parallel_estimate_cost: more complex non-linear graph") {
    /* Non-linear graph example, 7 nodes
    graph TD
    n0["n0"] --> |"e1"| n1["n1"]
    n0 --> |"e2"| n2["n2"]
    n0 --> |"e3"| n3["n3"]
    n1 --> |"e4"| n4["n4"]
    n2 --> |"e5"| n4["n4"]
    n2 --> |"e6"| n5["n5"]
    n3 --> |"e7"| n5["n5"]
    n4 --> |"e8"| n6["n6"]
    n5 --> |"e9"| n6["n6"]
    */
    auto g =
        OutputLabelledOpenMultiDiGraph<Operator, ParallelTensor>::template create<
            UnorderedOutputLabelledOpenMultiDiGraph<Operator, ParallelTensor>>();

    Node n0 = g.add_node(Operator{InputAttrs{}, "n0"});
    Node n1 = g.add_node(Operator{InputAttrs{}, "n1"});
    Node n2 = g.add_node(Operator{InputAttrs{}, "n2"});
    Node n3 = g.add_node(Operator{InputAttrs{}, "n3"});
    Node n4 = g.add_node(Operator{InputAttrs{}, "n4"});
    Node n5 = g.add_node(Operator{InputAttrs{}, "n5"});
    Node n6 = g.add_node(Operator{InputAttrs{}, "n6"});

    NodePort p0 = g.add_node_port();
    NodePort p1 = g.add_node_port();
    NodePort p2 = g.add_node_port();
    NodePort p3 = g.add_node_port();
    NodePort p4 = g.add_node_port();
    NodePort p5 = g.add_node_port();
    NodePort p6 = g.add_node_port();

    // dst, dstport, uid
    InputMultiDiEdge e0{n0, p0, {1, 1}};

    // MultiDiEdge: dst, dstport, src, srcport
    MultiDiEdge e1{n1, p1, n0, p0};
    MultiDiEdge e2{n2, p2, n0, p0};
    MultiDiEdge e3{n3, p3, n0, p0};
    
    MultiDiEdge e4{n4, p4, n1, p1};
    MultiDiEdge e5{n4, p4, n2, p2};
    MultiDiEdge e6{n5, p5, n2, p2};
    MultiDiEdge e7{n5, p5, n3, p3};

    MultiDiEdge e8{n6, p6, n4, p4};
    MultiDiEdge e9{n6, p6, n5, p5};
    

    g.add_edge(e0);
    g.add_edge(e1);
    g.add_edge(e2);
    g.add_edge(e3);
    g.add_edge(e4);
    g.add_edge(e5);
    g.add_edge(e6);
    g.add_edge(e7);
    g.add_edge(e8);
    g.add_edge(e9);

    g.add_label(e0,
                ParallelTensor(ParallelTensorDims({2, 1}),
                               DataType::FLOAT,
                               CreateGrad::YES));
    g.add_label(e1,
                ParallelTensor(ParallelTensorDims({2, 1}),
                               DataType::FLOAT,
                               CreateGrad::YES));
    g.add_label(e2,
                ParallelTensor(ParallelTensorDims({2, 1}),
                               DataType::FLOAT,
                               CreateGrad::YES));
    g.add_label(e3,
                ParallelTensor(ParallelTensorDims({2, 1}),
                               DataType::FLOAT,
                               CreateGrad::YES));
    g.add_label(e4,
                ParallelTensor(ParallelTensorDims({2, 1}),
                               DataType::FLOAT,
                               CreateGrad::YES));
    g.add_label(e5,
                ParallelTensor(ParallelTensorDims({2, 1}),
                               DataType::FLOAT,
                               CreateGrad::YES));
    g.add_label(e6,
                ParallelTensor(ParallelTensorDims({2, 1}),
                               DataType::FLOAT,
                               CreateGrad::YES));
    g.add_label(e7,
                ParallelTensor(ParallelTensorDims({2, 1}),
                               DataType::FLOAT,
                               CreateGrad::YES));
    g.add_label(e8,
                ParallelTensor(ParallelTensorDims({2, 1}),
                               DataType::FLOAT,
                               CreateGrad::YES));
    g.add_label(e9,
                ParallelTensor(ParallelTensorDims({2, 1}),
                               DataType::FLOAT,
                               CreateGrad::YES));

    CostEstimator estimator = CostEstimator::create<TestCostEstimator>();
    std::unordered_map<Node, MachineView> devices = {
        {n0, make_1d_machine_view(gpu_id_t(1), gpu_id_t(2))},
        {n1, make_1d_machine_view(gpu_id_t(1), gpu_id_t(2))},
        {n2, make_1d_machine_view(gpu_id_t(2), gpu_id_t(3))},
        {n3, make_1d_machine_view(gpu_id_t(3), gpu_id_t(4))},
        {n4, make_1d_machine_view(gpu_id_t(1), gpu_id_t(3))},
        {n5, make_1d_machine_view(gpu_id_t(2), gpu_id_t(4))},
        {n6, make_1d_machine_view(gpu_id_t(1), gpu_id_t(2))},        
      };

    MachineMapping device_mapping = {devices};
    auto frontier_machine_views =
        std::unordered_map<InputMultiDiEdge, MachineView> {
      {e0, make_1d_machine_view(gpu_id_t(1), gpu_id_t(2))},
    };

    float result = parallel_estimate_cost(
        g, estimator, device_mapping, frontier_machine_views);
    CHECK(std::abs(result - 0.5) < 1e-7);
  }
}