#include "compiler/cost_estimate.h"
#include "compiler/cost_estimator.h"
#include "doctest/doctest.h"
#include "test_cost_estimator.h"



using namespace FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  
  TEST_CASE("parallel_estimate_cost: linear graph") {
    // Straight line example, 3 nodes (with the last 3 being input to the cost estimator)
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
    // Non-linear graph example, 4 nodes
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

    CostEstimator estimator = CostEstimator::create<TestCostEstimator>();
    std::unordered_map<Node, MachineView> devices = {
        {n0, make_1d_machine_view(gpu_id_t(1), gpu_id_t(2))},
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
    CHECK(std::abs(result - 0.3) < 1e-7);
  }
}