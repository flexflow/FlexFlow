#include "compiler/cost_estimate.h"
#include "compiler/cost_estimator.h"
#include "doctest/doctest.h"
#include "test_cost_estimator.h"

using namespace FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {

  TEST_CASE("parallel_estimate_cost: linear graph") {

    // Straight line example, 3 nodes: ->(n1)->(n2)->(n3)
    auto g = OutputLabelledOpenMultiDiGraph<Operator, ParallelTensor>::
        template create<
            UnorderedOutputLabelledOpenMultiDiGraph<Operator,
                                                    ParallelTensor>>();

    Node n1 = g.add_node(Operator{InputAttrs{}, "n1"});
    Node n2 = g.add_node(Operator{InputAttrs{}, "n2"});
    Node n3 = g.add_node(Operator{InputAttrs{}, "n3"});

    NodePort p1 = g.add_node_port();
    NodePort p2 = g.add_node_port();
    NodePort p3 = g.add_node_port();

    // dst, dstport, uid
    InputMultiDiEdge e0{n1, p1, {1, 1}};
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

    CostEstimator estimator =
        CostEstimator::create<TestCostEstimator>(); // Returns 0.1 regardless
    std::unordered_map<Node, MachineView> devices = {
        // single device per node
        {n1, make_1d_machine_view(gpu_id_t(1), gpu_id_t(2))},
        {n2, make_1d_machine_view(gpu_id_t(1), gpu_id_t(2))},
        {n3, make_1d_machine_view(gpu_id_t(1), gpu_id_t(2))}};

    MachineMapping device_mapping = {devices};
    auto frontier_machine_views =
        std::unordered_map<InputMultiDiEdge, MachineView>{
            {e0, make_1d_machine_view(gpu_id_t(1), gpu_id_t(2))},
        };

    float result = parallel_estimate_cost(
        g, estimator, device_mapping, frontier_machine_views);
    CHECK(std::abs(result - .3) < 1e-7);
  }

  TEST_CASE("parallel_estimate_cost: non-linear graph") {
    // Non-linear graph example, diamond pattern, 4 nodes

    auto g = OutputLabelledOpenMultiDiGraph<Operator, ParallelTensor>::
        template create<
            UnorderedOutputLabelledOpenMultiDiGraph<Operator,
                                                    ParallelTensor>>();

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
        {n1,
         make_1d_machine_view(gpu_id_t(1),
                              gpu_id_t(3))}, // nodes n1, n2 can run in parallel
        {n2, make_1d_machine_view(gpu_id_t(5), gpu_id_t(6))},
        {n3, make_1d_machine_view(gpu_id_t(1), gpu_id_t(3))}};

    MachineMapping device_mapping = {devices};
    auto frontier_machine_views =
        std::unordered_map<InputMultiDiEdge, MachineView>{
            {e0, make_1d_machine_view(gpu_id_t(1), gpu_id_t(3))},
        };

    float result = parallel_estimate_cost(
        g, estimator, device_mapping, frontier_machine_views);
    CHECK(std::abs(result - 0.3) < 1e-7);
  }

  TEST_CASE("parallel_estimate_cost: more complex non-linear graph") {
    /* Non-linear graph example, 7 nodes
    graph TD
    .(( )) --> |"e0"|n0
    n0 --> |"e1"| n1["n1"]
    n0 --> |"e2"| n2["n2"]
    n0 --> |"e3"| n3["n3"]
    n1 --> |"e4"| n4["n4"]
    n2 --> |"e5"| n4["n4"]
    n2 --> |"e6"| n5["n5"]
    n3 --> |"e7"| n5["n5"]
    n4 --> |"e8"| n6["n6"]
    n5 --> |"e9"| n6["n6"]
    */
    auto g = OutputLabelledOpenMultiDiGraph<Operator, ParallelTensor>::
        template create<
            UnorderedOutputLabelledOpenMultiDiGraph<Operator,
                                                    ParallelTensor>>();

    std::vector<Node> n;
    for (int i = 0; i < 7; ++i) {
      n.push_back(g.add_node(Operator{InputAttrs{}, "n" + std::to_string(i)}));
    }

    std::vector<NodePort> p;
    for (int i = 0; i < 7; ++i) {
      p.push_back(g.add_node_port());
    }
    // dst, dstport, uid
    InputMultiDiEdge e0{n[0], p[0], {1, 1}};
    g.add_edge(e0);
    g.add_label(e0,
                ParallelTensor(ParallelTensorDims({2, 1}),
                               DataType::FLOAT,
                               CreateGrad::YES));

    // MultiDiEdge: dst, dstport, src, srcport
    MultiDiEdge e1{n[1], p[1], n[0], p[0]};
    MultiDiEdge e2{n[2], p[2], n[0], p[0]};
    MultiDiEdge e3{n[3], p[3], n[0], p[0]};

    MultiDiEdge e4{n[4], p[4], n[1], p[1]};
    MultiDiEdge e5{n[4], p[4], n[2], p[2]};
    MultiDiEdge e6{n[5], p[5], n[2], p[2]};
    MultiDiEdge e7{n[5], p[5], n[3], p[3]};

    MultiDiEdge e8{n[6], p[6], n[4], p[4]};
    MultiDiEdge e9{n[6], p[6], n[5], p[5]};
    std::vector<MultiDiEdge> edges = {e1, e2, e3, e4, e5, e6, e7, e8, e9};

    for (auto &edge : edges) {
      g.add_edge(edge);
      g.add_label(edge,
                  ParallelTensor(ParallelTensorDims({2, 1}),
                                 DataType::FLOAT,
                                 CreateGrad::YES));
    }

    // All machines contain a
    CostEstimator estimator = CostEstimator::create<TestCostEstimator>();
    std::unordered_map<Node, MachineView> devices = {
        {n[0], make_1d_machine_view(gpu_id_t(1), gpu_id_t(2))},

        {n[1], make_1d_machine_view(gpu_id_t(1), gpu_id_t(2))},
        {n[2], make_1d_machine_view(gpu_id_t(2), gpu_id_t(3))},
        {n[3], make_1d_machine_view(gpu_id_t(3), gpu_id_t(4))},

        {n[4],
         make_1d_machine_view(
             gpu_id_t(1), gpu_id_t(3))}, // Note that GPUs overlap in this case,
                                         // so this layer cannot be parallelized
        {n[5], make_1d_machine_view(gpu_id_t(2), gpu_id_t(4))},

        {n[6], make_1d_machine_view(gpu_id_t(1), gpu_id_t(20))},
    };

    MachineMapping device_mapping = {devices};
    auto frontier_machine_views =
        std::unordered_map<InputMultiDiEdge, MachineView>{
            {e0, make_1d_machine_view(gpu_id_t(1), gpu_id_t(2))},
        };

    float result = parallel_estimate_cost(
        g, estimator, device_mapping, frontier_machine_views);
    CHECK(std::abs(result - 0.5) < 1e-7);
  }
}
