#include "compiler/cost_estimator.h"
#include "doctest/doctest.h"
#include "doctest/test_cost_estimator.h"
#include "compiler/cost_estimate.h"
#include "rapidcheck.h"
namespace FlexFlow {

struct TestCostEstimator : public ICostEstimator {
  float estimate_cost(PCGOperatorAttrs const &op,
                      std::vector<ParallelTensorShape> const &inputs,
                      MachineView const &mv) const override {
    return 0.1;
  }
};

using namespace FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("parallel_estimate_cost: linear graph") {
    //Straight line example, 3 nodes
    SubParallelComputationGraphView g =
        OutputLabelledOpenMultiDiGraphView<Operator, ParallelTensor>::create<
            UnorderedOutputLabelledOpenMultiDiGraph<Operator, ParallelTensor>>();
    Node n0 = g.add_node(Operator{InputAttrs{}, "n0"});
    Node n1 = g.add_node(Operator{InputAttrs{}, "n1"});
    Node n2 = g.add_node(Operator{InputAttrs{}, "n2"});

    NodePort p0 = g.add_node_port();
    NodePort p1 = g.add_node_port();
    NodePort p2 = g.add_node_port();
    
    // MultiDiEdge: dst, dstport, src, srcport
    MultiDiEdge e0{n1, p1, n0, p0};
    MultiDiEdge e1{n2, p2, n1, p1};

    g.add_edge(e0);
    g.add_edge(e1);
    g.add_label(e0, 10);
    g.add_label(e1, 11);
    
    CostEstimator estimator = CostEstimator::create<TestCostEstimator>();
    MachineMapping device_mapping = std::unordered_map<Node, MachineView>{
          {n0, make_1d_machine_view(gpu_id_t(1), gpu_id_t(2))},
          {n0, make_1d_machine_view(gpu_id_t(1), gpu_id_t(2))},
          {n0, make_1d_machine_view(gpu_id_t(1), gpu_id_t(2))}
    };
    auto frontier_machine_views = std::unordered_map<OpenMultiDiEdge, MachineView> {
      {}
    } 
    float result = parallel_estimate_cost(g, estimator, device_mapping, frontier_machine_views);
    CHECK(result==0.3);
  }

  TEST_CASE("parallel_estimate_cost: diamond graph") {
    //Straight line example, 3 nodes
    SubParallelComputationGraphView g =
        OutputLabelledOpenMultiDiGraphView<Operator, ParallelTensor>::create<
            UnorderedOutputLabelledOpenMultiDiGraph<Operator, ParallelTensor>>();
    Node n0 = g.add_node(Operator{InputAttrs{}, "n0"});
    Node n1 = g.add_node(Operator{InputAttrs{}, "n1"});
    Node n2 = g.add_node(Operator{InputAttrs{}, "n2"});

    NodePort p0 = g.add_node_port();
    NodePort p1 = g.add_node_port();
    NodePort p2 = g.add_node_port();
    
    // MultiDiEdge: dst, dstport, src, srcport
    MultiDiEdge e0{n1, p1, n0, p0};
    MultiDiEdge e1{n2, p2, n1, p1};

    g.add_edge(e0);
    g.add_edge(e1);
    
    g.add_label(e1, ParallelTensor(ParallelTensorDims({2, 1}),
                                  DataType::FLOAT,
                                  CreateGrad::YES)););
    g.add_input(e0,ParallelTensor(ParallelTensorDims({2, 1}),
                                  DataType::FLOAT,
                                  CreateGrad::YES));

    CostEstimator estimator = CostEstimator::create<TestCostEstimator>();
    MachineMapping device_mapping = std::unordered_map<Node,MachineView>{
          {n0, make_1d_machine_view(gpu_id_t(1), gpu_id_t(2))},
          {n1, make_1d_machine_view(gpu_id_t(1), gpu_id_t(2))},
          {n2, make_1d_machine_view(gpu_id_t(1), gpu_id_t(2))}
    };
    auto frontier_machine_views = std::unordered_map<OpenMultiDiEdge, MachineView> {
      {e0, make_1d_machine_view(gpu_id_t(1), gpu_id_t(2))},
    }
    float result = parallel_estimate_cost(g, estimator, device_mapping, frontier_machine_views);
    CHECK(result==0.3);
  }
}