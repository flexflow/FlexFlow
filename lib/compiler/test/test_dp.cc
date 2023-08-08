#include "compiler/unity_algorithm.h"
#include "doctest.h"

using namespace FlexFlow;

struct TestCostEstimator : public ICostEstimator {
  float estimate_cost(PCGOperatorAttrs const &op,
                      std::vector<ParallelTensorShape> const &inputs,
                      MachineView const &mv) const override {
    return 0.1;
  }
  float estimate_cost(ParallelTensorShape const &tensor_shape,
                      MachineView const &src,
                      MachineView const &dst) const override {
    return 1;
  }
};

TEST_CASE("optimal_cost") {
  auto g(NodeLabelledMultiDiGraph<PCGOperatorAttrs>::create<
         UnorderedNodeLabelledMultiDiGraph<PCGOperatorAttrs>>());

  Node n0 = g.add_node(InputAttrs());
  Node n1 = g.add_node(RepartitionAttrs(ff_dim_t(0), 2));
  Node n2 = g.add_node(ElementScalarUnaryAttrs(OP_SCALAR_ADD, 0));
  Node n3 = g.add_node(ElementScalarUnaryAttrs(OP_SCALAR_ADD, 1));
  Node n4 = g.add_node(ConcatAttrs(ff_dim_t(1)));
  Node n5 = g.add_node(CombineAttrs(ff_dim_t(0), 2));

  MultiDiEdge e0(n0, n1, 0, 0);
  MultiDiEdge e1(n1, n2, 0, 0);
  MultiDiEdge e2(n1, n3, 1, 0);
  MultiDiEdge e3(n2, n4, 0, 0);
  MultiDiEdge e4(n3, n4, 0, 1);
  MultiDiEdge e5(n4, n5, 0, 0);

  g.add_edge(e0);
  g.add_edge(e1);
  g.add_edge(e2);
  g.add_edge(e3);
  g.add_edge(e4);

  OptimizerPCG pcg = infer_tensor_shape(g);
  auto allowed_machine_views = [](PCGOperatorAttrs const &,
                                  MachineResource const &) {
    // TODO
    return std::unordered_set<MachineView>{};
  };
  MachineResource resource(1, 1, 2);
  Strategy s =
      optimal_cost(pcg, allowed_machine_views, TestCostEstimator{}, resource);

  // TODO: check result
}
