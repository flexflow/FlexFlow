#include "optimizations.h"
#include "op-attrs/get_op_type.h"
#include "utils/graph/rewriting.h"

namespace FlexFlow {

struct OptimizeUnnecessaryGradientCalculations {
  OptimizeUnnecessaryGradientCalculations() = default;

  Operator operator()(
      LabelledOpenMultiDiGraph<Operator, ParallelTensorAttrs> const &g,
      Node const &n,
      Operator const &op) {
    return op;
  }
  ParallelTensorAttrs operator()(
      LabelledOpenMultiDiGraph<Operator, ParallelTensorAttrs> const &g,
      MultiDiEdge const &e,
      ParallelTensorAttrs const &pt) {
    ParallelTensorAttrs result = pt;
    if (get_op_type(g.at(e.src).attrs) == OperatorType::INPUT) {
      result.create_gradients = CreateGrad::NO;
    }
    return result;
  }
};

ParallelComputationGraph optimize_unnecessary_gradient_calculations(
    ParallelComputationGraph const &pcg) {
  // If an operator's input is training data
  // No need to compute its gradients
  return pcg.on_underlying(
      [](LabelledOpenMultiDiGraph<Operator, ParallelTensor> const &g) {
        return rewrite(OptimizeUnnecessaryGradientCalculations{}, g);
      });
}

} // namespace FlexFlow
