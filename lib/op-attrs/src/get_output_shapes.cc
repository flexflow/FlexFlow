#include "op-attrs/get_output_shapes.h"

namespace FlexFlow {

ParallelTensorShape as_parallel(TensorShape const &) {
  NOT_IMPLEMENTED();
}

std::vector<ParallelTensorShape> as_parallel(std::vector<TensorShape> const &) {
  NOT_IMPLEMENTED();
}

TensorShape get_output_shape(AggregateAttrs const &attrs,
                             TensorShape const &gate_preds,
                             TensorShape const &gate_assign,
                             TensorShape const &true_gate_assign,
                             TensorShape const &full_gate_gradients,
                             std::vector<TensorShape> const &exp_preds) {
  return get_tensor_shape_unsafe(
      get_output_shape(attrs,
                       as_parallel(gate_preds),
                       as_parallel(gate_assign),
                       as_parallel(true_gate_assign),
                       as_parallel(full_gate_gradients),
                       as_parallel(exp_preds)));
}

ParallelTensorShape get_output_shape(ElementBinaryAttrs const &,
                                     ParallelTensorShape const &,
                                     ParallelTensorShape const &) {
  NOT_IMPLEMENTED();
}

// FIXME: These are added to get rid of the linker errors about missing
// definitions.
template <>
TensorShape get_output_shape(BatchNormAttrs const &, TensorShape const &) {
  NOT_IMPLEMENTED();
}

template <>
TensorShape get_output_shape(Conv2DAttrs const &, TensorShape const &) {
  NOT_IMPLEMENTED();
}

template <>
TensorShape get_output_shape(DropoutAttrs const &, TensorShape const &) {
  NOT_IMPLEMENTED();
}

template <>
TensorShape get_output_shape(ElementBinaryAttrs const &,
                             TensorShape const &,
                             TensorShape const &) {
  NOT_IMPLEMENTED();
}

template <>
TensorShape get_output_shape(EmbeddingAttrs const &, TensorShape const &) {
  NOT_IMPLEMENTED();
}

template <>
TensorShape FlexFlow::get_output_shape(
    variant<FlexFlow::ElementUnaryAttrs,
            FlexFlow::ElementScalarUnaryAttrs> const &,
    TensorShape const &) {
  NOT_IMPLEMENTED();
}

template <>
std::vector<TensorShape> get_output_shapes(ElementBinaryAttrs const &attrs,
                                           TensorShape const &,
                                           TensorShape const &) {
  NOT_IMPLEMENTED();
}

template <>
std::vector<TensorShape> get_output_shapes(GatherAttrs const &attrs,
                                           TensorShape const &,
                                           TensorShape const &) {
  NOT_IMPLEMENTED();
}

} // namespace FlexFlow
