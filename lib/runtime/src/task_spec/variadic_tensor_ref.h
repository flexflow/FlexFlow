#ifndef _FLEXFLOW_RUNTIME_SRC_TASK_SPEC_VARIADIC_TENSOR_ARG_REF_H
#define _FLEXFLOW_RUNTIME_SRC_TASK_SPEC_VARIADIC_TENSOR_ARG_REF_H

#include "arg_ref.h"
#include "op_tensor_spec.h"

namespace FlexFlow {

enum class VariadicTensorRefType { INPUT_TENSORS, NUM_INPUTS };

template <typename T>
using VariadicTensorRef = ArgRef<VariadicTensorRefType, T>;

VariadicTensorRef<OpTensorSpec> get_input_tensors() {
  return {VariadicTensorRefType::INPUT_TENSORS};
}

VariadicTensorRef<int> get_number_inputs() {
  return {VariadicTensorRefType::NUM_INPUTS};
}

} // namespace FlexFlow

#endif