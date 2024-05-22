#ifndef _FLEXFLOW_LOCAL_EXECUTION_VARIADIC_TENSOR_ARG_REF_H
#define _FLEXFLOW_LOCAL_EXECUTION_VARIADIC_TENSOR_ARG_REF_H

#include "arg_ref.h"
#include "op_tensor_spec.h"

namespace FlexFlow {

enum class VariadicTensorRefType { INPUT_TENSORS };

template <typename T>
using VariadicTensorRef = ArgRef<VariadicTensorRefType, T>;

VariadicTensorRef<OpTensorSpec> get_input_tensors();

} // namespace FlexFlow

#endif
