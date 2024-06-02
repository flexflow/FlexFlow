#include "local-execution/op_arg_ref.h"

namespace FlexFlow {

OpArgRef<ParallelTensorShape> input_parallel_tensor_shape(int idx) {
  return {OpArgRefType::PARALLEL_TENSOR_SHAPE};
}

} // namespace FlexFlow
