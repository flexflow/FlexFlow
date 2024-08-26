#include "local-execution/op_arg_ref.h"

namespace FlexFlow {

OpArgRef<ParallelTensorShape> input_parallel_tensor_shape(int idx) {
  OpArgRefType arg_ref_type = OpArgRefType{ParallelTensorShapeRefType{idx}};
  ArgRef<OpArgRefType, ParallelTensorShape> arg_ref = {arg_ref_type};
  return arg_ref;
}

} // namespace FlexFlow
