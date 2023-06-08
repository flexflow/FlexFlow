#include "fused_parallel_op_attrs.h"

namespace FlexFlow {

FusedParallelOpAttrs::FusedParallelOpAttrs(
    stack_vector<ParallelOpInfo, MAX_NUM_FUSED_OPERATORS> const &_parallel_ops)
    : parallel_ops(_parallel_ops) {}

} // namespace FlexFlow
