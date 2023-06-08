#include "parallel_op_info.h"

namespace FlexFlow {

ParallelOpInfo::ParallelOpInfo(OperatorType _op_type,
                               ff_dim_t _parallel_dim,
                               int _parallel_degree)
    : op_type(_op_type), parallel_dim(_parallel_dim),
      parallel_degree(_parallel_degree) {}

} // namespace FlexFlow
