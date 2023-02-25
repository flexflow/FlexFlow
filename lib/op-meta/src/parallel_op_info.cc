#include "op-meta/parallel_op_info.h"
#include "op-meta/visit_struct.h"

namespace FlexFlow {
namespace opmeta {

bool operator==(ParallelOpInfo const &lhs, ParallelOpInfo const &rhs) {
  return visit_eq(lhs, rhs);
}

bool operator<(ParallelOpInfo const &lhs, ParallelOpInfo const &rhs) {
  return visit_lt(lhs, rhs);
}

}
}

namespace std {
using ::FlexFlow::opmeta::ParallelOpInfo;

size_t hash<ParallelOpInfo>::operator()(
    ParallelOpInfo const &params) const {
  return visit_hash(params);
}
}
