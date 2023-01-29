#ifndef _FLEXFLOW_PARALLEL_OPS_PARALLEL_OP_INFO_H
#define _FLEXFLOW_PARALLEL_OPS_PARALLEL_OP_INFO_H

#include "op-meta/ffconst.h"
#include <tuple>
#include <functional>

namespace FlexFlow {

struct ParallelOpInfo {
public:
  using AsConstTuple = std::tuple<OperatorType, int, int>;
  AsConstTuple as_tuple() const;
public:
  OperatorType op_type;
  int parallel_dim;
  int parallel_degree;
};

bool operator==(ParallelOpInfo const &, ParallelOpInfo const &);
bool operator<(ParallelOpInfo const &, ParallelOpInfo const &);
void swap(ParallelOpInfo &, ParallelOpInfo &);

}

namespace std {
template <>
struct hash<FlexFlow::ParallelOpInfo> {
  size_t operator()(FlexFlow::ParallelOpInfo const &) const;
};
}

#endif /* _FLEXFLOW_PARALLEL_OPS_PARALLEL_OP_INFO_H */
