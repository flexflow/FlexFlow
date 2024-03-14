#ifndef _FLEXFLOW_SAMPLING_PARAMS_H
#define _FLEXFLOW_SAMPLING_PARAMS_H

#include "flexflow/ffconst.h"
#include "flexflow/parallel_tensor.h"

namespace FlexFlow {

struct SamplingParams {
  float top_p;
  char name[MAX_OPNAME];
  bool is_valid(ParallelTensorShape const &) const;
};
bool operator==(SamplingParams const &, SamplingParams const &);

} // namespace FlexFlow

namespace std {
template <>
struct hash<FlexFlow::SamplingParams> {
  size_t operator()(FlexFlow::SamplingParams const &) const;
};
} // namespace std

#endif // _FLEXFLOW_SAMPLING_PARAMS_H