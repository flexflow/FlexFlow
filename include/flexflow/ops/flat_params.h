#ifndef _FLEXFLOW_FLAT_PARAMS_H
#define _FLEXFLOW_FLAT_PARAMS_H

#include "flexflow/ffconst.h"
#include "flexflow/parallel_tensor.h"

namespace FlexFlow {

struct FlatParams {
  char name[MAX_OPNAME];
  bool is_valid(ParallelTensorShape const &) const;
  void solve_dims(ParallelTensorShape const &input,
                  ParallelDim output_dims[MAX_TENSOR_DIM],
                  int *output_ndims) const;

private:
  int output_size(ParallelTensorShape const &input,
                  ParallelDim output_dims[MAX_TENSOR_DIM]) const;
};

bool operator==(FlatParams const &, FlatParams const &);

} // namespace FlexFlow

namespace std {
template <>
struct hash<FlexFlow::FlatParams> {
  size_t operator()(FlexFlow::FlatParams const &) const;
};
} // namespace std

#endif // _FLEXFLOW_FLAT_PARAMS_H
