#include "pcg/file_format/v1/parallel_tensor_dims.h"

namespace FlexFlow {

V1ParallelTensorDims to_v1(ParallelTensorDims const &dims) {
  std::vector<V1ParallelDim> data;
  for (ParallelDim const &pdim : dims.data) {
    data.emplace_back(to_v1(pdim));
  }
  return {data};
}

ParallelTensorDims from_v1(V1ParallelTensorDims const &vdims) {
  std::vector<ParallelDim> dims;
  for (V1ParallelDim const &pdim : vdims.data) {
    dims.emplace_back(from_v1(pdim));
  }
  return ParallelTensorDims(dims);
}

} // namespace FlexFlow
