#include "pcg/file_format/v1/ops/broadcast.h"
#include "pcg/file_format/v1/v1.h"

namespace FlexFlow {

V1BroadcastAttrs to_v1(BroadcastAttrs const &a) {
  return {std::vector<int>(a.target_dims.begin(), a.target_dims.end())};
}

BroadcastAttrs from_v1(V1BroadcastAttrs const &va) {
  stack_vector<int, MAX_TENSOR_DIM> dims;
  for (const int& dim : va.target_dims)
    dims.emplace_back(dim);
  return BroadcastAttrs{dims};
}

} // namespace FlexFlow
