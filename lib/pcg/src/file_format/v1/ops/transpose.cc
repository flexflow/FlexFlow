#include "pcg/file_format/v1/ops/transpose.h"
#include "pcg/file_format/v1/ff_dim.h"
#include "pcg/file_format/v1/v1.h"

namespace FlexFlow {

V1TransposeAttrs to_v1(TransposeAttrs const &a) {
  return {std::vector<int>(a.perm.begin(), a.perm.end())};
}

TransposeAttrs from_v1(V1TransposeAttrs const &va) {
  stack_vector<ff_dim_t, MAX_TENSOR_DIM> perm;
  for (int const &i : va.perm) {
    perm.push_back(ff_dim_t(i));
  }
  return {perm};
}

} // namespace FlexFlow
