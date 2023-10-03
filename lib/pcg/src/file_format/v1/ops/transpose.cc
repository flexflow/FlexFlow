#include "pcg/file_format/v1/ops/transpose.h"
#include "pcg/file_format/v1/ff_dim.h"
#include "pcg/file_format/v1/v1.h"

namespace FlexFlow {

V1TransposeAttrs to_v1(TransposeAttrs const &a) {
  return {std::vector<int>(a.perm.begin(), a.perm.end())};
}

} // namespace FlexFlow
