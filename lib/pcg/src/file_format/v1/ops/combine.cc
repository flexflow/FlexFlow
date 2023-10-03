#include "pcg/file_format/v1/ops/combine.h"
#include "pcg/file_format/v1/ff_dim.h"
#include "pcg/file_format/v1/v1.h"

namespace FlexFlow {

V1CombineAttrs to_v1(CombineAttrs const &a) {
  return {to_v1(a.combine_dim), to_v1(a.combine_degree)};
}

} // namespace FlexFlow
