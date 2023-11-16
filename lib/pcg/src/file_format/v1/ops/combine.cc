#include "pcg/file_format/v1/ops/combine.h"
#include "pcg/file_format/v1/ff_dim.h"

namespace FlexFlow {

V1CombineAttrs to_v1(CombineAttrs const &a) {
  return {to_v1(a.combine_dim), a.combine_degree};
}

CombineAttrs from_v1(V1CombineAttrs const &va) {
  return {from_v1(va.combine_dim), va.combine_degree};
}

} // namespace FlexFlow
