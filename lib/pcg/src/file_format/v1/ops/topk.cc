#include "pcg/file_format/v1/ops/topk.h"
#include "pcg/file_format/v1/v1.h"

namespace FlexFlow {

V1TopKAttrs to_v1(TopKAttrs const &a) {
  return {to_v1(a.k), to_v1(a.sorted)};
}

} // namespace FlexFlow
