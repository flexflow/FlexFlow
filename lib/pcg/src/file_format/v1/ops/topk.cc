#include "pcg/file_format/v1/ops/topk.h"

namespace FlexFlow {

V1TopKAttrs to_v1(TopKAttrs const &a) {
  return {a.k, a.sorted};
}

TopKAttrs from_v1(V1TopKAttrs const &va) {
  return {va.k, va.sorted};
}

} // namespace FlexFlow
