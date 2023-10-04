#include "pcg/file_format/v1/ops/topk.h"

namespace FlexFlow {

V1TopKAttrs to_v1(TopKAttrs const &a) {
  return {a.k, a.sorted};
}

} // namespace FlexFlow
