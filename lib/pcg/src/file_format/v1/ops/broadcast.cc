#include "pcg/file_format/v1/ops/broadcast.h"
#include "pcg/file_format/v1/v1.h"

namespace FlexFlow {

V1BroadcastAttrs to_v1(BroadcastAttrs const &a) {
  return {std::vector<int>(a.target_dims.begin(), a.target_dims.end())};
}

} // namespace FlexFlow
