#include "pcg/file_format/v1/ops/repartition.h"
#include "pcg/file_format/v1/ff_dim.h"
#include "pcg/file_format/v1/v1.h"

namespace FlexFlow {

V1RepartitionAttrs to_v1(RepartitionAttrs const &a) {
  return {to_v1(a.repartition_dim), to_v1(a.repartition_degree)};
}

} // namespace FlexFlow
