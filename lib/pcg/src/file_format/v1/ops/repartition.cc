#include "pcg/file_format/v1/ops/repartition.h"
#include "pcg/file_format/v1/ff_dim.h"

namespace FlexFlow {

V1RepartitionAttrs to_v1(RepartitionAttrs const &a) {
  return {to_v1(a.repartition_dim), a.repartition_degree};
}

} // namespace FlexFlow
