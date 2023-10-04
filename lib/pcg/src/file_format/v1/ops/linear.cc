#include "pcg/file_format/v1/ops/linear.h"
#include "pcg/file_format/v1/v1.h"

namespace FlexFlow {

V1L1RegularizerAttrs to_v1(L1RegularizerAttrs const &a) {
  return {a.lambda};
}

V1L2RegularizerAttrs to_v1(L2RegularizerAttrs const &a) {
  return {a.lambda};
}

V1RegularizerAttrs to_v1(RegularizerAttrs const &a) {
  // There should be a better way of doing this.
  if (const auto* l1 = get_if<L1RegularizerAttrs>(&a))
    return to_v1(*l1);
  else if (const auto* l2 = get_if<L2RegularizerAttrs>(&a))
    return to_v1(*l2);
  else
    NOT_REACHABLE();
}

V1LinearAttrs to_v1(LinearAttrs const &a) {
  return {a.out_channels,
          a.use_bias,
          to_v1(a.data_type),
          to_v1(a.activation),
          to_v1<V1RegularizerAttrs>(a.regularizer)};
}

} // namespace FlexFlow
