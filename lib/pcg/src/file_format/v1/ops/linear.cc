#include "pcg/file_format/v1/ops/linear.h"
#include "pcg/file_format/v1/v1.h"

namespace FlexFlow {

V1L1RegularizerAttrs to_v1(L1RegularizerAttrs const &a) {
  return {to_v1(a.lambda)};
}

V1L2RegularizerAttrs to_v1(L2RegularizerAttrs const &a) {
  return {to_v1(a.lambda)};
}

V1RegularizerAttrs to_v1(RegularizerAttrs const &a) {
  NOT_IMPLEMENTED();
}

V1LinearAttrs to_v1(LinearAttrs const &a) {
  return {to_v1(a.out_channels),
          to_v1(a.use_bias),
          to_v1(a.data_type),
          to_v1(a.activation),
          to_v1<V1RegularizerAttrs>(a.regularizer)};
}

} // namespace FlexFlow
