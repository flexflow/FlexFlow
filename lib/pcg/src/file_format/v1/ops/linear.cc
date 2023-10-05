#include "pcg/file_format/v1/ops/linear.h"
#include "pcg/file_format/v1/v1.h"

namespace FlexFlow {

V1L1RegularizerAttrs to_v1(L1RegularizerAttrs const &a) {
  return {a.lambda};
}

L1RegularizerAttrs from_v1(V1L1RegularizerAttrs const &va) {
  return {va.lambda};
}

V1L2RegularizerAttrs to_v1(L2RegularizerAttrs const &a) {
  return {a.lambda};
}

L2RegularizerAttrs from_v1(V1L2RegularizerAttrs const &va) {
  return {va.lambda};
}

V1RegularizerAttrs to_v1(RegularizerAttrs const &a) {
  // There should be a better way of doing this.
  if (auto const *l1 = get_if<L1RegularizerAttrs>(&a)) {
    return to_v1(*l1);
  } else if (auto const *l2 = get_if<L2RegularizerAttrs>(&a)) {
    return to_v1(*l2);
  } else {
    NOT_REACHABLE();
  }
}

RegularizerAttrs from_v1(V1RegularizerAttrs const &a) {
  // There should be a better way of doing this.
  if (auto const *l1 = get_if<V1L1RegularizerAttrs>(&a)) {
    return from_v1(*l1);
  } else if (auto const *l2 = get_if<V1L2RegularizerAttrs>(&a)) {
    return from_v1(*l2);
  } else {
    NOT_REACHABLE();
  }
}

V1LinearAttrs to_v1(LinearAttrs const &a) {
  return {a.out_channels,
          a.use_bias,
          to_v1(a.data_type),
          to_v1(a.activation),
          to_v1<V1RegularizerAttrs>(a.regularizer)};
}

LinearAttrs from_v1(V1LinearAttrs const &va) {
  return {va.out_channels,
          va.use_bias,
          from_v1(va.data_type),
          from_v1(va.activation),
          from_v1<RegularizerAttrs>(va.regularizer)};
}

} // namespace FlexFlow
