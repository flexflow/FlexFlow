#include "pcg/file_format/v1/ops/batch_matmul.h"
#include "pcg/file_format/v1/v1.h"

namespace FlexFlow {

V1BatchMatmulAttrs to_v1(BatchMatmulAttrs const &a) {
  return {a.a_seq_length_dim, a.b_seq_length_dim};
}

BatchMatmulAttrs from_v1(V1BatchMatmulAttrs const &va) {
  return {va.a_seq_length_dim, va.b_seq_length_dim};
}

} // namespace FlexFlow
