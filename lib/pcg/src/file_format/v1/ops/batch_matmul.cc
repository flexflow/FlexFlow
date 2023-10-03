#include "pcg/file_format/v1/ops/batch_matmul.h"
#include "pcg/file_format/v1/v1.h"

namespace FlexFlow {

V1BatchMatmulAttrs to_v1(BatchMatmulAttrs const &a) {
  return {to_v1(a.a_seq_length_dim), to_v1(a.b_seq_length_dim)};
}

} // namespace FlexFlow
