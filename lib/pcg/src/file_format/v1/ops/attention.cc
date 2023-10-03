#include "pcg/file_format/v1/ops/attention.h"
#include "pcg/file_format/v1/v1.h"

namespace FlexFlow {

V1MultiHeadAttentionAttrs to_v1(MultiHeadAttentionAttrs const &a) {
  return {to_v1(a.embed_dim),
          to_v1(a.num_heads),
          to_v1(a.kdim),
          to_v1(a.vdim),
          to_v1(a.dropout),
          to_v1(a.bias),
          to_v1(a.add_bias_kv),
          to_v1(a.add_zero_attn)};
}

} // namespace FlexFlow
