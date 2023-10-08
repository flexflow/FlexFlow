#include "op-attrs/ops/embedding.h"

namespace FlexFlow {

bool EmbeddingAttrs::is_valid(ParallelTensorShape const & input) const {
    if(!input.is_valid()) {
        return false;
    }
    return true;
}

//pytorch nn.Embedding
//Embedding OP: (num_embeddings, embedding_dim) (num_entries, out_channels)
//Input: (batch_size, seq_len)
//Output: (batch_size, seq_len, embedding_dim)
ParallelTensorShape get_output_shape(EmbeddingAttrs const & atts,
                                     ParallelTensorShape const & input) {
  ParallelTensorShape output = input;
  output.at(ff_dim_t(1)).size = input.at(ff_dim_t(1)).size;
  output.at(ff_dim_t(2)).size= atts.out_channels;
  return output;
} // namespace FlexFlow
