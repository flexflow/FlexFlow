#include "op-attrs/ops/embedding.h"
#include "op-attrs/ff_dim.h"
#include "op-attrs/parallel_dim.h"
#include "op-attrs/parallel_tensor_dims.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "op-attrs/tensor_shape.h"
#include "utils/exception.h"

namespace FlexFlow {

// pytorch nn.Embedding
// Embedding OP: (num_embeddings, embedding_dim) (num_entries, out_channels)
// input:(<ri, di1, t>, < b, di2, f>, < seq_len, di3, f>)
// EmbeddingAttrs:req<int> num_entries, out_channels;
// output:(<ro, do1, t>, <b, do2, f>, <seq_len, do3, f>, <embedding_dim, do4,
// f>)
ParallelTensorShape get_output_shape(EmbeddingAttrs const &attrs,
                                     ParallelTensorShape const &input) {
  if (input.num_dims() != 3) {
    throw mk_runtime_error("for embedding, input shape must be 3D");
  }

  std::vector<ParallelDim> data;
  data.resize(4);
  data[0] = input.at(ff_dim_t(0));
  data[0].is_replica_dim = true;
  data[1] = input.at(ff_dim_t(1));
  data[2] = input.at(ff_dim_t(2));
  data[3].size = attrs.out_channels; // TODO:what's the embedding_dim?
  data[3].is_replica_dim = false;

  ParallelTensorShape output = ParallelTensorShape(
      ParallelTensorDims(TensorDims(data.begin(), data.end())),
      attrs.data_type);

  return output;
}

} // namespace FlexFlow
