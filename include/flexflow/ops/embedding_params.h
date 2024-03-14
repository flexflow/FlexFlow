#ifndef _FLEXFLOW_EMBEDDING_PARAMS_H
#define _FLEXFLOW_EMBEDDING_PARAMS_H

#include "flexflow/ffconst.h"
#include "flexflow/fftype.h"
#include "flexflow/parallel_tensor.h"

namespace FlexFlow {

struct EmbeddingParams {
  int num_entries, out_channels;
  LayerID layer_guid;
  AggrMode aggr;
  DataType data_type;
  char name[MAX_OPNAME];

  bool is_valid(ParallelTensorShape const &) const;
};
bool operator==(EmbeddingParams const &, EmbeddingParams const &);

} // namespace FlexFlow

namespace std {
template <>
struct hash<FlexFlow::EmbeddingParams> {
  size_t operator()(FlexFlow::EmbeddingParams const &) const;
};
} // namespace std

#endif // _FLEXFLOW_EMBEDDING_PARAMS_H
