#ifndef _FLEXFLOW_EMBEDDING_PARAMS_H
#define _FLEXFLOW_EMBEDDING_PARAMS_H

#include "op-meta/ffconst.h"
#include "op-meta/parallel_tensor_shape.h"

namespace FlexFlow {

struct EmbeddingParams {
  int num_entries, out_channels;
  AggrMode aggr;
  DataType data_type;

  bool is_valid(ParallelTensorShape const &) const;

  using AsConstTuple = std::tuple<int, int, AggrMode, DataType>;
  AsConstTuple as_tuple() const;
};

bool operator==(EmbeddingParams const &, EmbeddingParams const &);
bool operator<(EmbeddingParams const &, EmbeddingParams const &);

} // namespace FlexFlow

namespace std {
template <>
struct hash<FlexFlow::EmbeddingParams> {
  size_t operator()(FlexFlow::EmbeddingParams const &) const;
};
} // namespace std

#endif // _FLEXFLOW_EMBEDDING_PARAMS_H
