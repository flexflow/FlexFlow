#ifndef _FLEXFLOW_EMBEDDING_PARAMS_H
#define _FLEXFLOW_EMBEDDING_PARAMS_H

#include "op-meta/ffconst.h"
#include "op-meta/parallel_tensor_shape.h"
#include "op-meta/ops/unary_op.h"

namespace FlexFlow {

struct EmbeddingParams : public UnaryOpParams {
public:
  using AsConstTuple = std::tuple<int, int, AggrMode, DataType>;
  AsConstTuple as_tuple() const;

  bool is_valid(ParallelTensorShape const &) const;
  OperatorType op_type() const;
public:
  int num_entries, out_channels;
  AggrMode aggr;
  DataType data_type;
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
