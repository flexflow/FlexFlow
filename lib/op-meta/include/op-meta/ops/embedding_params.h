#ifndef _FLEXFLOW_EMBEDDING_PARAMS_H
#define _FLEXFLOW_EMBEDDING_PARAMS_H

#include "op-meta/ffconst.h"
#include "op-meta/parallel_tensor_shape.h"
#include "op-meta/ops/unary_op.h"
#include "visit_struct/visit_struct.hpp"

namespace FlexFlow {
namespace opmeta {

struct EmbeddingParams : public UnaryOpParams {
public:
  ParallelTensorShape output_shape(ParallelTensorShape const &input_shape) const override;
  OperatorType op_type() const override;
public:
  int num_entries, out_channels;
  AggrMode aggr;
  DataType data_type;
};

bool operator==(EmbeddingParams const &, EmbeddingParams const &);
bool operator<(EmbeddingParams const &, EmbeddingParams const &);

}
}

VISITABLE_STRUCT(::FlexFlow::opmeta::EmbeddingParams, num_entries, out_channels, aggr, data_type);

namespace std {
template <>
struct hash<::FlexFlow::opmeta::EmbeddingParams> {
  size_t operator()(::FlexFlow::opmeta::EmbeddingParams const &) const;
};
} 

#endif // _FLEXFLOW_EMBEDDING_PARAMS_H
