#include "op-meta/ops/embedding_params.h"

namespace FlexFlow {

typename EmbeddingParams::AsConstTuple EmbeddingParams::as_tuple() const {
  return {this->num_entries, this->out_channels, this->aggr, this->data_type};
}

bool operator==(EmbeddingParams const &lhs, EmbeddingParams const &rhs) {
  return lhs.as_tuple() == rhs.as_tuple();
}

bool operator<(EmbeddingParams const &lhs, EmbeddingParams const &rhs) {
  return lhs.as_tuple() < rhs.as_tuple();
}

bool EmbeddingParams::is_valid(ParallelTensorShape const &input) const {
  return input.is_valid();
}

}
