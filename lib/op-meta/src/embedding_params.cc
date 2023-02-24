#include "op-meta/ops/embedding_params.h"
#include "op-meta/visit_struct.h"

namespace FlexFlow {
namespace opmeta {

bool operator==(EmbeddingParams const &lhs, EmbeddingParams const &rhs) {
  return visit_eq(lhs, rhs);
}

bool operator<(EmbeddingParams const &lhs, EmbeddingParams const &rhs) {
  return visit_lt(lhs, rhs);
}

}
}
