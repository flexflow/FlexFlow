#include "op-attrs/ops/embedding.h"
#include "utils/visitable_funcs.h"

namespace FlexFlow {

bool operator==(EmbeddingAttrs const &lhs, EmbeddingAttrs const &rhs) {
  return visit_eq(lhs, rhs);
}

bool operator<(EmbeddingAttrs const &lhs, EmbeddingAttrs const &rhs) {
  return visit_lt(lhs, rhs);
}

}

namespace std {

using ::FlexFlow::EmbeddingAttrs;

size_t hash<EmbeddingAttrs>::operator()(EmbeddingAttrs const &attrs) const {
  return visit_hash(attrs);
}

}

