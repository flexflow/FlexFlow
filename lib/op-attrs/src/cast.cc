#include "op-attrs/ops/cast.h"
#include "utils/visitable_funcs.h"

namespace FlexFlow {

bool operator==(CastAttrs const &lhs, CastAttrs const &rhs) {
  return visit_eq(lhs, rhs);
}

bool operator<(CastAttrs const &lhs, CastAttrs const &rhs) {
  return visit_lt(lhs, rhs);
}

/* bool CastAttrs::is_valid(ParallelTensorShape const &input) const { */
/*   bool valid = input.is_valid(); */
/*   valid &= (input.at(input.num_dims() - 1).degree == 1); */
/*   return valid; */
/* } */

}

namespace std {
using ::FlexFlow::CastAttrs;

size_t hash<CastAttrs>::operator()(
    CastAttrs const &params) const {
  return visit_hash(params);
} 
}

