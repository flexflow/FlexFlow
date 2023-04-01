#include "op-attrs/ops/reduction.h"
#include "utils/visitable_funcs.h"

namespace FlexFlow {

bool operator==(ReductionAttrs const &lhs, ReductionAttrs const &rhs) {
  return visit_eq(lhs, rhs);
}

bool operator<(ReductionAttrs const &lhs, ReductionAttrs const &rhs) {
  return visit_lt(lhs, rhs);
}

/* ParallelTensorShape ReductionAttrs::output_shape(ParallelTensorShape const &input_shape) const { */
/*   ParallelTensorShape output = input_shape; */
/*   output.at(this->reduction_legion_dim).degree /= this->reduction_degree; */
/*   output.at(this->reduction_legion_dim).size /= this->reduction_degree; */
/*   return output; */
/* } */

}

namespace std {
using ::FlexFlow::ReductionAttrs;

size_t hash<ReductionAttrs>::operator()(
    ReductionAttrs const &params) const {
  return visit_hash(params);
} 
}
