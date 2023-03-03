#include "op-meta/ops/combine.h"
#include "utils/hash-utils.h"
#include "op-meta/visit_struct.h"

namespace FlexFlow {

bool operator==(CombineAttrs const &lhs, CombineAttrs const &rhs) {
  return visit_eq(lhs, rhs);
}

bool operator<(CombineAttrs const &lhs, CombineAttrs const &rhs) {
  return visit_lt(lhs, rhs);
}

bool CombineAttrs::is_valid(ParallelTensorShape const &input) const {
  return input.at(this->combine_legion_dim).degree % this->combine_degree == 0;
}

ParallelTensorShape CombineAttrs::output_shape(ParallelTensorShape const &input_shape) const {
  ParallelTensorShape output = input_shape;
  output.at(this->combine_legion_dim).degree /= this->combine_degree;
  return output;
}

}

namespace std {
using ::FlexFlow::CombineAttrs;

size_t hash<CombineAttrs>::operator()(
    CombineAttrs const &params) const {
  return visit_hash(params);
} 
}
