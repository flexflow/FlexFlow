#include "op-meta/ops/reduction.h"
#include "utils/hash-utils.h"

namespace FlexFlow {

bool operator==(ReductionAttrs const &lhs, ReductionAttrs const &rhs) {
  return lhs.as_tuple() == rhs.as_tuple();
}

bool operator<(ReductionAttrs const &lhs, ReductionAttrs const &rhs) {
  return lhs.as_tuple() < rhs.as_tuple();
}

ParallelTensorShape ReductionAttrs::output_shape(ParallelTensorShape const &input_shape) const {
  ParallelTensorShape output = input_shape;
  output.at(this->reduction_legion_dim).degree /= this->reduction_degree;
  output.at(this->reduction_legion_dim).size /= this->reduction_degree;
  return output;
}

}

namespace std {
using ::FlexFlow::ReductionAttrs;

size_t hash<ReductionAttrs>::operator()(
    ReductionAttrs const &params) const {
  return get_std_hash(params.as_tuple());
} 
}
