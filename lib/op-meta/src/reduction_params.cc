#include "op-meta/ops/reduction_params.h"
#include "utils/hash-utils.h"

namespace FlexFlow {
namespace opmeta {

typename ReductionParams::AsConstTuple ReductionParams::as_tuple() const {
  return {this->reduction_legion_dim, this->reduction_degree};
}

bool operator==(ReductionParams const &lhs, ReductionParams const &rhs) {
  return lhs.as_tuple() == rhs.as_tuple();
}

bool operator<(ReductionParams const &lhs, ReductionParams const &rhs) {
  return lhs.as_tuple() < rhs.as_tuple();
}

ParallelTensorShape ReductionParams::output_shape(ParallelTensorShape const &input_shape) const {
  ParallelTensorShape output = input_shape;
  output.at(this->reduction_legion_dim).degree /= this->reduction_degree;
  output.at(this->reduction_legion_dim).size /= this->reduction_degree;
  return output;
}

}
}

namespace std {
using ::FlexFlow::opmeta::ReductionParams;

size_t hash<ReductionParams>::operator()(
    ReductionParams const &params) const {
  return get_std_hash(params.as_tuple());
} 
}
