#include "op-meta/ops/combine_params.h"
#include "utils/hash-utils.h"
#include "op-meta/visit_struct.h"

namespace FlexFlow {
namespace opmeta {

typename CombineParams::AsConstTuple CombineParams::as_tuple() const {
  return {this->combine_legion_dim, this->combine_degree};
}

bool operator==(CombineParams const &lhs, CombineParams const &rhs) {
  return visit_eq(lhs, rhs);
}

bool operator<(CombineParams const &lhs, CombineParams const &rhs) {
  return visit_lt(lhs, rhs);
}

bool CombineParams::is_valid(ParallelTensorShape const &input) const {
  return input.at(this->combine_legion_dim).degree % this->combine_degree == 0;
}

ParallelTensorShape CombineParams::output_shape(ParallelTensorShape const &input_shape) const {
  ParallelTensorShape output = input_shape;
  output.at(this->combine_legion_dim).degree /= this->combine_degree;
  return output;
}

}
}

namespace std {
using ::FlexFlow::opmeta::CombineParams;

size_t hash<CombineParams>::operator()(
    CombineParams const &params) const {
  return get_std_hash(params.as_tuple());
} 
}
