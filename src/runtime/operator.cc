#include "flexflow/operator.h"
#include "flexflow/ffconst_utils.h"
#include "flexflow/simulator.h"
#include <stdexcept>

namespace FlexFlow {

size_t Op::get_untyped_params_hash() const {
  size_t hash = this->get_params_hash();
  hash_combine(hash, this->op_type);
  return hash;
}

size_t Op::get_params_hash() const {
  throw std::runtime_error(
      "No overload of get_params_hash defined for op type " +
      get_operator_type_name(this->op_type));
}

}; // namespace FlexFlow