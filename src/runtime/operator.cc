#include "flexflow/operator.h"
#include "flexflow/simulator.h"

namespace FlexFlow {

size_t Op::get_untyped_params_hash() const {
  size_t hash = this->get_params_hash();
  hash_combine(hash, this->op_type);
  return hash;
}

size_t Op::get_params_hash() const {
  assert (false);
}

}; // namespace FlexFlow