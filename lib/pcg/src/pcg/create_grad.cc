#include "pcg/create_grad.h"
#include <libassert/assert.hpp>

namespace FlexFlow {

bool bool_from_create_grad(CreateGrad cg) {
  switch (cg) {
    case CreateGrad::YES:
      return true;
    case CreateGrad::NO:
      return false;
    default:
      PANIC(fmt::format("Unknown CreateGrad value {}", cg));
  }
}

} // namespace FlexFlow
