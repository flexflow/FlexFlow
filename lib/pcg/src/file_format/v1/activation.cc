#include "pcg/file_format/v1/activation.h"

namespace FlexFlow {

V1Activation to_v1(Activation const &a) {
  // There should be a better way of doing this.
  switch (a) {
    case Activation::RELU:
      return V1Activation::RELU;
    case Activation::SIGMOID:
      return V1Activation::SIGMOID;
    case Activation::TANH:
      return V1Activation::TANH;
    case Activation::GELU:
      return V1Activation::GELU;
    default:
      NOT_REACHABLE();
  }
}

} // namespace FlexFlow
