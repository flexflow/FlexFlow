#ifndef _FLEXFLOW_OPATTRS_INCLUDE_OPATTRS_ACTIVATION_H
#define _FLEXFLOW_OPATTRS_INCLUDE_OPATTRS_ACTIVATION_H

#include "utils/fmt.h"

namespace FlexFlow {

enum class Activation { RELU, SIGMOID, TANH, GELU };

}

namespace fmt {

template <>
struct formatter<::FlexFlow::Activation> : formatter<string_view> {
  template <typename FormatContext>
  auto format(::FlexFlow::Activation a, FormatContext &ctx) const
      -> decltype(ctx.out()) {
    using namespace FlexFlow;

    string_view name = "unknown";
    switch (a) {
      case Activation::RELU:
        name = "ReLU";
        break;
      case Activation::SIGMOID:
        name = "Sigmoid";
        break;
      case Activation::TANH:
        name = "Tanh";
        break;
      case Activation::GELU:
        name = "GeLU";
        break;
    }
    return formatter<string_view>::format(name, ctx);
  }
};

} // namespace fmt

#endif
