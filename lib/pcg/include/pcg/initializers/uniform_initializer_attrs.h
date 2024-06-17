#ifndef _FLEXFLOW_LIB_PCG_INCLUDE_PCG_INITIALIZERS_UNIFORM_INITIALIZER_ATTRS_H
#define _FLEXFLOW_LIB_PCG_INCLUDE_PCG_INITIALIZERS_UNIFORM_INITIALIZER_ATTRS_H

#include "pcg/initializers/uniform_initializer_attrs.dtg.h"
#include <rapidcheck.h>

namespace rc {

template <>
struct Arbitrary<::FlexFlow::UniformInitializerAttrs> {
  static Gen<::FlexFlow::UniformInitializerAttrs> arbitrary();
};

}

#endif
