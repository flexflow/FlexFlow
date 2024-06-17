#include "pcg/initializers/uniform_initializer_attrs.h"

namespace rc {

using ::FlexFlow::UniformInitializerAttrs;

Gen<UniformInitializerAttrs> Arbitrary<UniformInitializerAttrs>::arbitrary() {
  return gen::map<std::tuple<float, float, int>>([](std::tuple<float, float, int> const &generated) {
    auto [f1, f2, seed] = generated; 
    float minval = std::min(f1, f2);
    float maxval = std::max(f1, f2);
    return ::FlexFlow::UniformInitializerAttrs{
      seed,
      minval,
      maxval,
    };
  });
};

}
