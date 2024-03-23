#include "pcg/layer.h"

namespace FlexFlow {

Layer::Layer(CompGraphOperatorAttrs const &_attrs,
             std::optional<std::string> const &_name)
    : attrs(_attrs), name(_name) {}

} // namespace FlexFlow
