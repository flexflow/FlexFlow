#include "layer.h"
#include "op-attrs/ffconst_utils.h"

namespace FlexFlow {

Layer::Layer(CompGraphOperatorAttrs const &_attrs,
             std::string const &_name)
  : attrs(_attrs), name(_name)
{ }

}
