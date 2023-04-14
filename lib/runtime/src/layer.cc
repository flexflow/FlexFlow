#include "layer.h"
#include "op-attrs/ffconst_utils.h"
#include "model.h"

namespace FlexFlow {

Layer::Layer(LayerID _guid,
             DataType _data_type,
             std::string const &_name,
             CompGraphOperatorAttrs const &_attrs)
  : guid(_guid), data_type(_data_type), name(_name), attrs(_attrs)
{ }

}
