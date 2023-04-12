#include "layer.h"
#include "op-attrs/ffconst_utils.h"
#include "model.h"
#include "utils/visitable_funcs.h"

namespace FlexFlow {

Layer::Layer(LayerID _guid,
             DataType _data_type,
             std::string const &_name,
             CompGraphOperatorAttrs const &_attrs)
  : guid(_guid), data_type(_data_type), name(_name), attrs(_attrs)
{ }

bool operator==(Layer const &lhs, Layer const &rhs) {
  return visit_eq(lhs, rhs);
}

bool operator!=(Layer const &lhs, Layer const &rhs) {
  return visit_neq(lhs, rhs);
}

bool operator<(Layer const &lhs, Layer const &rhs) {
  return visit_lt(lhs, rhs);
}

}

namespace std {

using ::FlexFlow::Layer;

size_t hash<Layer>::operator()(Layer const &layer) const {
  return visit_hash(layer);
}

}
