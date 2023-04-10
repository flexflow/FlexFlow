#include "layer.h"
#include "op-attrs/ffconst_utils.h"
#include "model.h"

namespace FlexFlow {

static std::string get_name(OperatorType op_type, char const *name, size_t const &layer_guid) {
  std::string pcname;
  if (name == nullptr) {
    pcname = get_operator_type_name(op_type);
  } else {
    pcname = std::string(name);
  }
  pcname = pcname + "_" + std::to_string(layer_guid);
  return pcname;
}

Layer::Layer(LayerID _layer_guid,
             OperatorType _op_type,
             DataType _data_type,
             std::string const &_name,
             CompGraphOperatorAttrs const &_attrs)
  : layer_guid(_layer_guid), op_type(_op_type), data_type(_data_type), name(_name), attrs(_attrs)
{ }

}
