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

Layer::Layer(size_t _layer_guid,
             OperatorType _otype,
             DataType _dtype,
             char const *_name,
             int _numInputs,
             int _numWeights,
             int _numOutputs,
             CompGraphOperatorAttrs const &_attrs,
             optional<Tensor const &> _input1,
             optional<Tensor const &> _input2,
             optional<Tensor const &> _input3,
             optional<Tensor const &> _input4)
    : op_type(_otype), data_type(_dtype),
      layer_guid(_layer_guid), numInputs(_numInputs),
      numWeights(_numWeights), numOutputs(_numOutputs), attrs(_attrs),
      name(get_name(_otype, _name, _layer_guid)) {

  std::vector<optional<Tensor const &>> tensors;
  tensors.push_back(_input1);
  tensors.push_back(_input2);
  tensors.push_back(_input3);
  tensors.push_back(_input4);

  for (int i = 0; i < numInputs; i++) {
    inputs[i] = tensors[i].value();
  }
}

Layer::Layer(size_t _layer_guid,
             OperatorType _otype,
             DataType _dtype,
             char const *_name,
             int _numInputs,
             int _numWeights,
             int _numOutputs,
             CompGraphOperatorAttrs const &_attrs,
             Tensor const *_tensors)
    : op_type(_otype), data_type(_dtype),
      layer_guid(_layer_guid), numInputs(_numInputs),
      numWeights(_numWeights), numOutputs(_numOutputs), attrs(_attrs),
      name(get_name(_otype, _name, _layer_guid)) {

  for (int i = 0; i < numInputs; i++) {
    inputs[i] = _tensors[i];
  }
}

void Layer::add_initializer(std::string const &key, Initializer *initializer) {
  initializers[key] = initializer;
}

Initializer *Layer::get_initializer(std::string const &key) const {
  return this->initializers.at(key);
}

Tensor Layer::get_parameter(int index) {
  return this->weights.at(index);
}

}
