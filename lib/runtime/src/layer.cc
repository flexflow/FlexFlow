#include "layer.h"
#include "op-attrs/ffconst_utils.h"
#include "model.h"

namespace FlexFlow {

Layer::Layer(size_t _layer_guid,
             OperatorType _otype,
             DataType _dtype,
             char const *_name,
             int _numInputs,
             int _numWeights,
             int _numOutputs,
             optional<Tensor const &> _input1,
             optional<Tensor const &> _input2,
             optional<Tensor const &> _input3,
             optional<Tensor const &> _input4)
    : op_type(_otype), data_type(_dtype),
      layer_guid(_layer_guid), numInputs(_numInputs),
      numWeights(_numWeights), numOutputs(_numOutputs) {
  std::string pcname;
  if (_name == nullptr) {
    pcname = get_operator_type_name(op_type);
  } else {
    pcname = std::string(_name);
  }
  pcname = pcname + "_" + std::to_string(this->layer_guid.id);
  assert(pcname.length() < MAX_OPNAME);
  std::strcpy(name, pcname.c_str());

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
             Tensor const *_tensors)
    : op_type(_otype), data_type(_dtype),
      layer_guid(_layer_guid), numInputs(_numInputs),
      numWeights(_numWeights), numOutputs(_numOutputs) {
  std::string pcname;
  if (_name == nullptr) {
    pcname = get_operator_type_name(op_type);
  } else {
    pcname = std::string(_name);
  }
  pcname = pcname + "_" + std::to_string(layer_guid.id);
  assert(pcname.length() < MAX_OPNAME);
  std::strcpy(name, pcname.c_str());
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
