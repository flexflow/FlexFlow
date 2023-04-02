#include "layer.h"
#include "op-attrs/ffconst_utils.h"
#include "model.h"

namespace FlexFlow {

Layer::Layer(FFModel *model,
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
      layer_guid(model->layer_global_guid++), numInputs(_numInputs),
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

Layer::Layer(FFModel *model,
             OperatorType _otype,
             DataType _dtype,
             char const *_name,
             int _numInputs,
             int _numWeights,
             int _numOutputs,
             Tensor const *_tensors)
    : op_type(_otype), data_type(_dtype),
      layer_guid(model->layer_global_guid++), numInputs(_numInputs),
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

void Layer::add_int_property(std::string const &key, long long value) {
  int_properties[key] = value;
}

void Layer::add_float_property(std::string const &key, float value) {
  float_properties[key] = value;
}

void Layer::add_int_vector_property(std::string const &key,
                                    std::vector<int> const &value) {
  int_vector_properties[key] = value;
}

void Layer::add_initializer(std::string const &key, Initializer *initializer) {
  initializers[key] = initializer;
}

long long Layer::get_int_property(std::string const &key) const {
  return this->int_properties.at(key);
}

float Layer::get_float_property(std::string const &key) const {
  return this->float_properties.at(key);
}

std::vector<int> Layer::get_int_vector_property(std::string const &key) const {
  return this->int_vector_properties.at(key);
}

Initializer *Layer::get_initializer(std::string const &key) const {
  return this->initializers.at(key);
}

Tensor Layer::get_parameter(int index) {
  return this->weights.at(index);
}

}
