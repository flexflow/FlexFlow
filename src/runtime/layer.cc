#include "flexflow/layer.h"
#include "flexflow/ffconst_utils.h"
#include "flexflow/model.h"

namespace FlexFlow {

Layer::Layer(FFModel *model,
             OperatorType _otype,
             DataType _dtype,
             char const *_name,
             int _numInputs,
             int _numWeights,
             int _numOutputs,
             const Tensor _input1,
             const Tensor _input2,
             const Tensor _input3,
             const Tensor _input4)
    : op_type(_otype), data_type(_dtype),
      layer_guid(model->layer_global_guid++,
                 model->current_transformer_layer_id,
                 model->model_id),
      numInputs(_numInputs), numWeights(_numWeights), numOutputs(_numOutputs) {
  std::string pcname;
  if (_name == nullptr) {
    pcname = get_operator_type_name(op_type);
  } else {
    pcname = std::string(_name);
  }
  pcname = pcname + "_" + std::to_string(this->layer_guid.id);
  assert(pcname.length() < MAX_OPNAME);
  std::strcpy(name, pcname.c_str());
  std::vector<Tensor> tensors;
  tensors.push_back(_input1);
  tensors.push_back(_input2);
  tensors.push_back(_input3);
  tensors.push_back(_input4);
  for (int i = 0; i < numInputs; i++) {
    assert(tensors[i] != nullptr);
    inputs[i] = tensors[i];
  }
  for (int i = 0; i < MAX_NUM_OUTPUTS; i++) {
    outputs[i] = nullptr;
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
      layer_guid(model->layer_global_guid++,
                 model->current_transformer_layer_id,
                 model->model_id),
      numInputs(_numInputs), numWeights(_numWeights), numOutputs(_numOutputs) {
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
    assert(_tensors[i] != nullptr);
    inputs[i] = _tensors[i];
  }
  for (int i = 0; i < MAX_NUM_OUTPUTS; i++) {
    outputs[i] = nullptr;
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

bool Layer::get_int_property(std::string const &key, long long &value) const {
  auto const &it = int_properties.find(key);
  if (it == int_properties.end()) {
    assert(false);
    return false;
  } else {
    value = it->second;
    return true;
  }
}

bool Layer::get_float_property(std::string const &key, float &value) const {
  auto const &it = float_properties.find(key);
  if (it == float_properties.end()) {
    assert(false);
    return false;
  } else {
    value = it->second;
    return true;
  }
}

bool Layer::get_int_vector_property(std::string const &key,
                                    std::vector<int> &value) const {
  auto const &it = int_vector_properties.find(key);
  if (it == int_vector_properties.end()) {
    assert(false);
    return false;
  } else {
    value = it->second;
    return true;
  }
}

bool Layer::get_initializer(std::string const &key,
                            Initializer *&initializer) const {
  auto const &it = initializers.find(key);
  if (it == initializers.end()) {
    assert(false);
    return false;
  } else {
    initializer = it->second;
    return true;
  }
}

void Layer::print() {}

Tensor Layer::get_parameter(int index) {
  assert(index < numWeights);
  return weights[index];
}

}; // namespace FlexFlow
