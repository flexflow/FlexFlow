#include "flexflow/layer.h"
#include "flexflow/model.h"

namespace FlexFlow {

Layer::Layer(FFModel* model,
             OperatorType _type,
             const char* _name,
             int _numInputs,
             int _numWeights,
             int _numOutputs,
             const Tensor _input1,
             const Tensor _input2,
             const Tensor _input3,
             const Tensor _input4)
: op_type(_type), numInputs(_numInputs), numWeights(_numWeights),
  numOutputs(_numOutputs)
{
  std::string pcname;
  if (_name == NULL) {
    pcname = model->get_operator_type_name(op_type);
  } else {
    pcname = std::string(_name);
  }
  pcname = pcname + "_" + std::to_string(op_guid);
  assert(pcname.length() < MAX_OPNAME);
  std::strcpy(name, pcname.c_str());
  std::vector<Tensor> tensors;
  tensors.push_back(_input1);
  tensors.push_back(_input2);
  tensors.push_back(_input3);
  tensors.push_back(_input4);
  for (int i = 0; i < numInputs; i++) {
    assert(tensors[i] != NULL);
    inputs[i] = tensors[i];
  }
  for (int i = 0; i < MAX_NUM_OUTPUTS; i++) {
    outputs[i] = NULL;
  }
}

Layer::Layer(FFModel* model,
             OperatorType _type,
             const char* _name,
             int _numInputs,
             int _numWeights,
             int _numOutputs,
             const Tensor* _tensors)
: op_type(_type), numInputs(_numInputs), numWeights(_numWeights),
  numOutputs(_numOutputs)
{
  std::string pcname;
  if (_name == NULL) {
    pcname = model->get_operator_type_name(op_type);
  } else {
    pcname = std::string(_name);
  }
  pcname = pcname + "_" + std::to_string(op_guid);
  assert(pcname.length() < MAX_OPNAME);
  std::strcpy(name, pcname.c_str());
  for (int i = 0; i < numInputs; i++) {
    assert(_tensors[i] != NULL);
    inputs[i] = _tensors[i];
  }
  for (int i = 0; i < MAX_NUM_OUTPUTS; i++) {
    outputs[i] = NULL;
  }
}

void Layer::add_int_property(const std::string& key, long long value)
{
  int_properties[key] = value;
}

void Layer::add_float_property(const std::string& key, float value)
{
  float_properties[key] = value;
}

void Layer::add_initializer(const std::string& key,
                            Initializer* initializer)
{
  initializers[key] = initializer;
}

bool Layer::get_int_property(const std::string& key, long long& value) const
{
  const auto& it = int_properties.find(key);
  if (it == int_properties.end()) {
    assert(false);
    return false;
  } else {
    value = it->second;
    return true;
  }
}

bool Layer::get_float_property(const std::string& key, float& value) const
{
  const auto& it = float_properties.find(key);
  if (it == float_properties.end()) {
    assert(false);
    return false;
  } else {
    value = it->second;
    return true;
  }
}

bool Layer::get_initializer(const std::string& key,
                            Initializer*& initializer) const
{
  const auto& it = initializers.find(key);
  if (it == initializers.end()) {
    assert(false);
    return false;
  } else {
    initializer = it->second;
    return true;
  }
}

void Layer::print()
{}

}; // namespace FlexFlow
