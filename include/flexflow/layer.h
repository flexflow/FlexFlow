#pragma once

#include "tensor.h"
#include "ffconst.h"
#include "fftype.h"

namespace FlexFlow {

class FFModel;
class Layer {
public:
  Layer(FFModel* model,
        OperatorType type,
        const char* name,
        int numInputs,
        int numWeights,
        int numOutputs,
        const Tensor input1 = NULL,
        const Tensor input2 = NULL,
        const Tensor input3 = NULL,
        const Tensor input4 = NULL);
  Layer(FFModel* model,
        OperatorType type,
        const char* name,
        int numInputs,
        int numWeights,
        int numOutputs,
        const Tensor* tensors = NULL);
  void add_int_property(const std::string& key,
                        long long value);
  void add_float_property(const std::string& key,
                          float value);
  void add_int_vector_property(const std::string& key,
                               const std::vector<int>& value);
  void add_initializer(const std::string& key,
                       Initializer* initializer);
  bool get_int_property(const std::string& key,
                        long long& value) const;
  bool get_float_property(const std::string& key,
                          float& value) const;
  bool get_int_vector_property(const std::string& key,
                               std::vector<int>& value) const;
  bool get_initializer(const std::string& key,
                       Initializer* &initializer) const;
  Tensor get_parameter(int index);
  void print();
public:
  OperatorType op_type;
  DataType data_type;
  LayerID layer_guid;
  char name[MAX_OPNAME];
  Tensor outputs[MAX_NUM_OUTPUTS];
  Tensor inputs[MAX_NUM_INPUTS];
  Tensor weights[MAX_NUM_WEIGHTS];
  bool trainableInputs[MAX_NUM_INPUTS];
  int numInputs, numWeights, numOutputs;
  bool profiling;
private:
  std::unordered_map<std::string, long long> int_properties;
  std::unordered_map<std::string, float> float_properties;
  std::unordered_map<std::string, Initializer*> initializers;
  std::unordered_map<std::string, std::vector<int> > int_vector_properties;
};

}; // namespace FlexFlow
