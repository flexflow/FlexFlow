#pragma once

#include "ffconst.h"
#include "fftype.h"
#include "tensor.h"

namespace FlexFlow {

class FFModel;
class Layer {
public:
  Layer(FFModel *model,
        OperatorType otype,
        DataType dtype,
        char const *name,
        int numInputs,
        int numWeights,
        int numOutputs,
        const Tensor input1 = NULL,
        const Tensor input2 = NULL,
        const Tensor input3 = NULL,
        const Tensor input4 = NULL);
  Layer(FFModel *model,
        OperatorType otype,
        DataType dtype,
        char const *name,
        int numInputs,
        int numWeights,
        int numOutputs,
        Tensor const *tensors = NULL);
  void add_int_property(std::string const &key, long long value);
  void add_float_property(std::string const &key, float value);
  void add_int_vector_property(std::string const &key,
                               std::vector<int> const &value);
  void add_initializer(std::string const &key, Initializer *initializer);
  bool get_int_property(std::string const &key, long long &value) const;
  bool get_float_property(std::string const &key, float &value) const;
  bool get_int_vector_property(std::string const &key,
                               std::vector<int> &value) const;
  bool get_initializer(std::string const &key, Initializer *&initializer) const;
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
  bool inference_debugging;

private:
  std::unordered_map<std::string, long long> int_properties;
  std::unordered_map<std::string, float> float_properties;
  std::unordered_map<std::string, Initializer *> initializers;
  std::unordered_map<std::string, std::vector<int>> int_vector_properties;
};

}; // namespace FlexFlow
