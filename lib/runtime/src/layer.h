#ifndef _FLEXFLOW_RUNTIME_SRC_LAYER_H
#define _FLEXFLOW_RUNTIME_SRC_LAYER_H

#include "op-attrs/ffconst.h"
#include "layer_id.h"
#include "tensor.h"
#include "utils/optional.h"
#include "utils/stack_vector.h"

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
        int numOutputs);
  Layer(FFModel *model,
        OperatorType otype,
        DataType dtype,
        char const *name,
        int numInputs,
        int numWeights,
        int numOutputs,
        optional<Tensor const &> input1,
        optional<Tensor const &> input2 = nullopt,
        optional<Tensor const &> input3 = nullopt,
        optional<Tensor const &> input4 = nullopt);
  Layer(FFModel *model,
        OperatorType otype,
        DataType dtype,
        char const *name,
        int numInputs,
        int numWeights,
        int numOutputs,
        Tensor const *tensors);
  void add_int_property(std::string const &key, long long value);
  void add_float_property(std::string const &key, float value);
  void add_int_vector_property(std::string const &key,
                               std::vector<int> const &value);
  void add_initializer(std::string const &key, Initializer *initializer);
  long long get_int_property(std::string const &key) const;
  float get_float_property(std::string const &key) const;
  std::vector<int> get_int_vector_property(std::string const &key) const;
  Initializer *get_initializer(std::string const &key) const;
  Tensor get_parameter(int index);

public:
  OperatorType op_type;
  DataType data_type;
  LayerID layer_guid;
  char name[MAX_OPNAME];
  stack_vector<Tensor, MAX_NUM_OUTPUTS> outputs;
  stack_vector<Tensor, MAX_NUM_INPUTS> inputs;
  stack_vector<Tensor, MAX_NUM_WEIGHTS> weights;
  stack_vector<Tensor, MAX_NUM_INPUTS> trainableInputs;
  int numInputs, numWeights, numOutputs;
  bool profiling;

private:
  std::unordered_map<std::string, long long> int_properties;
  std::unordered_map<std::string, float> float_properties;
  std::unordered_map<std::string, Initializer *> initializers;
  std::unordered_map<std::string, std::vector<int>> int_vector_properties;
};

}

#endif
