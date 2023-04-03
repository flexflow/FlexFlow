#ifndef _FLEXFLOW_RUNTIME_SRC_LAYER_H
#define _FLEXFLOW_RUNTIME_SRC_LAYER_H

#include "op-attrs/ffconst.h"
#include "layer_id.h"
#include "tensor.h"
#include "utils/optional.h"
#include "utils/stack_vector.h"
#include "utils/stack_string.h"
#include "op-attrs/op-attrs.h"

namespace FlexFlow {

class FFModel;
class Layer {
public:
  Layer() = delete;
  Layer(size_t layer_guid,
        OperatorType otype,
        DataType dtype,
        char const *name,
        int numInputs,
        int numWeights,
        int numOutputs,
        CompGraphOperatorAttrs const &attrs);
  Layer(size_t layer_guid,
        OperatorType otype,
        DataType dtype,
        char const *name,
        int numInputs,
        int numWeights,
        int numOutputs,
        CompGraphOperatorAttrs const &attrs,
        optional<Tensor const &> input1,
        optional<Tensor const &> input2 = nullopt,
        optional<Tensor const &> input3 = nullopt,
        optional<Tensor const &> input4 = nullopt);
  Layer(size_t layer_guid,
        OperatorType otype,
        DataType dtype,
        char const *name,
        int numInputs,
        int numWeights,
        int numOutputs,
        CompGraphOperatorAttrs const &attrs,
        Tensor const *tensors);
  Layer(size_t layer_guid,
        CompGraphOperatorAttrs const &attrs,
        DataType data_type,
        char const *name,
        std::vector<Tensor> const &inputs,
        std::vector<Tensor> const &weights,
        std::vector<Tensor> const &outputs);
  void add_initializer(std::string const &key, Initializer *initializer);
  Initializer *get_initializer(std::string const &key) const;
  Tensor get_parameter(int index);

public:
  OperatorType op_type;
  DataType data_type;
  LayerID layer_guid;
  stack_string<MAX_OPNAME> name;
  stack_vector<Tensor, MAX_NUM_OUTPUTS> outputs;
  stack_vector<Tensor, MAX_NUM_INPUTS> inputs;
  stack_vector<Tensor, MAX_NUM_WEIGHTS> weights;
  stack_vector<bool, MAX_NUM_INPUTS> trainableInputs;
  int numInputs, numWeights, numOutputs;
  bool profiling;
  CompGraphOperatorAttrs attrs;
private:
  std::unordered_map<std::string, Initializer *> initializers;
};

struct LayerManager {
public:
  template <typename ...Args>
  Layer *create(Args&&...args) {
    layers.emplace_back(new Layer(this->layer_global_guid++, std::forward<Args>(args)...));
    return this->layers.back().get();
  }
private:
  std::vector<std::unique_ptr<Layer>> layers;
  size_t layer_global_guid = LAYER_GUID_FIRST_VALID;
};

}

#endif
