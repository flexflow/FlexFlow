#ifndef _FLEXFLOW_RUNTIME_SRC_LAYER_H
#define _FLEXFLOW_RUNTIME_SRC_LAYER_H

#include "op-attrs/ffconst.h"
#include "layer_id.h"
#include "tensor.h"
#include "utils/optional.h"
#include "utils/stack_vector.h"
#include "utils/stack_string.h"
#include "op-attrs/op-attrs.h"
#include "utils/strong_typedef.h"

namespace FlexFlow {

struct layer_guid_t : strong_typedef<layer_guid_t, size_t> {
  using strong_typedef::strong_typedef;
};

class Layer {
public:
  Layer() = delete;
  Layer(LayerID,
        OperatorType,
        DataType,
        std::string const &name,
        CompGraphOperatorAttrs const &attrs, 
        Initializer *initializer);
  void add_initializer(std::string const &key, Initializer *initializer);
  Initializer *get_initializer(std::string const &key) const;
  Tensor get_parameter(int index);
public:
  OperatorType op_type;
  DataType data_type;
  LayerID layer_guid;
  stack_string<MAX_OPNAME> name;
  bool profiling;
  CompGraphOperatorAttrs attrs;
private:
  std::unordered_map<std::string, Initializer *> initializers;
};

struct LayerManager {
public:
  Layer create(CompGraphOperatorAttrs const &attrs,
        DataType data_type,
        std::string const &name) {
    return {this->next_id(), get_op_type(attrs), data_type, name, attrs};
  }

  template <typename Variant>
  Layer create(Variant const &attrs, DataType data_type, std::string const &name) {
    return this->create(variant_cast<CompGraphOperatorAttrs>(attrs), data_type, name);
  }

  LayerID next_id() {
    return LayerID(this->layer_global_guid++);
  }
private:
  size_t layer_global_guid = LAYER_GUID_FIRST_VALID;
};

}

#endif
