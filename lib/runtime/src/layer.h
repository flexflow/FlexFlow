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

class Layer {
public:
  Layer() = delete;
  Layer(LayerID,
        OperatorType,
        DataType,
        std::string const &name,
        CompGraphOperatorAttrs const &attrs);
public:
  LayerID layer_guid;
  OperatorType op_type;
  DataType data_type;
  stack_string<MAX_OPNAME> name;
  bool profiling;
  CompGraphOperatorAttrs attrs;
};

struct LayerManager {
public:
  Layer create(CompGraphOperatorAttrs const &attrs,
        DataType data_type,
        std::string const &name) {
    return {this->next_id(), get_op_type(attrs), data_type, name, attrs};
  }

  template <typename ...Args>
  Layer create(variant<Args...> const &attrs, DataType data_type, std::string const &name) {
    return this->create(widen<CompGraphOperatorAttrs>(attrs), data_type, name);
  }

private:
  LayerID next_id() {
    return LayerID(this->layer_global_guid++);
  }
private:
  size_t layer_global_guid = LAYER_GUID_FIRST_VALID;
};

}

#endif
