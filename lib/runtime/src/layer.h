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

struct Layer : public use_visitable_cmp<Layer> {
public:
  Layer() = delete;
  Layer(LayerID,
        DataType,
        std::string const &name,
        CompGraphOperatorAttrs const &attrs);
public:
  LayerID guid;
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
    return {this->next_id(), data_type, name, attrs};
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

VISITABLE_STRUCT(::FlexFlow::Layer, guid, data_type, name, profiling, attrs);
MAKE_VISIT_HASHABLE(::FlexFlow::Layer);

namespace FlexFlow {

static_assert(is_equal_comparable<Layer>::value, "Layer must be comparable via ==");
static_assert(is_neq_comparable<Layer>::value, "Layer must be comparable via !=");
static_assert(is_lt_comparable<Layer>::value, "Layer must be comparable via <");
static_assert(std::is_copy_constructible<Layer>::value, "Layer must be copy constructible");
static_assert(std::is_move_constructible<Layer>::value, "Layer must be move constructible");
static_assert(std::is_copy_assignable<Layer>::value, "Layer must be copy assignable");
static_assert(std::is_move_assignable<Layer>::value, "Layer must be move assignable");
static_assert(!std::is_default_constructible<Layer>::value, "Layer must not be default constructible");

}

#endif
