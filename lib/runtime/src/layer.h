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
  Layer(CompGraphOperatorAttrs const &attrs,
        std::string const &name);

public:
  stack_string<MAX_OPNAME> name;
  CompGraphOperatorAttrs attrs;
};

}

VISITABLE_STRUCT(::FlexFlow::Layer, attrs, name);
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
static_assert(is_fmtable<Layer>::value, "Layer must be fmtable");

}

#endif
