#ifndef _FLEXFLOW_OP_ATTRS_OPS_NOOP_H
#define _FLEXFLOW_OP_ATTRS_OPS_NOOP_H

#include "utils/visitable.h"
#include <functional>

namespace FlexFlow {

struct NoopAttrs { };

bool operator==(NoopAttrs const &, NoopAttrs const &);
bool operator!=(NoopAttrs const &, NoopAttrs const &);
bool operator<(NoopAttrs const &, NoopAttrs const &);

struct InputAttrs { };

bool operator==(InputAttrs const &, InputAttrs const &);
bool operator!=(InputAttrs const &, InputAttrs const &);
bool operator<(InputAttrs const &, InputAttrs const &);

}

VISITABLE_STRUCT_EMPTY(::FlexFlow::NoopAttrs);
VISITABLE_STRUCT_EMPTY(::FlexFlow::InputAttrs);

namespace std {

template <>
struct hash<::FlexFlow::NoopAttrs> {
  size_t operator()(::FlexFlow::NoopAttrs const &) const;
};

template <>
struct hash<::FlexFlow::InputAttrs> {
  size_t operator()(::FlexFlow::InputAttrs const &) const;
};

}

#endif
