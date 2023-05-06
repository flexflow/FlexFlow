#ifndef _FLEXFLOW_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_LOSS_FUNCTIONS_H
#define _FLEXFLOW_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_LOSS_FUNCTIONS_H

#include "utils/visitable.h"
#include "op-attrs/ffconst.h"
#include "utils/variant.h"

namespace FlexFlow {

struct SparseCategoricalCrossEntropyLossAttrs
  : public use_visitable_cmp<SparseCategoricalCrossEntropyLossAttrs> {
public:
  SparseCategoricalCrossEntropyLossAttrs() = delete;
  explicit SparseCategoricalCrossEntropyLossAttrs(bool replace_labels);

public:
  bool replace_labels; // for aggregate_spec: More predictions than labels
};

struct OtherLossAttrs {
public:
  explicit OtherLossAttrs() = delete;
  OtherLossAttrs(LossType);

public:
  LossType loss_type;
};

using LossAttrs = variant<SparseCategoricalCrossEntropyLossAttrs, OtherLossAttrs>;

LossType get_loss_type(OtherLossAttrs const &);
LossType get_loss_type(SparseCategoricalCrossEntropyLossAttrs const &);
LossType get_loss_type(LossAttrs const &);

}

VISITABLE_STRUCT(::FlexFlow::SparseCategoricalCrossEntropyLossAttrs, replace_labels);
VISITABLE_STRUCT(::FlexFlow::OtherLossAttrs, loss_type);

#endif
