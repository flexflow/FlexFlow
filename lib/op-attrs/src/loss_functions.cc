#include "op-attrs/ops/loss_functions.h"
#include <cassert>

namespace FlexFlow {

SparseCategoricalCrossEntropyLossAttrs::SparseCategoricalCrossEntropyLossAttrs(bool _replace_labels)
  : replace_labels(_replace_labels)
{ }

OtherLossAttrs::OtherLossAttrs(LossType _loss_type)
  : loss_type(_loss_type) {
  assert (this->loss_type != LOSS_SPARSE_CATEGORICAL_CROSSENTROPY);
}

LossType get_loss_type(OtherLossAttrs const &attrs) { return attrs.loss_type; }
LossType get_loss_type(SparseCategoricalCrossEntropyLossAttrs const &attrs) { return LOSS_SPARSE_CATEGORICAL_CROSSENTROPY; }

struct GetLossType {
  template <typename T>
  LossType operator()(T const &t) {
    return get_loss_type(t);
  }
};

LossType get_loss_type(LossAttrs const &attrs) {
  return visit(GetLossType{}, attrs);
}

}
