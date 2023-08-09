#ifndef _FLEXFLOW_RUNTIME_SRC_TENSOR_USES_H
#define _FLEXFLOW_RUNTIME_SRC_TENSOR_USES_H

#include "layer.h"
#include "tensor.h"

namespace FlexFlow {

enum class TensorUseType { INPUT, WEIGHT, OUTPUT };

struct TensorUseDescription {
  TensorUseDescription() = delete;
  TensorUseDescription(TensorUseType const &, Layer const &, int);

  TensorUseType type;
  Layer const &layer;
  int idx;
};

struct TensorUses {
  TensorUses() = default;

  std::vector<TensorUseDescription> at(Tensor const &) const;
  std::vector<TensorUseDescription> at(size_t tensor_guid) const;

  void update(Layer const &);
  void remove(Layer const &);

private:
  std::unordered_map<size_t, std::vector<TensorUseDescription>> uses;
};

} // namespace FlexFlow

#endif
