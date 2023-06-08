#ifndef _FLEXFLOW_LINEAR_ATTRS_H
#define _FLEXFLOW_LINEAR_ATTRS_H

#include "op-attrs/activation.h"
#include "op-attrs/datatype.h"
#include "op-attrs/ops/core.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "utils/visitable.h"

namespace FlexFlow {

struct L1RegularizerAttrs : public use_visitable_cmp<L1RegularizerAttrs> {
public:
  L1RegularizerAttrs() = delete;
  explicit L1RegularizerAttrs(float);

public:
  float lambda;
};

struct L2RegularizerAttrs : public use_visitable_cmp<L2RegularizerAttrs> {
public:
  L2RegularizerAttrs() = delete;
  explicit L2RegularizerAttrs(float);

public:
  float lambda;
};

using RegularizerAttrs = variant<L1RegularizerAttrs, L2RegularizerAttrs>;

struct LinearAttrs : public use_visitable_cmp<LinearAttrs> {
public:
  LinearAttrs(int out_channels, bool use_bias, DataType data_type,
              Activation activation,
              optional<RegularizerAttrs> const &regularizer = nullopt);

public:
  int out_channels;
  bool use_bias;
  DataType data_type;
  Activation activation;
  optional<RegularizerAttrs> regularizer;
};

} // namespace FlexFlow

VISITABLE_STRUCT(::FlexFlow::L1RegularizerAttrs, lambda);
MAKE_VISIT_HASHABLE(::FlexFlow::L1RegularizerAttrs);

VISITABLE_STRUCT(::FlexFlow::L2RegularizerAttrs, lambda);
MAKE_VISIT_HASHABLE(::FlexFlow::L2RegularizerAttrs);

VISITABLE_STRUCT(::FlexFlow::LinearAttrs, out_channels, use_bias, data_type,
                 activation, regularizer);
MAKE_VISIT_HASHABLE(::FlexFlow::LinearAttrs);

namespace FlexFlow {
static_assert(is_well_behaved_value_type<RegularizerAttrs>::value, "");
static_assert(is_valid_opattr<LinearAttrs>::value, "");
} // namespace FlexFlow

#endif
