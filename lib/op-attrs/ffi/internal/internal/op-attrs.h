#ifndef _FLEXFLOW_OPATTRS_FFI_INTERNAL_INTERNAL_OPATTRS_H
#define _FLEXFLOW_OPATTRS_FFI_INTERNAL_INTERNAL_OPATTRS_H

#include "flexflow/op-attrs.h"
#include "internal/opaque.h"
#include "op-attrs/activation.h"
#include "op-attrs/datatype.h"
#include "op-attrs/op.h"
#include "op-attrs/ops/embedding.h"
#include "op-attrs/ops/linear.h"
#include "op-attrs/ops/pool_2d.h"
#include "op-attrs/param_sync.h"

using namespace FlexFlow;

REGISTER_OPAQUE(flexflow_regularizer_attrs_t, optional<RegularizerAttrs>);

optional<ParamSync> to_internal(flexflow_param_sync_t);
flexflow_param_sync_t to_external(optional<ParamSync>);

DataType to_internal(flexflow_datatype_t);
flexflow_datatype_t to_external(DataType);

optional<Activation> to_internal(flexflow_activation_t);
flexflow_activation_t to_external(optional<Activation>);

PoolOp to_internal(flexflow_pool_op_t e);
flexflow_pool_op_t to_external(PoolOp i);

AggregateOp to_internal(flexflow_aggregate_op_t e);
flexflow_aggregate_op_t to_external(AggregateOp i);

OperatorType to_internal(flexflow_op_type_t e);
flexflow_op_type_t to_external(OperatorType i);

#endif
