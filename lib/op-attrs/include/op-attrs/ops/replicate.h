#ifndef _FLEXFLOW_REPLICATE_ATTRS_H
#define _FLEXFLOW_REPLICATE_ATTRS_H

#include "op-attrs/parallel_tensor_shape.h"
#include "utils/visitable.h"
#include "op-attrs/ff_dim.h"

namespace FlexFlow {

struct ReplicateAttrs {
public:
  ReplicateAttrs() = delete;
  ReplicateAttrs(ff_dim_t dim, int degree);
/* public: */
/*   ParallelTensorShape output_shape(ParallelTensorShape const &input_shape) const override; */
/*   OperatorType op_type() const override; */
public:
  ff_dim_t replicate_dim;
  int replicate_degree;
};

}

VISITABLE_STRUCT(::FlexFlow::ReplicateAttrs, replicate_dim, replicate_degree);
MAKE_VISIT_HASHABLE(::FlexFlow::ReplicateAttrs);

#endif 
