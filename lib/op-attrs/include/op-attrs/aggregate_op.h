#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_AGGREGATE_OP_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_AGGREGATE_OP_H

#include "nlohmann/json.hpp"
#include "utils/fmt.h"

namespace FlexFlow {

enum class AggregateOp {
  SUM,
  AVG,
};

NLOHMANN_JSON_SERIALIZE_ENUM(AggregateOp,
                             {{AggregateOp::SUM, "SUM"},
                              {AggregateOp::AVG, "AVG"}});

std::string format_as(AggregateOp);

} // namespace FlexFlow

#endif
