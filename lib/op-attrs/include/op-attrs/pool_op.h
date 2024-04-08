#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_POOL_OP_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_POOL_OP_H

#include "nlohmann/json.hpp"
#include "utils/fmt.h"

namespace FlexFlow {

enum class PoolOp {
  MAX,
  AVG,
};

NLOHMANN_JSON_SERIALIZE_ENUM(PoolOp,
                             {{PoolOp::MAX, "MAX"}, {PoolOp::AVG, "AVG"}});

std::string format_as(PoolOp);

} // namespace FlexFlow
#endif
