#ifndef _FLEXFLOW_PCG_FILE_FORMAT_V1_PARAM_SYNC_H
#define _FLEXFLOW_PCG_FILE_FORMAT_V1_PARAM_SYNC_H

#include "utils/json.h"

namespace FlexFlow {

enum class V1ParamSync { PARAM_SERVER, NCCL };

NLOHMANN_JSON_SERIALIZE_ENUM(V1ParamSync,
                             {{V1ParamSync::PARAM_SERVER, "PARAM_SERVER"},
                              {V1ParamSync::NCCL, "NCCL"}});

} // namespace FlexFlow

#endif
