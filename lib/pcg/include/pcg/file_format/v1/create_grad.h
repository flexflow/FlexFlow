#ifndef _FLEXFLOW_PCG_INCLUDE_PCG_FILE_FORMAT_V1_CREATE_GRAD_H
#define _FLEXFLOW_PCG_INCLUDE_PCG_FILE_FORMAT_V1_CREATE_GRAD_H

#include "pcg/create_grad.h"
#include "utils/json.h"

namespace FlexFlow {

enum class V1CreateGrad { YES, NO };

NLOHMANN_JSON_SERIALIZE_ENUM(V1CreateGrad,
                             {{V1CreateGrad::YES, "YES"},
                              {V1CreateGrad::NO, "NO"}});

V1CreateGrad to_v1(CreateGrad const &cg);

} // namespace FlexFlow

#endif
