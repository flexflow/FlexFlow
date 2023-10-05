#ifndef _FLEXFLOW_PCG_INCLUDE_PCG_FILE_FORMAT_V1_ACTIVATION_H
#define _FLEXFLOW_PCG_INCLUDE_PCG_FILE_FORMAT_V1_ACTIVATION_H

#include "op-attrs/activation.h"
#include "utils/json.h"

namespace FlexFlow {

enum class V1Activation { RELU, SIGMOID, TANH, GELU };

NLOHMANN_JSON_SERIALIZE_ENUM(V1Activation,
                             {{V1Activation::RELU, "RELU"},
                              {V1Activation::SIGMOID, "SIGMOID"},
                              {V1Activation::TANH, "TANH"},
                              {V1Activation::GELU, "GELU"}});

V1Activation to_v1(Activation const &a);
Activation from_v1(V1Activation const &va);

} // namespace FlexFlow

#endif
