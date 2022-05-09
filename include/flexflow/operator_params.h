/**
 * @file operator_params.h
 * @brief Common Header for Operator Parameters
 *
 * @copyright Copyright 2022 Stanford
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef _OPERATOR_PARAMS_H
#define _OPERATOR_PARAMS_H

#include "mpark/variant.hpp"

#include "flexflow/ops/conv_2d.h"
#include "flexflow/ops/linear.h"
#include "flexflow/ops/noop.h"

namespace mp = mpark;

namespace FlexFlow {


using OperatorParameters = mp::variant<Conv2DParams, LinearParams, NoOpParams>;

};  // namespace FlexFlow

#endif  // _OPERATOR_PARAMS_H
