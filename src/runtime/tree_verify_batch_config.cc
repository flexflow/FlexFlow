/* Copyright 2023 CMU, Stanford, Facebook, LANL
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

#include "flexflow/batch_config.h"
#include "legion.h"
#include <cassert>
#include <climits>

namespace FlexFlow {

LegionRuntime::Logger::Category log_tree_bc("TreeVerifyBatchConfig");

TreeVerifyBatchConfig::TreeVerifyBatchConfig() : BatchConfig() {}

TreeVerifyBatchConfig::~TreeVerifyBatchConfig() {}

BatchConfig::Mode TreeVerifyBatchConfig::get_mode() const {
  return TREE_VERIFY_MODE;
}

}; // namespace FlexFlow
