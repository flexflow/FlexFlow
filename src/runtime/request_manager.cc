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

#include "flexflow/inference.h"
#include "flexflow/parallel_ops/parallel_op.h"

namespace FlexFlow {

using namespace Legion;

RequestManager::RequestManager() : next_available_guid(1000000) {}

RequestManager::RequestGuid
    RequestManager::register_new_request(std::vector<TokenId> const &prompt) {
  const std::lock_guard<std::mutex> lock(request_queue_mutex);
  RequestGuid guid = next_available_guid++;

  // Add a new request
  pending_request_queue[guid] = prompt;
  return guid;
}

bool RequestManager::prepare_next_batch(BatchConfig &bc) {
  const std::lock_guard<std::mutex> lock(request_queue_mutex);
  return true;
};

}; // namespace FlexFlow
