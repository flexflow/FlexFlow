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

#pragma once

#include <cstdlib>


namespace FlexFlow {

struct InferenceResult {
  static const int MAX_NUM_TOKENS = 1024;
  int results[MAX_NUM_TOKENS];
};

class BatchConfig {
public:
  BatchConfig();
  bool register_new_request(int guid, int length);
  void prepare_next_batch();
  int update_results(InferenceResult const &ir);
  int num_processing_requests();
  static const int MAX_NUM_REQUESTS = 256;
  static const int MAX_NUM_TOKENS = InferenceResult::MAX_NUM_TOKENS;
  static const int MAX_SEQUENCE_LENGTH = 1024;
  // These are set by update
  int token_start_idx[MAX_NUM_REQUESTS];
  int token_last_available_idx[MAX_NUM_REQUESTS];
  int num_processing_tokens[MAX_NUM_REQUESTS];
  size_t request_guid[MAX_NUM_REQUESTS];
  // This is set by the app
  bool request_completed[MAX_NUM_REQUESTS];
};

}; // namespace FlexFlow
