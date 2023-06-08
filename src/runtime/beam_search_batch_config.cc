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

#define DEFAULT_BEAM_WIDTH 1
#define DEFAULT_TARGET_ITERATIONS 3

namespace FlexFlow {

LegionRuntime::Logger::Category log_beam_bc("BeamSearchBatchConfig");

BeamSearchBatchConfig::BeamSearchBatchConfig() : BatchConfig() {
  this->beam_width = DEFAULT_BEAM_WIDTH;
  this->target_iterations = DEFAULT_TARGET_ITERATIONS;
  current_iteration = 0;
}

BeamSearchBatchConfig::BeamSearchBatchConfig(int model_id) : BatchConfig() {
  this->model_id = model_id;
  std::cout << "==================\n"
            << "Register Batch Config with Model " << this->model_id
            << std::endl;
  current_iteration = 0;
}

BeamSearchBatchConfig::BeamSearchBatchConfig(size_t beam_width,
                                             size_t target_iterations)
    : BatchConfig() {
  this->beam_width = beam_width;
  this->target_iterations = target_iterations;
  current_iteration = 0;
}

BeamSearchBatchConfig::BeamSearchBatchConfig(BeamSearchBatchConfig const &other,
                                             int model_id)
    : BatchConfig() {
  this->beam_width = other.beam_width;
  this->target_iterations = other.target_iterations;
  this->model_id = model_id;
  current_iteration = 0;
}

BeamSearchBatchConfig::~BeamSearchBatchConfig() {}

InferenceMode BeamSearchBatchConfig::get_mode() const {
  return BEAM_SEARCH_MODE;
}

bool BeamSearchBatchConfig::done() const {
  assert(current_iteration <= target_iterations);
  return current_iteration == target_iterations;
}

int BeamSearchBatchConfig::max_beam_depth_all_requests() const {
  int max_depth_all_requests = 0;
  for (int i = 0; i < BeamSearchBatchConfig::MAX_NUM_REQUESTS; i++) {
    if (!request_completed[i] &&
        beamRequestsInfo[i].max_depth > max_depth_all_requests) {
      /* printf("\treq %i has max_depth=%i. Increasing max_depth_all_requests "
             "from %i\n",
             i,
             beamRequestsInfo[i].max_depth,
             max_depth_all_requests); */
      max_depth_all_requests = beamRequestsInfo[i].max_depth;
    }
  }
  assert(max_depth_all_requests <= BeamSearchBatchConfig::MAX_BEAM_DEPTH);
  return max_depth_all_requests;
}

int BeamSearchBatchConfig::current_depth_all_requests() const {
  int current_depth = 0;
  for (int i = 0; i < BeamSearchBatchConfig::MAX_NUM_REQUESTS; i++) {
    if (!request_completed[i] &&
        beamRequestsInfo[i].current_depth > current_depth) {
      /* printf("\treq %i has current_depth=%i. Increasing "
             "current_depth_all_requests from %i\n",
             i,
             beamRequestsInfo[i].current_depth,
             current_depth); */
      current_depth = beamRequestsInfo[i].current_depth;
    }
  }
  assert(current_depth <= BeamSearchBatchConfig::MAX_BEAM_DEPTH + 1);
  return current_depth;
}

void BeamSearchBatchConfig::print() const {
  std::cout << "@@@@@@@@@@@@@@ BeamSearchBatchConfig (mode " << get_mode()
            << ") @@@@@@@@@@@@@@" << std::endl;
  std::cout << "Max number of requests: " << MAX_NUM_REQUESTS << std::endl;
  std::cout << "Max number of tokens: " << MAX_NUM_TOKENS << std::endl;
  std::cout << "Number of tokens: " << num_tokens << std::endl;
  std::cout << "Number of requests: " << num_active_requests() << std::endl;
  std::cout << "Beam width: " << beam_width << std::endl;
  std::cout << "Target Iterations: " << target_iterations << std::endl;
  std::cout << "Current Iterations: " << current_iteration << std::endl;

  std::cout << "Per-request info:\n";
  for (int i = 0; i < MAX_NUM_REQUESTS; i++) {
    // assert(beamRequestsInfo[i].request_completed == request_completed[i]);
    if (!request_completed[i]) {
      std::cout << "  Request " << i << ":\n";
      std::cout << "    Token start offset: "
                << requestsInfo[i].token_start_offset << std::endl;
      std::cout << "    Number of tokens in batch: "
                << requestsInfo[i].num_tokens_in_batch << std::endl;
      std::cout << "    GUID: " << requestsInfo[i].request_guid << std::endl;
      std::cout << "    Max sequence length: "
                << requestsInfo[i].max_sequence_length << std::endl;
      std::cout << "    Beam Search Specific: " << std::endl;
      std::cout << "        beam_size: " << beamRequestsInfo[i].beam_size
                << std::endl;
      std::cout << "        current_depth: "
                << beamRequestsInfo[i].current_depth << std::endl;
      std::cout << "        max_depth: " << beamRequestsInfo[i].max_depth
                << std::endl;
      std::cout << "        tokens: ";
      for (int j = 0; j < MAX_BEAM_WIDTH; j++) {
        std::cout << beamRequestsInfo[i].tokens[j] << ", ";
      }
      std::cout << std::endl;
      std::cout << "        probs: ";
      for (int j = 0; j < MAX_BEAM_WIDTH; j++) {
        std::cout << beamRequestsInfo[i].probs[j] << ", ";
      }
      std::cout << std::endl;
      std::cout << "        parent_id: ";
      for (int j = 0; j < MAX_BEAM_WIDTH; j++) {
        std::cout << beamRequestsInfo[i].parent_id[j] << ", ";
      }
      std::cout << std::endl;
    }
  }

  std::cout << "Per-token info:\n";
  for (int i = 0; i < num_tokens; i++) {
    std::cout << "  Token " << i << ":\n";
    std::cout << "    Absolute depth in request: "
              << tokensInfo[i].abs_depth_in_request << std::endl;
    std::cout << "    Request index: " << tokensInfo[i].request_index
              << std::endl;
    std::cout << "    Token id: " << tokensInfo[i].token_id << std::endl;
    std::cout << "    Beam Search Specific: " << std::endl;
    std::cout << "        beam_size: " << beamTokenInfo[i].sub_request_index
              << std::endl;
    // std::cout << "    Parent token id: " << tokensInfo[i].parent_token_id <<
    // std::endl; std::cout << "    Accumulated log prob: "
    //           << tokensInfo[i].cum_log_prob << std::endl;
  }
  std::cout << "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
            << std::endl;
}

}; // namespace FlexFlow
