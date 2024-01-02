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
#include "flexflow/request_manager.h"
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
  for (int i = 0; i < BeamSearchBatchConfig::max_requests_per_batch(); i++) {
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

int BeamSearchBatchConfig::get_speculative_request_num() const {
  return speculative_request_num;
}

int BeamSearchBatchConfig::current_depth_all_requests() const {
  int current_depth = 0;
  for (int i = 0; i < BeamSearchBatchConfig::max_requests_per_batch(); i++) {
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

std::ostream &operator<<(std::ostream &os, BeamSearchBatchConfig const &bc) {
  os << "@@@@@@@@@@@@@@ BeamSearchBatchConfig (mode " << bc.get_mode()
     << ") @@@@@@@@@@@@@@" << std::endl;
  // Max values
  os << "Max number of requests: " << bc.max_requests_per_batch() << std::endl;
  os << "Max number of tokens: " << bc.max_tokens_per_batch() << std::endl;
  os << "Max sequence length: " << bc.max_sequence_length() << std::endl;
  // Current values
  os << "Number of tokens: " << bc.num_active_tokens() << std::endl;
  os << "Number of requests: " << bc.num_active_requests() << std::endl;
  // BeamSearch-specific
  os << "Model ID: " << bc.model_id << std::endl;
  os << "Max Beam Depth (all requests): " << bc.max_beam_depth_all_requests()
     << std::endl;
  os << "Current depth (all requests): " << bc.current_depth_all_requests()
     << std::endl;
  os << "Beam width: " << bc.beam_width << std::endl;
  os << "Target Iterations: " << bc.target_iterations << std::endl;
  os << "Current Iterations: " << bc.current_iteration << std::endl;

  os << "Per-request info:\n";
  for (int i = 0; i < bc.max_requests_per_batch(); i++) {
    if (!bc.request_completed[i]) {
      os << "  Request " << i << ":\n";
      os << "    First token depth in request: "
         << bc.requestsInfo[i].first_token_depth_in_request << std::endl;
      os << "    First token offset in batch: "
         << bc.requestsInfo[i].first_token_offset_in_batch << std::endl;
      os << "    Number of tokens in batch: "
         << bc.requestsInfo[i].num_tokens_in_batch << std::endl;
      os << "    GUID: " << bc.requestsInfo[i].request_guid << std::endl;
      os << "    Max sequence length: "
         << bc.requestsInfo[i].max_sequence_length << std::endl;
      os << "    Request completed: " << bc.request_completed[i] << std::endl;
      os << "    Request running: " << bc.request_running[i] << std::endl;
      os << "    Beam Search Specific: " << std::endl;
      os << "        beam_size: " << bc.beamRequestsInfo[i].beam_size
         << std::endl;
      os << "        current_depth: " << bc.beamRequestsInfo[i].current_depth
         << std::endl;
      os << "        max_depth: " << bc.beamRequestsInfo[i].max_depth
         << std::endl;
      os << "        tokens: ";
      for (int j = 0; j < bc.MAX_BEAM_WIDTH; j++) {
        os << bc.beamRequestsInfo[i].tokens[j] << ", ";
      }
      os << std::endl;
      os << "        probs: ";
      for (int j = 0; j < bc.MAX_BEAM_WIDTH; j++) {
        os << bc.beamRequestsInfo[i].probs[j] << ", ";
      }
      os << std::endl;
      os << "        parent_id: ";
      for (int j = 0; j < bc.MAX_BEAM_WIDTH; j++) {
        os << bc.beamRequestsInfo[i].parent_id[j] << ", ";
      }
      os << std::endl;
    }
  }

  os << "Per-token info:\n";
  for (int i = 0; i < bc.num_tokens; i++) {
    os << "  Token " << i << ":\n";
    os << "    Absolute depth in request: "
       << bc.tokensInfo[i].abs_depth_in_request << std::endl;
    os << "    Request index: " << bc.tokensInfo[i].request_index << std::endl;
    os << "    Token id: " << bc.tokensInfo[i].token_id << std::endl;
    os << "    Beam Search Specific: " << std::endl;
    os << "        beam_size: " << bc.beamTokenInfo[i].sub_request_index
       << std::endl;
  }
  os << "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@" << std::endl;
  return os;
}

void BeamSearchBatchConfig::print() const {
  std::cout << *this << std::endl;
}

void BeamSearchBatchConfig::save_to_file(std::string const &filename) const {
  std::ofstream outputFile(filename);
  if (outputFile.is_open()) {
    outputFile << *this << std::endl;
    outputFile.close();
  } else {
    std::cerr << "Error: Unable to open the batch config output file: "
              << filename << std::endl;
    assert(false);
  }
}

}; // namespace FlexFlow
