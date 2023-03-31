/* Copyright 2023 CMU, Facebook, LANL, MIT, NVIDIA, and Stanford (alphabetical)
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

#include <cassert>
#include <chrono>
#include <ctime>
#include <iostream>
#include <iterator>
#include <math.h>
#include <random>
#include <thread>
#include <unistd.h>

using namespace std;

typedef std::chrono::high_resolution_clock Clock;
typedef std::chrono::milliseconds milliseconds;

class DataGenerator {
public:
  DataGenerator(size_t _num_requests,
                size_t _vocab_size,
                size_t _min_input_tokens,
                size_t _max_input_tokens,
                size_t _min_tokens_to_generate,
                size_t _max_tokens_to_generate,
                bool _poisson_distr,
                double _lambda);

  // Generate random requests by filling each tensor with random tokens. For
  // now, assume all requests have the same sequence length.
  void generate_requests(int *req_ptr);
  void start_timer(void);
  // Get number of requests that have arrived since the last time this function
  // was called
  std::pair<size_t, size_t> get_requests(size_t max_requests,
                                         size_t max_tokens);
  std::pair<size_t, size_t> get_request_length(size_t guid);

private:
  // Compute the arrival times of each request and save them in the arrivals
  // vector.
  // void generate_arrival_times(void);
  void generate_requests_meta();

  size_t num_requests; // total number of requests
  size_t vocab_size;   // number of words in the vocab
  size_t min_input_tokens;
  size_t max_input_tokens;
  size_t min_tokens_to_generate;
  size_t max_tokens_to_generate;
  bool poisson_distr; // false implies uniform distribution
  double lambda;      // mean #num of arrivals per sec
  bool timer_started; // whether timer was initiated
  // time when get_requests() is called for the first time
  Clock::time_point start_time;
  // arrival times (ms) generated based on distribution
  std::vector<double> arrivals;
  std::vector<double>::iterator arrivals_ptr;
  // sequence lengths generated based on uniform distribution
  std::vector<std::pair<size_t, size_t>> seq_lengths;
};
