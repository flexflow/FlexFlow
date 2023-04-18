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

#include "data_generator.h"
#include "flexflow/batch_config.h"
#include <algorithm>
#include <iostream>
#include <vector>
using namespace std;

DataGenerator::DataGenerator(size_t _num_requests,
                             size_t _vocab_size,
                             size_t _min_input_tokens,
                             size_t _max_input_tokens,
                             size_t _min_tokens_to_generate,
                             size_t _max_tokens_to_generate,
                             bool _poisson_distr,
                             double _lambda)
    : num_requests(_num_requests), vocab_size(_vocab_size),
      min_input_tokens(_min_input_tokens), max_input_tokens(_max_input_tokens),
      min_tokens_to_generate(_min_tokens_to_generate),
      max_tokens_to_generate(_max_tokens_to_generate),
      poisson_distr(_poisson_distr), lambda(_lambda), timer_started(false) {
  assert(max_input_tokens >= min_input_tokens);
  assert(max_tokens_to_generate >= min_tokens_to_generate);
  assert(max_input_tokens + max_tokens_to_generate <= MAX_SEQ_LEN);
  generate_requests_meta();
};

// generate each request's arrival time and sequence length
void DataGenerator::generate_requests_meta() {
  random_device rnd1, rnd2, rnd3;
  mt19937 gen1(rnd1()), gen2(rnd2()), gen3(rnd3());
  // set up a uniform number generator with range [0,1) (in seconds) for the
  // arrival times
  uniform_real_distribution<double> dist1{0, 1.0};
  double cur_arrival = 0; // assume first request comes in at time 0
  // set up a uniform number generator for the initial/generated sequence length
  uniform_int_distribution<unsigned long> dist2{min_input_tokens,
                                                max_input_tokens};
  uniform_int_distribution<unsigned long> dist3{min_tokens_to_generate,
                                                max_tokens_to_generate};
  size_t cur_seq_len = dist2(gen2);
  size_t tokens_to_generate = dist3(gen3);

  for (size_t i = 0; i < num_requests; i++) {
    arrivals.push_back(cur_arrival);
    if (poisson_distr) {
      double u = dist1(gen1);
      double interval = -(1 / lambda) * log(1 - u) * 1000;
      cur_arrival += interval;
    } else {
      cur_arrival += (1000 / lambda);
    }
    seq_lengths.push_back(std::make_pair(cur_seq_len, tokens_to_generate));
    cur_seq_len = dist2(gen2);
    tokens_to_generate = dist3(gen3);
  }
  // cout << "Arrivals : [";
  // copy(arrivals.begin(), arrivals.end(), ostream_iterator<int>(cout, " "));
  // cout << "]" << endl;
};

void DataGenerator::generate_requests(int *req_ptr) {
  assert(req_ptr != nullptr);
  /* for (size_t i=0; i<num_requests; i++) {
    for (size_t j=0; j<max_sequence_length; j++) {
      for (size_t k=0; k<token_dim; k++) {
        req_ptr[i * max_sequence_length + j] = (float)std::rand()/RAND_MAX;
      }
    }
  } */
  // faster generation assuming req_ptr points to a tensor with contiguous
  // memory of size token_dim * max_input_tokens * num_requests, enough to
  // contain all requests data
  random_device rnd_device;
  mt19937 mersenne_engine{rnd_device()};

  // uniform_real_distribution<float> float_dist{0, 1.0};
  //  auto gen = [&float_dist, &mersenne_engine]() {
  //    return float_dist(mersenne_engine);
  //  };
  std::uniform_int_distribution<int> int_dist(0, vocab_size - 1);
  auto gen = [&int_dist, &mersenne_engine]() {
    return int_dist(mersenne_engine);
  };
  std::generate(req_ptr, req_ptr + max_input_tokens * num_requests, gen);
};

bool DataGenerator::load_requests(int *req_ptr, std::string fpath1, std::string fpath2, std::string fpath3) {
  // Open input files
  std::ifstream inFile1(fpath1);
  std::ifstream inFile2(fpath2);
  std::ifstream inFile3(fpath3);

  std::cerr << "Loading requests from " << fpath1 << ", " << fpath2 << ", " << fpath3 << std::endl;
  if (!inFile1.is_open()) {
    std::cerr << "Error: Failed to open the request token file: " << fpath1 << std::endl;
  }
  if (!inFile2.is_open()) {
    std::cerr << "Error: Failed to open the token generation file: " << fpath2 << std::endl;
  }
  if (!inFile3.is_open()) {
    std::cerr << "Error: Failed to open the arrival info file: " << fpath3 << std::endl;
  }

  if (inFile1.is_open() && inFile2.is_open() && inFile3.is_open()) {
    // Read arrival times from file3
    std::vector<double> arrival_times;
    double time;
    while (inFile3 >> time) {
      arrival_times.push_back(time);
    }
    if (arrival_times.size() != num_requests) {
      std::cerr << "Error: The number of arrival times in the input file does not match the number of requests." << std::endl;
      return false;
    }

    // Read requests and tokens_to_generate from file1 and file2
    int counter = 0;
    for (auto const &pair : seq_lengths) {
      size_t initial_length = pair.first;
      size_t tokens_to_generate;
      if (!(inFile2 >> tokens_to_generate)) {
        std::cerr << "Error: Failed to read tokens_to_generate from the input file." << std::endl;
        return false;
      }
      if (tokens_to_generate != pair.second) {
        std::cerr << "Error: The number of tokens_to_generate in the input file does not match the expected value." << std::endl;
        return false;
      }
      for (int i = 0; i < initial_length; i++) {
        if (!(inFile1 >> req_ptr[counter * max_input_tokens + i])) {
          std::cerr << "Error: Failed to read a request from the input file." << std::endl;
          return false;
        }
      }
      counter += 1;
    }

    // Fill the remaining memory with zeros
    std::fill(req_ptr + counter * max_input_tokens, req_ptr + num_requests * max_input_tokens, 0);

    // Copy arrival times to member variable
    arrivals = std::move(arrival_times);

    std::cout << "Data loaded from input files successfully." << std::endl;
    return true;
  } else {
    std::cerr << "Error: Failed to open one or more input files." << std::endl;
    return false;
  }
}


void DataGenerator::start_timer(void) {
  arrivals_ptr = arrivals.begin();
  start_time = Clock::now();
  timer_started = true;
};

// In non-incremental mode, the number of requests we want is limited by the
// tensor's batch size. As long as each request has a length that is shorter
// than the tensor's max sequence length, we do not need to impose any
// additional requirement on the max number of tokens across requests. We can
// thus pass max_tokens = max_requests * tensor max sequence length as a
// placeholder. In incremental mode, the max number of requests is only limited
// by the BatchConfig request capacity (for storing each request's metadata),
// whereas the total number number of tokens across requests will be limited by
// the tensor's batch_size * sequence length.
std::pair<size_t, size_t> DataGenerator::get_requests(size_t max_requests,
                                                      size_t max_tokens) {
  // printf("\nget_requests(%lu, %lu)\n\n", max_requests, max_tokens);
  if (!timer_started) {
    std::cout << "Warning: tried to get number of requests before the timer "
                 "was started."
              << std::endl;
    return std::make_pair(0, 0);
  }
  Clock::time_point cur_time = Clock::now();
  size_t ms_from_start =
      chrono::duration_cast<milliseconds>(cur_time - start_time).count();
  std::vector<double>::iterator new_arrivals_ptr =
      upper_bound(arrivals_ptr, arrivals.end(), ms_from_start);
  // number of new requests received
  size_t received_requests = 0;
  // id of first received request
  size_t first_request_guid = arrivals_ptr - arrivals.begin();
  size_t new_tokens = 0;
  for (size_t j = 0;
       j < std::min((size_t)(new_arrivals_ptr - arrivals_ptr), max_requests) &&
       new_tokens < max_tokens;
       j++) {
    if (seq_lengths[first_request_guid + j].first <= max_tokens - new_tokens) {
      received_requests++;
      new_tokens += seq_lengths[first_request_guid + j].first;
    } else {
      break;
    }
  }
  std::advance(arrivals_ptr, received_requests);

  /* if (received_requests > 0) {
    std::cout << "received " << received_requests
              << " request(s) by arrival time +" << ms_from_start << "ms"
              << "\n";
  } */

  return std::make_pair(first_request_guid, received_requests);
}

std::pair<size_t, size_t> DataGenerator::get_request_length(size_t guid) {
  assert(seq_lengths.size() >
         guid); // make sure the guid is valid (seq_lengths has an entry for the
                // sequence with given guid)
  return seq_lengths[guid];
}

bool DataGenerator::print_arrival_times(std::string fpath) {
  std::ofstream outFile(fpath); // Create an output file stream
  if (outFile.is_open()) {      // Check if the file is opened successfully
    for (double const &time : arrivals) {
      outFile
          << time
          << std::endl; // Write each entry to the file followed by a newline
    }
    outFile.close(); // Close the file
    std::cout << "Data written to " << fpath << " successfully." << std::endl;
    return true;
  } else {
    std::cerr << "Unable to open " << fpath << " for writing." << std::endl;
    return false; // Return with an error code
  }
}

bool DataGenerator::print_all_requests(int *requests,
                                       int max_sequence_length,
                                       std::string fpath1,
                                       std::string fpath2) {
  std::ofstream outFile1(fpath1); // Create an output file stream
  std::ofstream outFile2(fpath2); // Create an output file stream
  if (outFile1.is_open() &&
      outFile2.is_open()) { // Check if the file is opened successfully

    int counter = 0;
    for (auto const &pair : seq_lengths) {
      size_t initial_length = pair.first;
      size_t tokens_to_generate = pair.second;
      for (int i = 0; i < initial_length; i++) {
        outFile1 << requests[counter * max_input_tokens + i] << " ";
      }
      outFile1 << std::endl;
      outFile2 << tokens_to_generate << std::endl;
      counter += 1;
    }
    outFile1.close(); // Close the file
    outFile2.close(); // Close the file
    std::cout << "Data written to " << fpath1 << " and " << fpath2
              << " successfully." << std::endl;
    return true;
  } else {
    std::cerr << "Unable to open " << fpath1 << " or " << fpath2
              << " for writing." << std::endl;
    return false; // Return with an error code
  }
}