#include "data_generator.h"
#include <algorithm>
#include <iostream>
#include <vector>
using namespace std;

DataGenerator::DataGenerator(size_t _num_requests,
                             size_t _token_dim,
                             size_t _max_sequence_length,
                             bool _poisson_distr,
                             double _lambda)
    : num_requests(_num_requests), token_dim(_token_dim),
      max_sequence_length(_max_sequence_length), poisson_distr(_poisson_distr),
      lambda(_lambda), timer_started(false), global_unique_id(1000000) {
  generate_requests_meta();
};

// generate each request's arrival time and sequence length
void DataGenerator::generate_requests_meta() {
  // set up a uniform number generator with range [0,1) for the arrival times
  random_device rnd1, rnd2;
  mt19937 gen1(rnd1()), gen2(rnd2());
  uniform_real_distribution<double> dist1{0, 1.0};
  double cur_arrival = 0; // assume first request comes in at time 0
  // set up a uniform number generator for the sequence length
  uniform_int_distribution<unsigned long> dist2{1, max_sequence_length};
  size_t cur_seq_len = dist2(gen2);

  for (size_t i = 0; i < num_requests; i++) {
    arrivals.push_back(cur_arrival);
    if (poisson_distr) {
      double u = dist1(gen1);
      double interval = -(1 / lambda) * log(1 - u) * 1000;
      cur_arrival += interval;
    } else {
      cur_arrival += (1000 / lambda);
    }
    seq_lengths.push_back(cur_seq_len);
    cur_seq_len = dist2(gen2);
  }
  // cout << "Arrivals : [";
  // copy(arrivals.begin(), arrivals.end(), ostream_iterator<int>(cout, " "));
  // cout << "]" << endl;
};

void DataGenerator::generate_requests(float *req_ptr) {
  assert(req_ptr != nullptr);
  /* for (size_t i=0; i<num_requests; i++) {
    for (size_t j=0; j<max_sequence_length; j++) {
      for (size_t k=0; k<token_dim; k++) {
        req_ptr[i * max_sequence_length + j] = (float)std::rand()/RAND_MAX;
      }
    }
  } */
  random_device rnd_device;
  mt19937 mersenne_engine{rnd_device()};

  uniform_real_distribution<float> float_dist{0, 1.0};
  auto gen = [&float_dist, &mersenne_engine]() {
    return float_dist(mersenne_engine);
  };
  std::generate(
      req_ptr, req_ptr + token_dim * max_sequence_length * num_requests, gen);
};

void DataGenerator::start_timer(void) {
  arrivals_ptr = arrivals.begin();
  start_time = Clock::now();
  timer_started = true;
};

std::pair<size_t, size_t> DataGenerator::get_requests(size_t batch_capacity) {
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
  //size_t received_requests =
  //    std::min((size_t)(new_arrivals_ptr - arrivals_ptr), max_num_requests);
  // id of first received request
  size_t first_request_guid = arrivals_ptr - arrivals.begin();
  size_t new_tokens = 0;
  for (size_t j=0; j < new_arrivals_ptr - arrivals_ptr && new_tokens < batch_capacity && received_requests < BatchConfig::MAX_NUM_REQUESTS; j++) {
    received_requests++;
    new_tokens += seq_lengths[first_request_guid+j];
  }
  std::advance(arrivals_ptr, received_requests);

  /* if (received_requests > 0) {
    std::cout << "received " << received_requests
              << " request(s) by arrival time +" << ms_from_start << "ms"
              << "\n";
  } */

  return std::make_pair(first_request_guid, received_requests);
}

ssize_t DataGenerator::get_request_length(size_t guid) {
  if (seq_lengths.size() <= guid) {
    return -1;
  }
  return seq_lengths[guid];
}
