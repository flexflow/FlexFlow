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
                size_t _token_dim,
                size_t _sequence_length,
                bool _poisson_distr,
                double _lambda);

  // Generate random requests by filling each token with random data. For now,
  // assume all requests have the same sequence length. Also generate random
  // labels (if label_ptr != nullptr and num_labels >0).
  void generate_requests(float *req_ptr,
                         int *label_ptr = nullptr,
                         int num_labels = 0);
  void start_timer(void);
  // Get number of requests that have arrived since the last time this function
  // was called
  std::pair<size_t, size_t> DataGenerator::get_requests(size_t max_num_requests)

      private :
      // Compute the arrival times of each request and save them in the arrivals
      // vector.
      // void generate_arrival_times(void);
      void generate_requests_meta();

  size_t num_requests;     // total number of requests
  size_t token_dim;        // embedding dim of each token
  size_t sequence_length;  // dimension of one request tensor
  bool poisson_distr;      // false implies uniform distribution
  double lambda;           // mean #num of arrivals per sec
  bool timer_started;      // whether timer was initiated
  size_t global_unique_id; // guid for requests
  // time when get_requests() is called for the first time
  Clock::time_point start_time;
  // arrival times (ms) generated based on distribution
  vector<double> arrivals;
  vector<double>::iterator arrivals_ptr;
  // sequence lengths generated based on uniform distribution
  vector<size_t> cur_seq_len;
};
