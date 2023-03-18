//
//  main.cpp
//  dataloader
//
//  Created by User on 11/15/22.
//

#include "data_generator.h"
#include <ctime>
#include <iostream>
#include <random>
#include <unistd.h>
using namespace std;

// This is for testing the request generator standalone
int main(int argc, char const *argv[]) {

  cout << "Starting the Data DataGenerator!\n";

  // DataGenerator parameters
  size_t total_requests = 2560;
  size_t token_dim = 16;
  size_t max_sequence_length = 512 + 128;
  bool use_poisson_distr = true;
  // average number of request arrivals per second
  double lambda = 250;

  size_t min_input_tokens = 32, max_input_tokens = 512,
         min_tokens_to_generate = 1, max_tokens_to_generate = 128;

  float *requests = (float *)calloc(
      token_dim * max_sequence_length * total_requests, sizeof(float));

  DataGenerator data_generator(total_requests,
                               token_dim,
                               min_input_tokens,
                               max_input_tokens,
                               min_tokens_to_generate,
                               max_tokens_to_generate,
                               use_poisson_distr,
                               lambda);
  data_generator.generate_requests(requests);
  data_generator.start_timer();

  size_t received_requests = 0;
  std::pair<size_t, size_t> reqs = data_generator.get_requests(0, 0);
  size_t guid = reqs.first;
  assert(reqs.second == 0);
  this_thread::sleep_for(milliseconds(50));

  reqs = data_generator.get_requests(2560, 2560 * (512));
  received_requests += reqs.second;
  std::cout << "t=0ms: received " << received_requests << std::endl;

  this_thread::sleep_for(milliseconds(1200));
  reqs = data_generator.get_requests(2560, 2560 * (512));
  received_requests += reqs.second;
  std::cout << "t=1200ms: received " << received_requests << std::endl;

  this_thread::sleep_for(milliseconds(10));
  reqs = data_generator.get_requests(2560, 2560 * (512));
  received_requests += reqs.second;
  std::cout << "t=1210ms: received " << received_requests << std::endl;

  this_thread::sleep_for(milliseconds(4000));
  reqs = data_generator.get_requests(2560, 2560 * (512));
  received_requests += reqs.second;
  std::cout << "t=5210ms: received " << received_requests << std::endl;
  this_thread::sleep_for(milliseconds(5000));

  reqs = data_generator.get_requests(2560, 2560 * (512));
  received_requests += reqs.second;
  std::cout << "t=10210ms: received " << received_requests << std::endl;

  free(requests);

  assert(received_requests == total_requests);

  return 0;
}
