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
  size_t total_requests = 256;
  size_t token_dim = 16;
  size_t sequence_length = 20;
  bool use_poisson_distr = true;
  // average number of request arrivals per second
  double lambda = 25;
  int label_dims = 10;

  float *requests = (float *)calloc(
      token_dim * sequence_length * total_requests, sizeof(float));
  int *labels = (int *)calloc(sequence_length * total_requests, sizeof(int));

  DataGenerator data_generator(
      total_requests, token_dim, sequence_length, use_poisson_distr, lambda);
  data_generator.generate_requests(requests, labels, label_dims);
  data_generator.start_timer();

  size_t received_requests = data_generator.get_requests();
  std::cout << "t=0ms: received " << received_requests << std::endl;

  this_thread::sleep_for(milliseconds(1200));
  received_requests = data_generator.get_requests();
  std::cout << "t=1200ms: received " << received_requests << std::endl;

  this_thread::sleep_for(milliseconds(10));
  received_requests = data_generator.get_requests();
  std::cout << "t=1210ms: received " << received_requests << std::endl;

  this_thread::sleep_for(milliseconds(4000));
  received_requests = data_generator.get_requests();
  std::cout << "t=5210ms: received " << received_requests << std::endl;
  this_thread::sleep_for(milliseconds(5000));
  received_requests = data_generator.get_requests();
  std::cout << "t=10210ms: received " << received_requests << std::endl;

  free(requests);
  free(labels);

  return 0;
}
