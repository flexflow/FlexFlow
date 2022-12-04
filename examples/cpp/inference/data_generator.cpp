//
//  main.cpp
//  dataloader
//
//  Created by User on 11/15/22.
//

#include "data_generator.h"
#include <chrono>
#include <ctime>
#include <iostream>
#include <math.h>
#include <random>
#include <thread>
#include <unistd.h>
using namespace std;

// This is for running the dataloader standalone
int main(int argc, char const *argv[]) {
  // insert code here...
  cout << "Hello, World!\n";
  Generator data_generator(10, 4, true, 1);

  vector<vector<double>> req0 = data_generator.get_requests();
  print_requests(req0);

  this_thread::sleep_for(milliseconds(1200));
  vector<vector<double>> req1200 = data_generator.get_requests();
  print_requests(req1200);

  this_thread::sleep_for(milliseconds(10));
  vector<vector<double>> req1210 = data_generator.get_requests();
  print_requests(req1210);

  this_thread::sleep_for(milliseconds(4000));
  vector<vector<double>> req5210 = data_generator.get_requests();
  print_requests(req5210);

  return 0;
}
