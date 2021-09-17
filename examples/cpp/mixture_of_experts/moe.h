/* Copyright 2017 Stanford, NVIDIA
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

#include "model.h"

#define MAX_K 2
#define MAX_BATCH_SIZE 50
#define MAX_N 5
// #define MOE_VERBOSE
// #define SPEC_SCORE
// #define MOE_SPEC_SCORE

#define MAX_NUM_SAMPLES 50

// data set defines
#define USE_RANDOM
#define NUM_SAMPLES (TRAIN_SAMPLES+TEST_SAMPLES)

#ifdef USE_MNIST
  #define TRAIN_SAMPLES 60000
  #define TEST_SAMPLES 0
  #define INPUT_DIM 28*28
  #define D_DIM 2
  #define OUT_DIM 10
  #define READ_DATA read_mnist
  #define USE_MLP
#elif defined(USE_RANDOM)
  #define TRAIN_SAMPLES 50000
  #define TEST_SAMPLES 0
  #define INPUT_DIM 1024
  #define D_DIM 2
  #define OUT_DIM INPUT_DIM
  #define READ_DATA read_random
  #define USE_MLP
#elif defined(USE_CIFAR100)
  #define TRAIN_SAMPLES 50000
  #define TEST_SAMPLES 10000
  #define INPUT_DIM 3,32,32
  #define D_DIM 4
  #define OUT_DIM 100
  #define READ_DATA read_cifar100
  #define USE_CNN
#elif defined(USE_CIFAR100_C)
  #define TRAIN_SAMPLES 50000
  #define TEST_SAMPLES 0
  #define INPUT_DIM 3,32,32
  #define D_DIM 4
  #define OUT_DIM 10
  #define READ_DATA read_cifar100_c
  #define USE_CNN
#elif defined(USE_CIFAR10)
  #define TRAIN_SAMPLES 50040
  #define TEST_SAMPLES 9960
  #define INPUT_DIM 3,32,32
  #define D_DIM 4
  #define OUT_DIM 10
  #define READ_DATA read_cifar10
  #define USE_CNN
#endif


using namespace Legion;
using namespace std;

struct MoeConfig {
  MoeConfig(void) {
    // Set default configurations here
  }
  std::string dataset_path;
};


class DataLoader {
public:
  DataLoader(FFModel& ff, const MoeConfig& alexnet,
             Tensor _input, Tensor _label);
  static void load_input(const Task *task,
                         const std::vector<PhysicalRegion> &regions,
                         Context ctx,
                         Runtime* runtime);
  static void load_label(const Task *task,
                         const std::vector<PhysicalRegion> &regions,
                         Context ctx,
                         Runtime* runtime);
  static void load_entire_dataset(const Task *task,
                                  const std::vector<PhysicalRegion> &regions,
                                  Context ctx,
                                  Runtime* runtime);
  void next_batch(FFModel&);
  void reset(void);
public:
  int num_samples, next_index;
  Tensor full_input, batch_input;
  Tensor full_label, batch_label;
};

struct SampleIdxs {
  int num_samples;
  int idxs[MAX_NUM_SAMPLES];
};
