/* Copyright 2022 CMU, Stanford
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


#include "flexflow/model.h"
#include "flexflow/inference.h"
using namespace Legion;
using namespace std;
using namespace FlexFlow;

#define MAX_NUM_SAMPLES 1024000

struct MLPConfig {
  MLPConfig(void);
  MLPConfig(int embedding_size,
            int sequence_length,
            std::vector<int> hidden_dims)
      : embedding_size(embedding_size), sequence_length(sequence_length),
        hidden_dims(hidden_dims) {}

  int embedding_size, sequence_length;
  std::vector<int> hidden_dims;
};

class DataLoader {
public:
  DataLoader(FFModel &ff,
             MLPConfig const &mlpConfig,
             InferenceManager const *im,
             Tensor input);
  /*static void load_input(Task const *task,
                         std::vector<PhysicalRegion> const &regions,
                         Context ctx,
                         Runtime *runtime);*/
  static void load_entire_dataset(Task const *task,
                                  std::vector<PhysicalRegion> const &regions,
                                  Context ctx,
                                  Runtime *runtime);
  void next_batch(FFModel &);
  void reset(void);

public:
  int num_samples, next_index;
  Tensor full_input, batch_input;
};

struct SampleIdxs {
  int num_samples;
  int idxs[MAX_NUM_SAMPLES];
};