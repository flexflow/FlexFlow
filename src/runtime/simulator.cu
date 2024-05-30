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

#include "flexflow/model.h"
#include "flexflow/ops/batch_norm.h"
#include "flexflow/ops/element_unary.h"
#include "flexflow/ops/kernels/batch_matmul_kernels.h"
#include "flexflow/ops/kernels/concat_kernels.h"
#include "flexflow/ops/kernels/conv_2d_kernels.h"
#include "flexflow/ops/kernels/element_binary_kernels.h"
#include "flexflow/ops/kernels/embedding_kernels.h"
#include "flexflow/ops/kernels/linear_kernels.h"
#include "flexflow/ops/kernels/pool_2d_kernels.h"
#include "flexflow/ops/kernels/transpose_kernels.h"
#include "flexflow/ops/linear.h"
#include "flexflow/simulator.h"
#include "flexflow/utils/cuda_helper.h"

namespace FlexFlow {
// declare Legion names
using Legion::ArgumentMap;
using Legion::Context;
using Legion::coord_t;
using Legion::Domain;
using Legion::FutureMap;
using Legion::IndexLauncher;
using Legion::InlineLauncher;
using Legion::Machine;
using Legion::Memory;
using Legion::PhysicalRegion;
using Legion::Predicate;
using Legion::Rect;
using Legion::RegionRequirement;
using Legion::Runtime;
using Legion::Task;
using Legion::TaskArgument;
using Legion::TaskLauncher;

typedef Realm::Point<1, coord_t> Point1;
typedef Realm::Rect<1, coord_t> Rect1;

Simulator::Simulator(FFModel const *model,
                     FFHandler _handler,
                     Memory _memory,
                     MachineModel *machine)
    : memory(_memory), handler(_handler), offset(0), warmup_times(5),
      repeat_times(10), computationMode(model->config.computationMode) {
  // Allocate simulator memory
  Rect1 bounds(Point1(0), Point1(0));
  std::vector<size_t> field_sizes;
  field_sizes.push_back(model->config.simulator_work_space_size);
  Realm::RegionInstance::create_instance(simulatorInst,
                                         memory,
                                         bounds,
                                         field_sizes,
                                         0,
                                         Realm::ProfilingRequestSet())
      .wait();
  base_ptr = (char *)simulatorInst.pointer_untyped(0, sizeof(char));
  capacity = model->config.simulator_work_space_size;

  // Set cublas/cudnn streams to allow Realm catch the events
  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  checkCUDA(cublasSetStream(handler.blas, stream));
  checkCUDNN(cudnnSetStream(handler.dnn, stream));

  size_t max_num_tasks = 1024 * 1024;

  cudaEventCreate(&start_event);
  cudaEventCreate(&end_event);
  conv2d_meta = new Conv2DMeta(handler);
  // linear_meta = new LinearMeta(handler, 4096);
  pool2d_meta = new Pool2DMeta(handler);
  ele_unary_meta = new ElementUnaryMeta(handler);
  // ele_binary_meta = new ElementBinaryMeta(handler);
  // embedding_meta = new EmbeddingMeta(handler);
  // softmax_meta = new SoftmaxMeta(handler);
  batch_matmul_meta = new BatchMatmulMeta(handler);
  concat_meta = new ConcatMeta(handler);
  // dropout_meta = new DropoutMeta(handler);
  transpose_meta = new TransposeMeta(handler);
  this->machine = machine;
  segment_size = model->config.simulator_segment_size;
  max_num_segments = model->config.simulator_max_num_segments;
  // Initialize task manager
  task_manager = new TaskManager(max_num_tasks);
}

Simulator::~Simulator(void) {
  simulatorInst.destroy();
  cudaEventDestroy(start_event);
  cudaEventDestroy(end_event);
  delete conv2d_meta;
  delete pool2d_meta;
  delete ele_unary_meta;
  delete batch_matmul_meta;
  delete concat_meta;
  delete transpose_meta;
  delete task_manager;
}

__host__ void
    Simulator::strategy_search_task(Task const *task,
                                    std::vector<PhysicalRegion> const &regions,
                                    Context ctx,
                                    Runtime *runtime) {
  // This method should no longer be used
  assert(false);
}

}; // namespace FlexFlow
