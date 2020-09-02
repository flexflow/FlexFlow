/* Copyright 2020 Stanford
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

#include "config.h"
#include "cuda_helper.h"
#include "model.h"
#include "simulator.h"

using namespace Legion;

LegionRuntime::Logger::Category log_app("AlexNetSim");

//int main(int argc, char * argv[]) {
void top_level_task(const Task* task,
                    const std::vector<PhysicalRegion>& regions,
                    Context ctx, Runtime* runtime) {
  FFConfig ffConfig;
  {
    const InputArgs &command_args = HighLevelRuntime::get_input_args();
    char **argv = command_args.argv;
    int argc = command_args.argc;
    ffConfig.parse_args(argv, argc);
    log_app.print("batchSize(%d) workersPerNodes(%d) numNodes(%d)",
        ffConfig.batchSize, ffConfig.workersPerNode, ffConfig.numNodes);
  }
  ffConfig.lg_ctx = ctx;
  ffConfig.lg_hlr = runtime;
  ffConfig.field_space = runtime->create_field_space(ctx);

  ffConfig.numNodes = 16;
  FFModel ff(ffConfig);
  //ffConfig.workersPerNode = 4;

  Tensor input;
  {
    const int dims[] = {ffConfig.batchSize, 3, 229, 229};
    input = ff.create_tensor<4>(dims, "", DT_FLOAT);
  }
  //Tensor label;
  //{
  //  const int dims[] = {ffConfig.batchSize, 1};
  //  label = ff.create_tensor<2>(dims, "", DT_INT32);
  //}
  // Add layers
  Tensor t = input, ts[2];
  t = ff.conv2d(input, 64, 11, 11, 4, 4, 2, 2, AC_MODE_RELU);
  //ts[1] = ff.conv2d("conv1", input, 64, 11, 11, 4, 4, 2, 2);
  //t = ff.concat("concat", 2, ts, 1/*axis*/);
  t = ff.pool2d(t, 3, 3, 2, 2, 0, 0);
  t = ff.conv2d(t, 192, 5, 5, 1, 1, 2, 2, AC_MODE_RELU);
  t = ff.pool2d(t, 3, 3, 2, 2, 0, 0);
  t = ff.conv2d(t, 384, 3, 3, 1, 1, 1, 1, AC_MODE_RELU);
  t = ff.conv2d(t, 256, 3, 3, 1, 1, 1, 1, AC_MODE_RELU);
  t = ff.conv2d(t, 256, 3, 3, 1, 1, 1, 1, AC_MODE_RELU);
  t = ff.pool2d(t, 3, 3, 2, 2, 0, 0);
  t = ff.flat(t);
  t = ff.dense(t, 4096, AC_MODE_RELU/*relu*/);
  t = ff.dense(t, 4096, AC_MODE_RELU/*relu*/);
  t = ff.dense(t, 10);
  t = ff.softmax(t);

  FFHandler & handler = ff.handlers[0];
  /*
  checkCUDNN(cudnnCreate(&handler.dnn));
  checkCUDA(cublasCreate(&handler.blas));
  cudaStream_t stream;
  checkCUDA(cudaStreamCreate(&stream));
  checkCUDNN(cudnnSetStream(handler.dnn, stream));
  checkCUDA(cublasSetStream(handler.blas, stream));
  handler.workSpaceSize = (size_t) 2 * 1024 * 1024 * 1024;
  checkCUDA(cudaMalloc(&handler.workSpace, handler.workSpaceSize));
  */

  Simulator sim(&ff, handler, handler.workSpace, handler.workSpaceSize);

  std::map<Op*, ParallelConfig> best;
  ff.optimize(&sim, best, 10000, 0.1);

  
}

void register_custom_tasks()
{
  /*
  // Load entire dataset
  {
    TaskVariantRegistrar registrar(CUSTOM_CPU_TASK_ID_1, "Load Entire Dataset");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<DataLoader::load_entire_dataset>(
        registrar, "Load Entire Dataset Task");
  }
  // Load input
  {
    TaskVariantRegistrar registrar(CUSTOM_GPU_TASK_ID_1, "Load Inputs");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<DataLoader::load_input>(
        registrar, "Load Input Task");
  }
  // Load label
  {
    TaskVariantRegistrar registrar(CUSTOM_GPU_TASK_ID_2, "Load Labels");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<DataLoader::load_label>(
        registrar, "Load Label Task");
  }
  */
}


