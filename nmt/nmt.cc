/* Copyright 2018 Stanford, NVIDIA
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

#include <cstdio>
#include "rnn.h"
#include "rnn_mapper.h"

using namespace Legion;

LegionRuntime::Logger::Category log_nmt("nmt");

void parse_input_args(char **argv, int argc,
                      int &batch_size, int &num_layers, int &seq_length,
                      int &hidden_size, int &embed_size);

void top_level_task(const Task *task, const std::vector<PhysicalRegion> &regions,
                    Context ctx, Runtime *runtime)
{
  int batch_size = 128;
  int num_layers = 1;
  int seq_length = 1;
  int hidden_size = 1024;
  int embed_size = 1024;
  int num_workers = 1;
  int num_parts = 1;
  int num_iterations = 10;
  {
    const InputArgs &command_args = HighLevelRuntime::get_input_args();
    char **argv = command_args.argv;
    int argc = command_args.argc;
    parse_input_args(argv, argc, batch_size, num_layers, seq_length,
                     hidden_size, embed_size);
  }
  RnnModel model(batch_size, num_layers, seq_length, hidden_size, embed_size, num_parts, num_workers, ctx, runtime);
  ArgumentMap local_args;
  size_t workSpaceSize = (size_t) 2 * 1024 * 1024 * 1024;
  Rect<1> workers_rect(Point<1>(0), Point<1>(num_workers-1));
  IndexSpaceT<1> workers_is = runtime->create_index_space(ctx, workers_rect);
  IndexLauncher init_launcher(CUDNN_INIT_TASK_ID, workers_is,
                    TaskArgument(&workSpaceSize, sizeof(workSpaceSize)), local_args);
  FutureMap fm = runtime->execute_index_space(ctx, init_launcher);
  fm.wait_all_results();
  int idx = 0;
  for (PointInRectIterator<1> it(workers_rect); it(); it++)
    model.dnn_handlers[idx++] = fm.get_result<DnnHandle>(*it);

  model.init();
  for (int i = 0; i < num_iterations; i++) {
    model.forward();
  }
}

int main(int argc, char **argv)
{
  Runtime::set_top_level_task_id(TOP_LEVEL_TASK_ID);
  {
    TaskVariantRegistrar registrar(TOP_LEVEL_TASK_ID, "top_level");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<top_level_task>(registrar, "top_level");
  }

  // DNN_INIT_TASK
  {
    TaskVariantRegistrar registrar(CUDNN_INIT_TASK_ID, "cudnn_init_task");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<DnnHandle, init_cudnn>(registrar, "cudnn_init_task");
  }
  // Pooling2D task
  {
    TaskVariantRegistrar registrar(LSTM_INIT_TASK_ID, "lstm_init_task");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<OpMeta*, LSTM::init_task>(registrar, "lstm_init_task");
  }
  {
    TaskVariantRegistrar registrar(LSTM_FWD_TASK_ID, "lstm_fwd_task");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<LSTM::forward_task>(registrar, "lstm_fwd_task");
  }
  {
    TaskVariantRegistrar registrar(LSTM_BWD_TASK_ID, "lstm_bwd_task");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<LSTM::backward_task>(registrar, "lstm_bwd_task");
  }
  
  Runtime::add_registration_callback(update_mappers);
  return Runtime::start(argc, argv);
}

void parse_input_args(char **argv, int argc,
                      int &batch_size, int &num_layers, int &seq_length,
                      int &hidden_size, int &embed_size)
{
  for (int i = 1; i < argc; i++)
  {
    if (!strcmp(argv[i], "-b"))
    {
      batch_size = atoi(argv[++i]);
      continue;
    }
    if (!strcmp(argv[i], "-l"))
    {
      num_layers = atoi(argv[++i]);
      continue;
    }
    if (!strcmp(argv[i], "-s"))
    {
      seq_length = atoi(argv[++i]);
      continue;
    }
    if (!strcmp(argv[i], "-h"))
    {
      hidden_size = atoi(argv[++i]);
      continue;
    }
    if (!strcmp(argv[i], "-e"))
    {
      embed_size = atoi(argv[++i]);
      continue;
    }
  }
}

