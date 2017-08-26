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

#include <cstdio>
#include "legion.h"
#include "ops.h"

using namespace Legion;

LegionRuntime::Logger::Category log_cnn("cnn");

void top_level_task(const Task *task, const std::vector<PhysicalRegion> &regions,
                    Context ctx, Runtime *runtime)
{
  // Set up config parameters
  int num_par_h = 2;
  int num_par_w = 2;
  int num_par_n = 1;
  int num_images = 32; // per_batch
  int height = 224;
  int width = 224;
  CnnModel model(num_images, height, width, num_par_n, num_par_h, num_par_w, ctx, runtime);
  int num_workers = num_par_h * num_par_w * num_par_n;
  // First, create cnnContexts
  ArgumentMap local_args;
  size_t workSpaceSize = (size_t) 2 * 1024 * 1024 * 1024;
  IndexLauncher init_launcher(CNN_INIT_TASK_ID, model.part_is,
                              TaskArgument(&workSpaceSize, sizeof(workSpaceSize)), local_args);
  FutureMap fm = runtime->execute_index_space(ctx, init_launcher);
  fm.wait_all_results();
  Realm::ZRect<3> rect = runtime->get_index_space_domain(ctx, model.part_is);
  int idx = 0;
  for (PointInRectIterator<3> it(rect); it(); it++) {
    model.cnn_handlers[idx++] = fm.get_result<CnnHandle>(*it);
  }

  // Construct model
  model.add_conv_layer(model.input_image, 64, 3, 3, 1, 1, 1, 1);

  // Initialize every layer
  model.init_layers();

  model.forward();
}

int main(int argc, char **argv)
{
  Runtime::set_top_level_task_id(TOP_LEVEL_TASK_ID);

  {
    TaskVariantRegistrar registrar(TOP_LEVEL_TASK_ID, "top_level");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<top_level_task>(registrar, "top_level");
  }

  // CNN_INIT_TASK
  {
    TaskVariantRegistrar registrar(CNN_INIT_TASK_ID, "cnn_init_task");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<CnnHandle, init_cudnn>(registrar, "init_task");
  }

  // Conv2D task
  {
    TaskVariantRegistrar registrar(CONV2D_INIT_TASK_ID, "conv2d_init_task");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<OpMeta*, Conv2D::init_task>(registrar, "init_task");
  }
  {
    TaskVariantRegistrar registrar(CONV2D_FWD_TASK_ID, "conv2d_fwd_task");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<Conv2D::forward_task>(registrar, "conv2d_fwd_task");
  }
  {
    TaskVariantRegistrar registrar(CONV2D_BWD_TASK_ID, "conv2d_bwd_task");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<Conv2D::backward_task>(registrar, "conv2d_bwd_task");
  }

  // Pooling2D task
  {
    TaskVariantRegistrar registrar(POOL2D_INIT_TASK_ID, "pooling2d_init_task");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<OpMeta*, Pooling2D::init_task>(registrar, "pooling2d_init_task");
  }
  {
    TaskVariantRegistrar registrar(POOL2D_FWD_TASK_ID, "pooling2d_fwd_task");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<Pooling2D::forward_task>(registrar, "pooling2d_fwd_task");
  }
  {
    TaskVariantRegistrar registrar(POOL2D_BWD_TASK_ID, "pooling2d_bwd_task");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<Pooling2D::backward_task>(registrar, "pooling2d_bwd_task");
  }

  return Runtime::start(argc, argv);
}
