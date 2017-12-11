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
#include "ops.h"
#include "cnn_mapper.h"
#include "inception.h"
#define USE_VGG

using namespace Legion;

LegionRuntime::Logger::Category log_cnn("cnn");

void parse_input_args(char **argv, int argc,
                      int &num_par_h, int &num_par_w, int& num_par_n,
                      int &batch_size, int &fc_num_par_c, int &fc_num_par_n);

void top_level_task(const Task *task, const std::vector<PhysicalRegion> &regions,
                    Context ctx, Runtime *runtime)
{
  // Set up config parameters
  int num_par_h = 1;
  int num_par_w = 1;
  int num_par_n = 4;
  int num_images = 128; // per_batch
  int fc_num_par_c = 4;
  int fc_num_par_n = 1;
  int height = 224;
  int width = 224;
  bool profiling = false;
  float learning_rate = 0.01;
  int num_iterations = 5;
  // parse input arguments
  {
    const InputArgs &command_args = HighLevelRuntime::get_input_args();
    char **argv = command_args.argv;
    int argc = command_args.argc;
    parse_input_args(argv, argc, num_par_h, num_par_w, num_par_n,
                     num_images, fc_num_par_c, fc_num_par_n);
    printf("batch_size(%d) par_h(%d) par_w(%d) par_n(%d)\n",
           num_images, num_par_h, num_par_w, num_par_n);
    printf("par_fc_c(%d) par_fc_n(%d)\n", fc_num_par_c, fc_num_par_n);
  }
  //assert(num_par_h * num_par_w * num_par_n == fc_num_par_c * fc_num_par_n);
  CnnModel model(num_images, height, width, num_par_n, num_par_h, num_par_w,
                 fc_num_par_n, fc_num_par_c, profiling, learning_rate, ctx, runtime);
  // First, create cnnContexts
  ArgumentMap local_args;
  size_t workSpaceSize = (size_t) 2 * 1024 * 1024 * 1024;
  IndexLauncher init_launcher(CNN_INIT_TASK_ID, model.part_is,
                              TaskArgument(&workSpaceSize, sizeof(workSpaceSize)), local_args);
  FutureMap fm = runtime->execute_index_space(ctx, init_launcher);
  fm.wait_all_results();
  Rect<3> rect = runtime->get_index_space_domain(ctx, model.part_is);
  int idx = 0;
  for (PointInRectIterator<3> it(rect); it(); it++) {
    model.cnn_handlers[idx++] = fm.get_result<CnnHandle>(*it);
  }

  // Construct model (AlexNet)
#ifdef USE_ALEXNET
  Tensor t = model.add_conv_layer(model.input_image, 64, 11, 11, 4, 4, 2, 2);
  t = model.add_pool_layer(t, 3, 3, 2, 2, 0, 0);
  t = model.add_conv_layer(t, 192, 5, 5, 1, 1, 2, 2);
  t = model.add_pool_layer(t, 3, 3, 2, 2, 0, 0);
  t = model.add_conv_layer(t, 384, 3, 3, 1, 1, 1, 1);
  t = model.add_conv_layer(t, 256, 3, 3, 1, 1, 1, 1);
  t = model.add_conv_layer(t, 256, 3, 3, 1, 1, 1, 1);
  t = model.add_pool_layer(t, 3, 3, 2, 2, 0, 0);
  t = model.add_flat_layer(t);
  t = model.add_linear_layer(t, 4096);
  t = model.add_linear_layer(t, 4096);
  t = model.add_linear_layer(t, 1000, false/*relu*/);
  t = model.add_softmax_layer(t);
#endif

  // Construct model (VGG-16Net)
#ifdef USE_VGG
  Tensor t = model.add_conv_layer(model.input_image, 64, 3, 3, 1, 1, 1, 1);
  t = model.add_conv_layer(t, 64, 3, 3, 1, 1, 1, 1);
  t = model.add_pool_layer(t, 2, 2, 2, 2, 0, 0);
  t = model.add_conv_layer(t, 128, 3, 3, 1, 1, 1, 1);
  t = model.add_conv_layer(t, 128, 3, 3, 1, 1, 1, 1);
  t = model.add_pool_layer(t, 2, 2, 2, 2, 0, 0);
  t = model.add_conv_layer(t, 256, 3, 3, 1, 1, 1, 1);
  t = model.add_conv_layer(t, 256, 3, 3, 1, 1, 1, 1);
  t = model.add_conv_layer(t, 256, 3, 3, 1, 1, 1, 1);
  t = model.add_pool_layer(t, 2, 2, 2, 2, 0, 0);
  t = model.add_conv_layer(t, 512, 3, 3, 1, 1, 1, 1);
  t = model.add_conv_layer(t, 512, 3, 3, 1, 1, 1, 1);
  t = model.add_conv_layer(t, 512, 3, 3, 1, 1, 1, 1);
  t = model.add_pool_layer(t, 2, 2, 2, 2, 0, 0);
  t = model.add_conv_layer(t, 512, 3, 3, 1, 1, 1, 1);
  t = model.add_conv_layer(t, 512, 3, 3, 1, 1, 1, 1);
  t = model.add_conv_layer(t, 512, 3, 3, 1, 1, 1, 1);
  t = model.add_pool_layer(t, 2, 2, 2, 2, 0, 0);
  t = model.add_flat_layer(t);
  t = model.add_linear_layer(t, 4096);
  t = model.add_linear_layer(t, 4096);
  t = model.add_linear_layer(t, 1000, false/*relu*/);
  t = model.add_softmax_layer(t);
#endif

  // Construct model (Inception-V3)
#ifdef USE_INCEPTION
  Tensor t = model.add_conv_layer(model.input_image, 32, 3, 3, 2, 2, 0, 0);
  t = model.add_conv_layer(t, 32, 3, 3, 1, 1, 0, 0);
  t = model.add_conv_layer(t, 64, 3, 3, 1, 1, 1, 1);
  t = model.add_pool_layer(t, 3, 3, 2, 2, 0, 0);
  t = model.add_conv_layer(t, 80, 1, 1, 1, 1, 0, 0);
  t = model.add_conv_layer(t, 192, 3, 3, 1, 1, 1, 1);
  t = model.add_pool_layer(t, 3, 3, 2, 2, 0, 0);
  t = InceptionA(model, t, 32);
  t = InceptionA(model, t, 64);
  t = InceptionA(model, t, 64);
  t = InceptionB(model, t);
  t = InceptionC(model, t, 128);
  t = InceptionC(model, t, 160);
  t = InceptionC(model, t, 160);
  t = InceptionC(model, t, 192);
  t = InceptionD(model, t);
  t = InceptionE(model, t);
  t = InceptionE(model, t);
  t = model.add_pool_layer(t, 8, 8, 1, 1, 0, 0, POOL2D_AVG);
  t = model.add_flat_layer(t);
  t = model.add_linear_layer(t, 1000, false/*relu*/);
  t = model.add_softmax_layer(t);
#endif
  
  // Initialize every layer
  model.init_layers();

  double ts_start = Realm::Clock::current_time_in_microseconds();
  for (int i = 0; i < num_iterations; i++) {
    model.forward();

    model.backward();

    model.update();
  }
  runtime->issue_execution_fence(ctx);
  TimingLauncher timer(MEASURE_MICRO_SECONDS);
  Future future = runtime->issue_timing_measurement(ctx, timer);
  future.get_void_result();
  double ts_end = Realm::Clock::current_time_in_microseconds();
  double run_time = 1e-6 * (ts_end - ts_start);
  printf("time = %.4fs, tp = %.2f images/s", run_time, num_images * num_iterations / run_time);
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
    Runtime::preregister_task_variant<CnnHandle, init_cudnn>(registrar, "cnn_init_task");
  }

  // IMAGE_INIT_TASK
  {
    TaskVariantRegistrar registrar(IMAGE_INIT_TASK_ID, "image_init_task");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<CnnModel::init_images_task>(registrar, "image_init_task");
  }

  // LABEL_INIT_TASK
  {
    TaskVariantRegistrar registrar(LABEL_INIT_TASK_ID, "label_init_task");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<CnnModel::init_labels_task>(registrar, "label_init_task");
  }

  // Conv2D task
  {
    TaskVariantRegistrar registrar(CONV2D_INIT_TASK_ID, "conv2d_init_task");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<OpMeta*, Conv2D::init_task>(registrar, "conv2d_init_task");
  }
  {
    TaskVariantRegistrar registrar(CONV2D_INIT_PARA_TASK_ID, "conv2d_init_para_task");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<Conv2D::init_para_task>(registrar, "conv2d_init_para_task");
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
  {
    TaskVariantRegistrar registrar(CONV2D_UPD_TASK_ID, "conv2d_upd_task");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<Conv2D::update_task>(registrar, "conv2d_upd_task");
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

  // Linear task
  {
    TaskVariantRegistrar registrar(LINEAR_INIT_TASK_ID, "linear_init_task");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<OpMeta*, Linear::init_task>(registrar, "linear_init_task");
  }
  {
    TaskVariantRegistrar registrar(LINEAR_INIT_PARA_TASK_ID, "linear_init_para_task");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<Linear::init_para_task>(registrar, "linear_init_para_task");
  }
  {
    TaskVariantRegistrar registrar(LINEAR_FWD_TASK_ID, "linear_fwd_task");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<Linear::forward_task>(registrar, "linear_fwd_task");
  }
  {
    TaskVariantRegistrar registrar(LINEAR_BWD_TASK_ID, "linear_bwd_task");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<Linear::backward_task>(registrar, "linear_bwd_task");
  }
  {
    TaskVariantRegistrar registrar(LINEAR_BWD2_TASK_ID, "linear_bwd_task (aggregate replica)");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<Linear::backward2_task>(registrar, "linear_bwd_task (aggregate replica)");
  }
  {
    TaskVariantRegistrar registrar(LINEAR_UPD_TASK_ID, "linear_upd_task");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<Linear::update_task>(registrar, "linear_upd_task");
  }

  // Flat task
  {
    TaskVariantRegistrar registrar(FLAT_INIT_TASK_ID, "flat_init_task");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<OpMeta*, Flat::init_task>(registrar, "flat_init_task");
  }
  {
    TaskVariantRegistrar registrar(FLAT_FWD_TASK_ID, "flat_fwd_task");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<Flat::forward_task>(registrar, "flat_fwd_task");
  }
  {
    TaskVariantRegistrar registrar(FLAT_BWD_TASK_ID, "flat_bwd_task");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<Flat::backward_task>(registrar, "flat_bwd_task");
  }

  // Softmax task
  {
    TaskVariantRegistrar registrar(SOFTMAX_INIT_TASK_ID, "softmax_init_task");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<OpMeta*, Softmax::init_task>(registrar, "softmax_init_task");
  }
  {
    TaskVariantRegistrar registrar(SOFTMAX_FWD_TASK_ID, "softmax_fwd_task");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<Softmax::forward_task>(registrar, "softmax_fwd_task");
  }
  {
    TaskVariantRegistrar registrar(SOFTMAX_BWD_TASK_ID, "softmax_bwd_task");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<Softmax::backward_task>(registrar, "softmax_bwd_task");
  }

  // Concat task
  {
    TaskVariantRegistrar registrar(CONCAT_INIT_TASK_ID, "concat_init_task");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<OpMeta*, Concat::init_task>(registrar, "concat_init_task");
  }
  {
    TaskVariantRegistrar registrar(CONCAT_FWD_TASK_ID, "concat_fwd_task");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<Concat::forward_task>(registrar, "concat_fwd_task");
  }
  {
    TaskVariantRegistrar registrar(CONCAT_BWD_TASK_ID, "concat_bwd_task");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<Concat::backward_task>(registrar, "concat_bwd_task");
  }

  Runtime::add_registration_callback(update_mappers);
  return Runtime::start(argc, argv);
}

void parse_input_args(char **argv, int argc,
                      int &num_par_h, int &num_par_w, int& num_par_n,
                      int &batch_size, int &fc_num_par_c, int &fc_num_par_n)
{
  for (int i = 1; i < argc; i++)
  {
    if (!strcmp(argv[i], "-ph"))
    {
      num_par_h = atoi(argv[++i]);
      continue;
    }
    if (!strcmp(argv[i], "-pw"))
    {
      num_par_w = atoi(argv[++i]);
      continue;
    }
    if (!strcmp(argv[i], "-pn"))
    {
      num_par_n = atoi(argv[++i]);
      continue;
    }
    if (!strcmp(argv[i], "-pfc"))
    {
      fc_num_par_c = atoi(argv[++i]);
      continue;
    }
    if (!strcmp(argv[i], "-pfn"))
    {
      fc_num_par_n = atoi(argv[++i]);
      continue;
    }
    if (!strcmp(argv[i], "-b"))
    {
      batch_size = atoi(argv[++i]);
      continue;
    }
  }
}
