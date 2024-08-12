#include "model.h"
#include "test_utils.h"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
using namespace Legion;
Realm::Logger log_app("linear_test");

struct LinearTestMeta {
  int batch_size, i_dim, num_channels, dense_projection_o_dim,
      dense_projection_i_dim;
  LinearTestMeta(int _batch_size,
                 int _i_dim,
                 int _num_channels,
                 int _dense_projection_o_dim,
                 int _dense_projection_i_dim) {
    batch_size = _batch_size, num_channels = _num_channels, i_dim = _i_dim,
    dense_projection_o_dim = _dense_projection_o_dim,
    dense_projection_i_dim = _dense_projection_i_dim;
  }
};

LinearTestMeta get_test_meta(const std::string file_path) {
  std::fstream myfile(file_path, std::ios_base::in);
  int batch_size, i_dim, num_channels, dense_projection_o_dim,
      dense_projection_i_dim;
  myfile >> batch_size >> i_dim >> num_channels >> dense_projection_o_dim >>
      dense_projection_i_dim;
  return LinearTestMeta(batch_size,
                        i_dim,
                        num_channels,
                        dense_projection_o_dim,
                        dense_projection_i_dim);
}

void top_level_task(Task const *task,
                    std::vector<PhysicalRegion> const &regions,
                    Context ctx,
                    Runtime *runtime) {
  std::cout << "test framework launched" << std::endl;
  auto test_meta = get_test_meta("test_meta.txt");
  FFConfig ffConfig;
  // create ff model object
  FFModel ff(ffConfig);
  IndexSpace task_is = IndexSpaceT<2>(ff.get_or_create_task_is(2, ""));
  Initializer *kernel_initializer = new ZeroInitializer();
  Initializer *bias_initializer = new ZeroInitializer();
  Tensor weights;
  {
    int const dims[2] = {test_meta.dense_projection_o_dim,
                         test_meta.dense_projection_i_dim};
    weights = ff.create_linear_weight<2>(
        dims, (IndexSpaceT<2>)task_is, DT_FLOAT, kernel_initializer);
    auto weights_file_path = "test_kernel1.txt";
    initialize_tensor_from_file(weights_file_path, weights, ff, "float", 2);
  }
  Tensor bias;
  {
    int const dims[1] = {test_meta.dense_projection_o_dim};
    bias = ff.create_linear_weight<1>(
        dims, (IndexSpaceT<2>)task_is, DT_FLOAT, bias_initializer);
    auto bias_file_path = "test_bias1.txt";
    initialize_tensor_from_file(bias_file_path, bias, ff, "float", 1);
  }

  auto dense_projection_file_path = "test_input1.txt";

  // create dense projection
  Tensor dense_projection;
  {
    int const dims[2] = {test_meta.batch_size,
                         test_meta.dense_projection_i_dim};
    dense_projection = ff.create_tensor<2>(dims, "", DT_FLOAT);
    // dense_projection = ff.create_linear_weight<2>(dims,
    // (IndexSpaceT<2>)task_is, DT_FLOAT, kernel_initializer);
    initialize_tensor_from_file(
        dense_projection_file_path, dense_projection, ff, "float", 2);
  }

  auto output_grad_file_path = "test_output_grad.txt";

  // build transpose layer
  Tensor ret = ff.dense("",
                        dense_projection,
                        test_meta.dense_projection_o_dim,
                        AC_MODE_NONE,
                        true,
                        NULL,
                        NULL,
                        &weights,
                        NULL);
  // init gradient
  initialize_tensor_gradient_from_file(
      output_grad_file_path, ret, ff, "float", 2);

  /*
  TODO
  1. mid size problem kernels dont match
  2. test linear consistency with large problems
     becasue we don't know if SGD perform consistently
  */
  ff.optimizer = new SGDOptimizer(&ff, 0.01f, 0.0f);
  // run forward and backward to produce results
  ff.init_layers();
  int epochs = 1;
  ff.forward();
  for (int i = 0; i < epochs; i++) {
    ff.backward();
    ff.update();
  }

  initialize_tensor_from_file(
      dense_projection_file_path, dense_projection, ff, "float", 2);
  ff.forward();
  // dump results to file for python validation
  dump_region_to_file(ff, ret.region, "output.txt", 2);
  // dump_region_to_file(ff, dense_projection.region, "dump.txt", 2);
  auto kernel = ff.parameters[0].tensor;
  dump_region_to_file(ff, kernel.region, "kernel_updated1.txt", 2);
  // kernel = ff.parameters[1].tensor;
  // dump_region_to_file(ff, kernel.region_grad, "kernel_grad2.txt", 1);
}

void register_custom_tasks() {
  {
    TaskVariantRegistrar registrar(INIT_TENSOR_1D_FROM_FILE_CPU_TASK,
                                   "Load 1d Tensor");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<initialize_tensor_from_file_task<1>>(
        registrar, "Load 1d tensor Task");
  }
  {
    TaskVariantRegistrar registrar(INIT_TENSOR_2D_FROM_FILE_CPU_TASK,
                                   "Load 2d Tensor");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<initialize_tensor_from_file_task<2>>(
        registrar, "Load 2d tensor Task");
  }
  {
    TaskVariantRegistrar registrar(INIT_TENSOR_3D_FROM_FILE_CPU_TASK,
                                   "Load 3d Tensor");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<initialize_tensor_from_file_task<3>>(
        registrar, "Load 3d tensor Task");
  }
  {
    TaskVariantRegistrar registrar(INIT_TENSOR_4D_FROM_FILE_CPU_TASK,
                                   "Load 4d Tensor");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<initialize_tensor_from_file_task<4>>(
        registrar, "Load 4d tensor Task");
  }

  {
    TaskVariantRegistrar registrar(DUMP_TENSOR_1D_CPU_TASK, "Dump Tensor");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<dump_tensor_task<1>>(registrar,
                                                           "Dump Tensor Task");
  }
  {
    TaskVariantRegistrar registrar(DUMP_TENSOR_2D_CPU_TASK, "Dump Tensor");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<dump_tensor_task<2>>(registrar,
                                                           "Dump Tensor Task");
  }
  {
    TaskVariantRegistrar registrar(DUMP_TENSOR_4D_CPU_TASK, "Dump Tensor");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<dump_tensor_task<4>>(registrar,
                                                           "Dump Tensor Task");
  }
  {
    TaskVariantRegistrar registrar(DUMP_TENSOR_3D_CPU_TASK, "Dump Tensor");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<dump_tensor_task<3>>(registrar,
                                                           "Dump Tensor Task");
  }
}
