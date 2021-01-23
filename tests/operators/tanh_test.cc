#include "model.h"
#include "test_utils.h"
#include <sstream>
#include <fstream>
#include <iomanip>
#include <iostream>
#define  PRECISION 16
using namespace Legion;
LegionRuntime::Logger::Category log_app("Tanh_test");

struct TanhTestMeta {
  int i_dim, o_dim;
  int* i_shape; 
  int* o_shape;
  TanhTestMeta(int _i_dim, int _o_dim, int* _i_shape, int* _o_shape) {
      i_dim = _i_dim;
      o_dim = _o_dim;
      i_shape = _i_shape;
      o_shape = _o_shape;
  }
};

TanhTestMeta get_test_meta(const std::string file_path) {
  std::fstream myfile(file_path, std::ios_base::in);
  int b;
  std::vector<int> buffer;
  while (myfile >> b) 
  { 
      buffer.push_back(b); 
  } 
  int i_dim(buffer[0]), o_dim(buffer[1]);
  int* i_shape = new int[i_dim];
  int* o_shape = new int[o_dim];
  int offset = 2;
  for (int i = 0; i < i_dim; i++){
    i_shape[i] = buffer[i+offset];
  }
  offset += i_dim;
  for (int i = 0; i < o_dim; i++){
    o_shape[i] = buffer[i+offset];
  }
  // int m,k,d;
  // myfile >> m >> k >> d;
  return TanhTestMeta(i_dim, o_dim, i_shape, o_shape);
}


void top_level_task(const Task* task,
                    const std::vector<PhysicalRegion>& regions,
                    Context ctx, Runtime* runtime)
{
// std::cout<< "test framework launched" << std::endl;
  auto test_meta = get_test_meta("test_meta.txt");
  FFConfig ffConfig;
  // Parse input arguments
  {
    const InputArgs &command_args = HighLevelRuntime::get_input_args();
    char **argv = command_args.argv;
    int argc = command_args.argc;
    ffConfig.parse_args(argv, argc);
  }
  ffConfig.lg_ctx = ctx;
  ffConfig.lg_hlr = runtime;
  ffConfig.profiling = false;
  ffConfig.field_space = runtime->create_field_space(ctx);
  // create ff model object
  FFModel ff(ffConfig);
  Tensor dense_input;
  if (test_meta.i_dim == 3) {
#define input_dim 3
    const int i_dims[input_dim] = {
      test_meta.i_shape[0], 
      test_meta.i_shape[1], 
      test_meta.i_shape[2]
    }; 
    dense_input = ff.create_tensor<input_dim>(i_dims, "", DT_FLOAT);
    Tensor ret = ff.tanh<input_dim>("", dense_input, i_dims);
    auto input1_file_path = "test_input1.txt";
    auto output_grad_file_path = "test_output_grad.txt";
    initialize_tensor_from_file(input1_file_path, dense_input, ff, "float", input_dim);
    initialize_tensor_gradient_from_file(output_grad_file_path, ret, ff, "float", input_dim);
    // run forward and backward to produce results
    ff.init_layers();
    // forward
    ff.forward();
    dump_region_to_file(ff, ret.region, "output.txt", input_dim);
    ff.backward();
    dump_region_to_file(ff, dense_input.region_grad, "input1_grad.txt", input_dim);
#undef input_dim
    }
  else if (test_meta.i_dim == 2) {
#define input_dim 2
    const int i_dims[input_dim] = {
      test_meta.i_shape[0], 
      test_meta.i_shape[1], 
    }; 
    dense_input = ff.create_tensor<input_dim>(i_dims, "", DT_FLOAT);
    Tensor ret = ff.tanh<input_dim>("", dense_input, i_dims);
    auto input1_file_path = "test_input1.txt";
    auto output_grad_file_path = "test_output_grad.txt";
    initialize_tensor_from_file(input1_file_path, dense_input, ff, "float", input_dim);
    initialize_tensor_gradient_from_file(output_grad_file_path, ret, ff, "float", input_dim);
    // run forward and backward to produce results
    ff.init_layers();
    // forward
    ff.forward();
    dump_region_to_file(ff, ret.region, "output.txt", input_dim);
    ff.backward();
    dump_region_to_file(ff, dense_input.region_grad, "input1_grad.txt", input_dim);
#undef input_dim
  }
  else if (test_meta.i_dim == 1) {
#define input_dim 1
    const int i_dims[input_dim] = {
      test_meta.i_shape[0]
    }; 
    dense_input = ff.create_tensor<input_dim>(i_dims, "", DT_FLOAT);
    Tensor ret = ff.tanh<input_dim>("", dense_input, i_dims);
    auto input1_file_path = "test_input1.txt";
    auto output_grad_file_path = "test_output_grad.txt";
    initialize_tensor_from_file(input1_file_path, dense_input, ff, "float", input_dim);
    initialize_tensor_gradient_from_file(output_grad_file_path, ret, ff, "float", input_dim);
    // run forward and backward to produce results
    ff.init_layers();
    // forward
    ff.forward();
    dump_region_to_file(ff, ret.region, "output.txt", input_dim);
    ff.backward();
    dump_region_to_file(ff, dense_input.region_grad, "input1_grad.txt", input_dim);
#undef input_dim
  }  

  else {
    printf("i_dim %d o_dim %d not supported\n" , test_meta.i_dim, test_meta.o_dim);
    throw 255;
  }
}




void register_custom_tasks()
{
  {
    TaskVariantRegistrar registrar(INIT_TENSOR_1D_FROM_FILE_CPU_TASK, "Load 1d Tensor");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<initialize_tensor_from_file_task<1>>(
        registrar, "Load 1d tensor Task");
  }
  {
    TaskVariantRegistrar registrar(INIT_TENSOR_2D_FROM_FILE_CPU_TASK, "Load 2d Tensor");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<initialize_tensor_from_file_task<2>>(
        registrar, "Load 2d tensor Task");
  }
  {
    TaskVariantRegistrar registrar(INIT_TENSOR_3D_FROM_FILE_CPU_TASK, "Load 3d Tensor");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<initialize_tensor_from_file_task<3>>(
        registrar, "Load 3d tensor Task");
  }
  {
    TaskVariantRegistrar registrar(INIT_TENSOR_4D_FROM_FILE_CPU_TASK, "Load 4d Tensor");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<initialize_tensor_from_file_task<4>>(
        registrar, "Load 4d tensor Task");
  }

  {      
    TaskVariantRegistrar registrar(DUMP_TENSOR_1D_CPU_TASK, "Dump Tensor");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<dump_tensor_task<1>>(
        registrar, "Dump Tensor Task");
  }
  {      
    TaskVariantRegistrar registrar(DUMP_TENSOR_2D_CPU_TASK, "Dump Tensor");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<dump_tensor_task<2>>(
        registrar, "Dump Tensor Task");
  }
  {      
    TaskVariantRegistrar registrar(DUMP_TENSOR_4D_CPU_TASK, "Dump Tensor");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<dump_tensor_task<4>>(
        registrar, "Dump Tensor Task");
  }
  {      
    TaskVariantRegistrar registrar(DUMP_TENSOR_3D_CPU_TASK, "Dump Tensor");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<dump_tensor_task<3>>(
        registrar, "Dump Tensor Task");
  }
}