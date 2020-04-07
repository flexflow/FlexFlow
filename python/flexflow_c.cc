#include "model.h"
#include "flexflow_c.h"

class ImgDataLoader {
public:
  ImgDataLoader(FFModel& ff, Tensor input, Tensor label, int flag);
  void set_num_samples(int samples);
  static void load_input(const Task *task,
                         const std::vector<PhysicalRegion> &regions,
                         Context ctx,
                         Runtime* runtime);
public:
  int num_samples;
};

class FFCObjectWrapper {
public:
#define FF_NEW_OPAQUE_WRAPPER(T_, T)                                   \
  static T_ wrap(T t) {                                             \
    T_ t_;                                                          \
    t_.impl = static_cast<void *>(t);                               \
    return t_;                                                      \
  }                                                                 \
  static const T_ wrap_const(const T t) {                           \
    T_ t_;                                                          \
    t_.impl = const_cast<void *>(static_cast<const void *>(t));     \
    return t_;                                                      \
  }                                                                 \
  static T unwrap(T_ t_) {                                          \
    return static_cast<T>(t_.impl);                                 \
  }                                                                 \
  static const T unwrap_const(const T_ t_) {                        \
    return static_cast<const T>(t_.impl);                           \
  }
  
  FF_NEW_OPAQUE_WRAPPER(flexflow_config_t, FFConfig *);
  FF_NEW_OPAQUE_WRAPPER(flexflow_model_t, FFModel *);
  FF_NEW_OPAQUE_WRAPPER(flexflow_tensor_t, Tensor *);
  FF_NEW_OPAQUE_WRAPPER(flexflow_sgd_optimizer_t, SGDOptimizer *);
  FF_NEW_OPAQUE_WRAPPER(flexflow_glorot_uniform_initializer_t, GlorotUniform *);
  FF_NEW_OPAQUE_WRAPPER(flexflow_zero_initializer_t, ZeroInitializer *);
  FF_NEW_OPAQUE_WRAPPER(flexflow_uniform_initializer_t, UniformInitializer *);
  FF_NEW_OPAQUE_WRAPPER(flexflow_norm_initializer_t, NormInitializer *);
  FF_NEW_OPAQUE_WRAPPER(flexflow_op_t, Op *);
  FF_NEW_OPAQUE_WRAPPER(flexflow_dataloader_t, ImgDataLoader *);
};

// -----------------------------------------------------------------------
// Tensor
// -----------------------------------------------------------------------

void
flexflow_tensor_inline_map(
  flexflow_tensor_t handle_,
  flexflow_config_t config_)
{
  Tensor *handle = FFCObjectWrapper::unwrap(handle_);
  FFConfig *config = FFCObjectWrapper::unwrap(config_);  
  handle->inline_map(*config);
}

void  
flexflow_tensor_inline_unmap(
  flexflow_tensor_t handle_,
  flexflow_config_t config_)
{
  Tensor *handle = FFCObjectWrapper::unwrap(handle_);
  FFConfig *config = FFCObjectWrapper::unwrap(config_);  
  handle->inline_unmap(*config);
}

float*  
flexflow_tensor_get_raw_ptr_float(
  flexflow_tensor_t handle_,
  flexflow_config_t config_)
{
  Tensor *handle = FFCObjectWrapper::unwrap(handle_);
  FFConfig *config = FFCObjectWrapper::unwrap(config_);  
  float *raw_ptr = handle->get_raw_ptr_float(*config);
  return raw_ptr;
}

int
flexflow_tensor_get_num_dims(
  flexflow_tensor_t handle_)
{
  Tensor *handle = FFCObjectWrapper::unwrap(handle_);
  return handle->numDim;
}

int*
flexflow_tensor_get_dims(
  flexflow_tensor_t handle_)
{
  Tensor *handle = FFCObjectWrapper::unwrap(handle_);
  return &(handle->adim[0]);
}

// -----------------------------------------------------------------------
// FFConfig
// -----------------------------------------------------------------------

flexflow_config_t
flexflow_config_create(void)
{
  FFConfig *config = new FFConfig();
  Runtime *runtime = Runtime::get_runtime();
  config->lg_hlr = runtime;
  config->lg_ctx = Runtime::get_context();
  config->field_space = runtime->create_field_space(config->lg_ctx);
  printf("new FFConfig %p\n", config);
  return FFCObjectWrapper::wrap(config);
}

void
flexflow_config_destroy(
  flexflow_config_t handle_)
{
  FFConfig *handle = FFCObjectWrapper::unwrap(handle_);
  printf("delete FFConfig %p\n", handle);
  delete handle;
}

void
flexflow_config_parse_args(
  flexflow_config_t handle_,
  char** argv, 
  int argc)
{
  FFConfig *handle = FFCObjectWrapper::unwrap(handle_);
  handle->parse_args(argv, argc);  
}

void
flexflow_config_parse_args_default(
  flexflow_config_t handle_)
{
  FFConfig *handle = FFCObjectWrapper::unwrap(handle_);
  const InputArgs &command_args = Runtime::get_input_args();
  char **argv = command_args.argv;
  int argc = command_args.argc;
  handle->parse_args(argv, argc);  
}

int
flexflow_config_get_batch_size(
  flexflow_config_t handle_)
{
  FFConfig *handle = FFCObjectWrapper::unwrap(handle_);
  return handle->batchSize;
}

int
flexflow_config_get_workers_per_node(
  flexflow_config_t handle_)
{
  FFConfig *handle = FFCObjectWrapper::unwrap(handle_);
  return handle->workersPerNode;
}

int
flexflow_config_get_num_nodes(
  flexflow_config_t handle_)
{
  FFConfig *handle = FFCObjectWrapper::unwrap(handle_);
  return handle->numNodes;
}

int
flexflow_config_get_epochs(
  flexflow_config_t handle_)
{
  FFConfig *handle = FFCObjectWrapper::unwrap(handle_);
  return handle->epochs;
}

// -----------------------------------------------------------------------
// FFModel
// -----------------------------------------------------------------------

flexflow_model_t
flexflow_model_create(
  flexflow_config_t config_)
{
  FFConfig *config = FFCObjectWrapper::unwrap(config_);
  FFModel *model = new FFModel(*config);
  printf("new FFModel %p\n", model);
  return FFCObjectWrapper::wrap(model);
}

void
flexflow_model_destroy(
  flexflow_model_t handle_)
{
  FFModel *handle = FFCObjectWrapper::unwrap(handle_);
  printf("delete FFModel %p\n", handle); 
  delete handle;
}

void
flexflow_model_reset_metrics(
  flexflow_model_t handle_)
{
  FFModel *handle = FFCObjectWrapper::unwrap(handle_);
  handle->reset_metrics();
}

void
flexflow_model_init_layers(
  flexflow_model_t handle_)
{
  FFModel *handle = FFCObjectWrapper::unwrap(handle_);
  handle->init_layers();
}

void
flexflow_model_prefetch(
  flexflow_model_t handle_)
{
  FFModel *handle = FFCObjectWrapper::unwrap(handle_);
  handle->prefetch();
}

void
flexflow_model_forward(
  flexflow_model_t handle_)
{
  FFModel *handle = FFCObjectWrapper::unwrap(handle_);
  handle->forward();
}

void
flexflow_model_backward(
  flexflow_model_t handle_)
{
  FFModel *handle = FFCObjectWrapper::unwrap(handle_);
  handle->backward();
}

void
flexflow_model_update(
  flexflow_model_t handle_)
{
  FFModel *handle = FFCObjectWrapper::unwrap(handle_);
  handle->update();
}

void
flexflow_model_zero_gradients(
  flexflow_model_t handle_)
{
  FFModel *handle = FFCObjectWrapper::unwrap(handle_);
  handle->zero_gradients();
}

flexflow_tensor_t
flexflow_model_add_conv2d(
  flexflow_model_t handle_,
  const char* name,
  const flexflow_tensor_t input_,
  int out_channels,
  int kernel_h, int kernel_w,
  int stride_h, int stride_w,
  int padding_h, int padding_w,
  enum ActiMode activation /* AC_MODE_NONE */,
  bool use_bias /* True */)
{
  FFModel *handle = FFCObjectWrapper::unwrap(handle_);
  const Tensor *input = FFCObjectWrapper::unwrap_const(input_);
  Tensor *tensor = new Tensor();
  *tensor = handle->conv2d(name, *input, out_channels, kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w, activation, use_bias);
  printf("Conv2d new Tensor 4D %p, activation %d, use_bias %d\n", tensor, activation, use_bias);
  return FFCObjectWrapper::wrap(tensor);   
}

flexflow_tensor_t
flexflow_model_add_embedding_with_glorot_uniform_initializer(
  flexflow_model_t handle_,
  const char* name,
  const flexflow_tensor_t input_,
  int num_entires, int out_dim,
  enum AggrMode aggr,
  flexflow_glorot_uniform_initializer_t kernel_initializer_)
{
  FFModel *handle = FFCObjectWrapper::unwrap(handle_);
  const Tensor *input = FFCObjectWrapper::unwrap_const(input_);
  Tensor *tensor = new Tensor();
  GlorotUniform *kernel_initializer = FFCObjectWrapper::unwrap(kernel_initializer_);
  Initializer *initializer = static_cast<Initializer *>(kernel_initializer);
  *tensor = handle->embedding(name, *input, num_entires, out_dim, aggr, initializer);
  printf("Embedding with GlorotUniform  new Tensor 4D %p\n", tensor);
  return FFCObjectWrapper::wrap(tensor);   
}
  
flexflow_tensor_t
flexflow_model_add_embedding_with_zero_initializer(
  flexflow_model_t handle_,
  const char* name,
  const flexflow_tensor_t input_,
  int num_entires, int out_dim,
  enum AggrMode aggr,
  flexflow_zero_initializer_t kernel_initializer_)
{
  FFModel *handle = FFCObjectWrapper::unwrap(handle_);
  const Tensor *input = FFCObjectWrapper::unwrap_const(input_);
  Tensor *tensor = new Tensor();
  ZeroInitializer *kernel_initializer = FFCObjectWrapper::unwrap(kernel_initializer_);
  Initializer *initializer = static_cast<Initializer *>(kernel_initializer);
  *tensor = handle->embedding(name, *input, num_entires, out_dim, aggr, initializer);
  printf("Embedding with ZeroInitializer new Tensor 4D %p\n", tensor);
  return FFCObjectWrapper::wrap(tensor);  
}
  
flexflow_tensor_t
flexflow_model_add_embedding_with_uniform_initializer(
  flexflow_model_t handle_,
  const char* name,
  const flexflow_tensor_t input_,
  int num_entires, int out_dim,
  enum AggrMode aggr,
  flexflow_uniform_initializer_t kernel_initializer_)
{
  FFModel *handle = FFCObjectWrapper::unwrap(handle_);
  const Tensor *input = FFCObjectWrapper::unwrap_const(input_);
  Tensor *tensor = new Tensor();
  UniformInitializer *kernel_initializer = FFCObjectWrapper::unwrap(kernel_initializer_);
  Initializer *initializer = static_cast<Initializer *>(kernel_initializer);
  *tensor = handle->embedding(name, *input, num_entires, out_dim, aggr, initializer);
  printf("Embedding with UniformInitializer new Tensor 4D %p\n", tensor);
  return FFCObjectWrapper::wrap(tensor);  
}
  
flexflow_tensor_t
flexflow_model_add_embedding_with_norm_initializer(
  flexflow_model_t handle_,
  const char* name,
  const flexflow_tensor_t input_,
  int num_entires, int out_dim,
  enum AggrMode aggr,
  flexflow_norm_initializer_t kernel_initializer_)
{
  FFModel *handle = FFCObjectWrapper::unwrap(handle_);
  const Tensor *input = FFCObjectWrapper::unwrap_const(input_);
  Tensor *tensor = new Tensor();
  NormInitializer *kernel_initializer = FFCObjectWrapper::unwrap(kernel_initializer_);
  Initializer *initializer = static_cast<Initializer *>(kernel_initializer);
  *tensor = handle->embedding(name, *input, num_entires, out_dim, aggr, initializer);
  printf("Embedding with NormInitializer new Tensor 4D %p\n", tensor);
  return FFCObjectWrapper::wrap(tensor);    
}

flexflow_tensor_t
flexflow_model_add_pool2d(
  flexflow_model_t handle_,
  const char* name,
  flexflow_tensor_t input_,
  int kernel_h, int kernel_w,
  int stride_h, int stride_w,
  int padding_h, int padding_w,
  enum PoolType type /* POOL_MAX */, 
  enum ActiMode activation /* AC_MODE_NONE */)
{
  FFModel *handle = FFCObjectWrapper::unwrap(handle_);
  Tensor *input = FFCObjectWrapper::unwrap(input_);
  Tensor *tensor = new Tensor();
  *tensor = handle->pool2d(name, *input, kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w, type, activation);
  printf("Pool2d new Tensor 4D %p, pool %d, activation %d\n", tensor, type, activation);
  return FFCObjectWrapper::wrap(tensor); 
}

flexflow_tensor_t
flexflow_model_add_dense_with_default_initializer(
  flexflow_model_t handle_,
  const char* name,
  const flexflow_tensor_t input_,
  int out_dim,
  enum ActiMode activation /* AC_MODE_NONE */,
  bool use_bias /* true */)
{
  FFModel *handle = FFCObjectWrapper::unwrap(handle_);
  const Tensor *input = FFCObjectWrapper::unwrap_const(input_);
  Tensor *tensor = new Tensor();
  *tensor = handle->dense(name, *input, out_dim, activation, use_bias);
  printf("Dense default new Tensor 4D %p, activation %d, use_bias %d\n", tensor, activation, use_bias);
  return FFCObjectWrapper::wrap(tensor); 
}

flexflow_tensor_t
flexflow_model_add_linear_with_default_initializer(
  flexflow_model_t handle_,
  const char* name,
  const flexflow_tensor_t input_,
  int out_dim,
  enum ActiMode activation /* AC_MODE_NONE */,
  bool use_bias /* true */)
{
  FFModel *handle = FFCObjectWrapper::unwrap(handle_);
  const Tensor *input = FFCObjectWrapper::unwrap_const(input_);
  Tensor *tensor = new Tensor();
  *tensor = handle->linear(name, *input, out_dim, activation, use_bias);
  printf("Linear default new Tensor 4D %p, activation %d, use_bias %d\n", tensor, activation, use_bias);
  return FFCObjectWrapper::wrap(tensor); 
}

flexflow_tensor_t
flexflow_model_add_concat(
  flexflow_model_t handle_,
  const char* name,
  int n,
  flexflow_tensor_t* input_,
  int axis)
{
  FFModel *handle = FFCObjectWrapper::unwrap(handle_);
  Tensor *tensor = new Tensor();
  std::vector<Tensor> input_vec;
  for (int i = 0; i < n; i++ ) {
    Tensor *t = FFCObjectWrapper::unwrap(input_[i]);
    input_vec.push_back(*t);
  }
  *tensor = handle->concat(name, n, input_vec.data(), axis);
  printf("concat new Tensor 4D %p\n", tensor);
  return FFCObjectWrapper::wrap(tensor); 
}

flexflow_tensor_t
flexflow_model_add_flat(
  flexflow_model_t handle_,
  const char* name,
  flexflow_tensor_t input_)
{
  FFModel *handle = FFCObjectWrapper::unwrap(handle_);
  Tensor *input = FFCObjectWrapper::unwrap(input_);
  Tensor *tensor = new Tensor();
  *tensor = handle->flat(name, *input);
  printf("Flat new Tensor 4D %p\n", tensor);
  return FFCObjectWrapper::wrap(tensor);  
}

flexflow_tensor_t
flexflow_model_add_softmax(
  flexflow_model_t handle_,
  const char* name,
  const flexflow_tensor_t input_,
  const flexflow_tensor_t label_)
{
  FFModel *handle = FFCObjectWrapper::unwrap(handle_);
  Tensor *input = FFCObjectWrapper::unwrap(input_);
  Tensor *label = FFCObjectWrapper::unwrap(label_);
  Tensor *tensor = new Tensor();
  *tensor = handle->softmax(name, *input, *label);
  printf("Softmax new Tensor 4D %p\n", tensor);
  return FFCObjectWrapper::wrap(tensor);   
}

void
flexflow_model_set_sgd_optimizer(
  flexflow_model_t handle_, 
  flexflow_sgd_optimizer_t optimizer_)
{
  FFModel *handle = FFCObjectWrapper::unwrap(handle_);
  SGDOptimizer *optimizer = FFCObjectWrapper::unwrap(optimizer_);
  handle->optimizer = static_cast<Optimizer *>(optimizer);
}

void
flexflow_model_print_layers(
  flexflow_model_t handle_)
{
  FFModel *handle = FFCObjectWrapper::unwrap(handle_);
  handle->print_layers();
}

flexflow_op_t
flexflow_model_get_layer_by_id(
  flexflow_model_t handle_,
  int layer_id)
{
  FFModel *handle = FFCObjectWrapper::unwrap(handle_);
  Op* layer = handle->layers[layer_id];
  return FFCObjectWrapper::wrap(layer);  
}

// -----------------------------------------------------------------------
// Tensor
// -----------------------------------------------------------------------

flexflow_tensor_t
flexflow_tensor_4d_create(
  flexflow_model_t model_,
  const int* dims, 
  const char* pc_name, 
  enum DataType data_type, 
  bool create_grad /* true */)
{
  Tensor *tensor = new Tensor();
  FFModel *model = FFCObjectWrapper::unwrap(model_);
  *tensor = model->create_tensor<4>(dims, pc_name, data_type, create_grad);
  printf("new Tensor 4D %p\n", tensor);
  return FFCObjectWrapper::wrap(tensor);
}

flexflow_tensor_t
flexflow_tensor_2d_create(
  flexflow_model_t model_,
  const int* dims, 
  const char* pc_name, 
  enum DataType data_type, 
  bool create_grad /* true */)
{
  Tensor *tensor = new Tensor();
  FFModel *model = FFCObjectWrapper::unwrap(model_);
  *tensor = model->create_tensor<2>(dims, pc_name, data_type, create_grad);
  printf("new Tensor 2D %p\n", tensor);
  return FFCObjectWrapper::wrap(tensor);
}

void
flexflow_tensor_destroy(
  flexflow_tensor_t handle_)
{
  Tensor *handle = FFCObjectWrapper::unwrap(handle_);
  printf("delete Tensor %p\n", handle);
  delete handle;
}

// -----------------------------------------------------------------------
// SGDOptimizer
// -----------------------------------------------------------------------

flexflow_sgd_optimizer_t
flexflow_sgd_optimizer_create(
  flexflow_model_t model_,
  double lr, /* 0.01f */
  double momentum, /* 0.0f */
  bool nesterov, /* false */
  double weight_decay /* 0.0f */ )
{
  const FFModel *model = FFCObjectWrapper::unwrap_const(model_);
  SGDOptimizer *optimizer = new SGDOptimizer(model, lr, momentum, nesterov, weight_decay);
  printf("new SGDOptimizer %p\n", optimizer);
  return FFCObjectWrapper::wrap(optimizer);
}

void 
flexflow_sgd_optimizer_destroy(
  flexflow_sgd_optimizer_t handle_)
{
  SGDOptimizer *handle = FFCObjectWrapper::unwrap(handle_);
  printf("delete SGDOptimizer %p\n", handle);
  delete handle;
}

// -----------------------------------------------------------------------
// GlorotUniform
// -----------------------------------------------------------------------

flexflow_glorot_uniform_initializer_t
flexflow_glorot_uniform_initializer_create(
  int seed)
{
  GlorotUniform *initializer = new GlorotUniform(seed);
  return FFCObjectWrapper::wrap(initializer); 
}

void  
flexflow_glorot_uniform_initializer_destroy(
  flexflow_glorot_uniform_initializer_t handle_)
{
  GlorotUniform *handle = FFCObjectWrapper::unwrap(handle_);
  delete handle;
}

// -----------------------------------------------------------------------
// ZeroInitializer
// -----------------------------------------------------------------------

flexflow_zero_initializer_t
flexflow_zero_initializer_create(void)
{
  ZeroInitializer *initializer = new ZeroInitializer();
  return FFCObjectWrapper::wrap(initializer); 
}

void  
flexflow_zero_initializer_destroy(
  flexflow_zero_initializer_t handle_)
{
  ZeroInitializer *handle = FFCObjectWrapper::unwrap(handle_);
  delete handle;
}

// -----------------------------------------------------------------------
// UniformInitializer
// -----------------------------------------------------------------------

flexflow_uniform_initializer_t
flexflow_uniform_initializer_create(
  int seed, 
  float min, 
  float max)
{
  UniformInitializer *initializer = new UniformInitializer(seed, min, max);
  return FFCObjectWrapper::wrap(initializer);  
}

void  
flexflow_uniform_initializer_destroy(
  flexflow_uniform_initializer_t handle_)
{
  UniformInitializer *handle = FFCObjectWrapper::unwrap(handle_);
  delete handle;
}

// -----------------------------------------------------------------------
// NormInitializer
// -----------------------------------------------------------------------

flexflow_norm_initializer_t
flexflow_norm_initializer_create(
  int seed, 
  float mean, 
  float stddev)
{
  NormInitializer *initializer = new NormInitializer(seed, mean, stddev);
  return FFCObjectWrapper::wrap(initializer);  
}

void  
flexflow_norm_initializer_destroy(
  flexflow_norm_initializer_t handle_)
{
  NormInitializer *handle = FFCObjectWrapper::unwrap(handle_);
  delete handle;
}

// -----------------------------------------------------------------------
// DataLoader
// -----------------------------------------------------------------------

flexflow_dataloader_t
flexflow_dataloader_create(
  flexflow_model_t ffmodel_, 
  flexflow_tensor_t input_, 
  flexflow_tensor_t label_,
  int flag)
{
  FFModel *ffmodel = FFCObjectWrapper::unwrap(ffmodel_);
  Tensor *input = FFCObjectWrapper::unwrap(input_);
  Tensor *label = FFCObjectWrapper::unwrap(label_);
  ImgDataLoader *dataloader = new ImgDataLoader(*ffmodel, *input, *label, flag);
  return FFCObjectWrapper::wrap(dataloader);  
}

void  
flexflow_dataloader_destroy(
  flexflow_dataloader_t handle_)
{
  ImgDataLoader *handle = FFCObjectWrapper::unwrap(handle_);
  delete handle;
}

void
flexflow_dataloader_set_num_samples(
  flexflow_dataloader_t handle_,
  int samples)
{
  ImgDataLoader *handle = FFCObjectWrapper::unwrap(handle_);
  handle->set_num_samples(samples);  
  printf("dataloader set number of samples %d\n", samples);
}

int
flexflow_dataloader_get_num_samples(
  flexflow_dataloader_t handle_)
{
  ImgDataLoader *handle = FFCObjectWrapper::unwrap(handle_);
  return handle->num_samples;
}

// -----------------------------------------------------------------------
// Timer
// -----------------------------------------------------------------------

double
flexflow_get_current_time(
  flexflow_config_t config_)
{
  FFConfig *config = FFCObjectWrapper::unwrap(config_);
  config->lg_hlr->issue_execution_fence(config->lg_ctx);
  TimingLauncher timer(MEASURE_MICRO_SECONDS);
  Future future = config->lg_hlr->issue_timing_measurement(config->lg_ctx, timer);
  future.get_void_result();
  double ts_start = Realm::Clock::current_time_in_microseconds();
  return ts_start;
}

// -----------------------------------------------------------------------
// Trace
// -----------------------------------------------------------------------

void
flexflow_begin_trace(
  flexflow_config_t config_, 
  int trace_id)
{
  FFConfig *config = FFCObjectWrapper::unwrap(config_);
  config->lg_hlr->begin_trace(config->lg_ctx, trace_id);
}

void
flexflow_end_trace(
  flexflow_config_t config_, 
  int trace_id)
{
  FFConfig *config = FFCObjectWrapper::unwrap(config_);
  config->lg_hlr->end_trace(config->lg_ctx, trace_id);
}

// -----------------------------------------------------------------------
// Op
// -----------------------------------------------------------------------
flexflow_tensor_t
flexflow_op_get_weight(
  flexflow_op_t handle_)
{
  Op *handle = FFCObjectWrapper::unwrap(handle_);
  Tensor *tensor = handle->get_weight();
  return FFCObjectWrapper::wrap(tensor);  
} 

flexflow_tensor_t
flexflow_op_get_bias(
  flexflow_op_t handle_)
{
  Op *handle = FFCObjectWrapper::unwrap(handle_);
  Tensor *tensor = handle->get_bias();
  return FFCObjectWrapper::wrap(tensor);  
}

int*
flexflow_malloc_int(
  int size)
{
  int *ptr = NULL;
  uintptr_t intptr; 
  ptr = (int*)malloc(sizeof(int) * size);
  for (int i = 0; i < size; i++) {
    ptr[i] = 1;
  }
  intptr = (uintptr_t)(ptr);
  printf("malloc int %p, %ld, size %ld\n", ptr, intptr, size);
  return ptr;
}

void
flexflow_print_array_int(
  int *base_ptr,
  int size)
{
  printf("base_ptr %p\n", base_ptr);
  for (int i = 0; i < size; i++) {
    printf("%d ", base_ptr[i]);
  }   
  printf("\n");
}

// -----------------------------------------------------------------------
// ImgDataLoader implementation
// -----------------------------------------------------------------------

ImgDataLoader::ImgDataLoader(FFModel& ff,
                       Tensor input, Tensor label, int flag)
{
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  num_samples = 0;
  printf("Use random dataset...");
  num_samples = 256 * 10 * ff.config.workersPerNode * ff.config.numNodes;
  printf("Number of random samples = %d\n", num_samples);
  // Init input
  {
    IndexSpaceT<4> task_is = IndexSpaceT<4>(ff.get_or_create_task_is(4, ""));
    ArgumentMap argmap;
    IndexLauncher launcher(CUSTOM_GPU_TASK_ID_1, task_is,
                           TaskArgument(NULL, 0), argmap,
                           Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
                           FFConfig::get_hash_id(std::string("")));
    launcher.add_region_requirement(
        RegionRequirement(input.part, 0/*projection id*/,
                          WRITE_ONLY, EXCLUSIVE, input.region));
    launcher.add_field(0, FID_DATA);
    runtime->execute_index_space(ctx, launcher);
  }
  // Init label
  if (flag == 1){
    IndexSpaceT<2> task_is = IndexSpaceT<2>(ff.get_or_create_task_is(2, ""));
    ArgumentMap argmap;
    IndexLauncher launcher(CUSTOM_GPU_TASK_ID_1, task_is,
                           TaskArgument(NULL, 0), argmap,
                           Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
                           FFConfig::get_hash_id(std::string("")));
    launcher.add_region_requirement(
        RegionRequirement(label.part, 0/*projection id*/,
                          WRITE_ONLY, EXCLUSIVE, label.region));
    launcher.add_field(0, FID_DATA);
    runtime->execute_index_space(ctx, launcher);
  }
}

void ImgDataLoader::set_num_samples(int samples)
{
  num_samples = samples;
}

void ImgDataLoader::load_input(const Task *task,
                            const std::vector<PhysicalRegion> &regions,
                            Context ctx,
                            Runtime* runtime)
{
  printf("CheckPoint#1\n");
}

void register_c_custom_tasks()
{
  // Load Input
  {
    TaskVariantRegistrar registrar(CUSTOM_GPU_TASK_ID_1, "Load Inputs");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<ImgDataLoader::load_input>(
        registrar, "Load Inputs Task");
  }
}
