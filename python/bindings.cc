#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "flexflow/config.h"
#include "flexflow/ffconst.h"
#include "flexflow/mapper.h"
#include "flexflow/metrics_functions.h"
#include "flexflow/model.h"
#include "flexflow/tensor.h"
#include "flexflow_c.h"
#include "flexflow_dataloader.h"

namespace py = pybind11;
using py::literals::operator""_a;

namespace {

using namespace FlexFlow;

static Context ctx;

void begin_flexflow_task(std::vector<std::string> args) {
  // This needs to be set, otherwise NCCL will try to use group kernel launches,
  // which are not compatible with the Realm CUDA hijack.
  setenv("NCCL_LAUNCH_MODE", "PARALLEL", true);

  std::vector<char const *> argvec;
  argvec.push_back("python");
  for (auto &arg : args) {
    if (arg == "-ll:py") {
      throw std::invalid_argument(
          "-ll:py is not supported when using native python");
    }
    argvec.push_back(arg.data());
  }
  int argc = argvec.size();
  char **argv = const_cast<char **>(argvec.data());

  register_flexflow_internal_tasks();

  register_c_custom_tasks();

  Runtime::add_registration_callback(FFMapper::update_mappers);

  // Start the runtime in background mode
  Runtime::start(argc, argv, true /*background*/);
  // Get the runtime now that we've started it
  Runtime *runtime = Runtime::get_runtime();
  // Then we can bind make this thread into an implicit top-level task
  ctx = runtime->begin_implicit_task(PYTHON_TOP_LEVEL_TASK_ID,
                                     0 /*mapper id*/,
                                     Processor::LOC_PROC,
                                     "flexflow_top_level_task",
                                     true /*control replicable*/);
}

void finish_flexflow_task() {
  Runtime *runtime = Runtime::get_runtime();
  runtime->finish_implicit_task(ctx);
  // The previous call is asynchronous so we still need to
  // wait for the shutdown of the runtime to complete
  Runtime::wait_for_shutdown();
}

double get_current_time(FFConfig &config) {
  config.lg_hlr->issue_execution_fence(config.lg_ctx);
  TimingLauncher timer(MEASURE_MICRO_SECONDS);
  Future future = config.lg_hlr->issue_timing_measurement(config.lg_ctx, timer);
  future.get_void_result();
  double ts_start = Realm::Clock::current_time_in_microseconds();
  return ts_start;
}

//-------- Tensor --------
py::array get_array(Tensor t, FFConfig &config) {
  std::vector<int> dims(t->num_dims);
  for (int i = 0; i < t->num_dims; i++) {
    dims[i] = t->dims[i];
  }
  std::reverse(dims.begin(), dims.end());
  if (t->data_type == DataType::DT_FLOAT) {
    printf("raw_ptr = %p\n", t->parallel_tensor->get_raw_ptr<float>(config));
    return py::array(
        py::dtype("f"),
        {dims},                                        // shape
        {},                                            // stride
        t->parallel_tensor->get_raw_ptr<float>(config) // the data pointer
    );
  } else if (t->data_type == DataType::DT_INT32) {
    return py::array(
        py::dtype("i"),
        {dims},                                          // shape
        {},                                              // stride
        t->parallel_tensor->get_raw_ptr<int32_t>(config) // the data pointer
    );
  } else {
    assert(0);
  }
}

void set_tensor(Tensor t, FFModel &model, py::array &np_array) {
  py::buffer_info info = np_array.request();
  bool retval = false;
  assert(info.ndim == t->num_dims);
  std::vector<int> dims(t->num_dims);
  for (int i = 0; i < t->num_dims; i++) {
    dims[i] = t->dims[i];
  }
  std::reverse(dims.begin(), dims.end());
  if (info.format == "f") {
    assert(t->data_type == DataType::DT_FLOAT);
    assert(t->parallel_tensor != nullptr);
    retval = t->parallel_tensor->set_tensor<float>(
        &model, dims, static_cast<float *>(info.ptr));
  } else if (info.format == "i") {
    assert(t->data_type == DataType::DT_INT32);
    assert(t->parallel_tensor != nullptr);
    retval = t->parallel_tensor->set_tensor<int32_t>(
        &model, dims, static_cast<int32_t *>(info.ptr));
  } else {
    assert(0);
  }
  assert(retval == true);
}

void attach_numpy_array(Tensor t, FFConfig &config, py::array &np_array) {
  py::buffer_info info = np_array.request();
  assert(info.ndim == t->num_dims);
  if (info.format == "f") {
    assert(t->data_type == DataType::DT_FLOAT);
  } else if (info.format == "i") {
    assert(t->data_type == DataType::DT_INT32);
  } else {
    assert(0);
  }
  assert(t->parallel_tensor != nullptr);
  t->parallel_tensor->attach_raw_ptr(config, info.ptr, true);
}

void detach_numpy_array(Tensor t, FFConfig &config) {
  assert(t->parallel_tensor != nullptr);
  t->parallel_tensor->detach_raw_ptr(config);
}

//-------- Parameter --------

bool get_weights(Parameter parameter, FFModel &model, py::array &full_array) {
  py::buffer_info info = full_array.request();
  if (info.format == "f") {
    assert(parameter->parallel_tensor != nullptr);
    return parameter->parallel_tensor->get_tensor<float>(
        &model, static_cast<float *>(info.ptr), false /*get_gradients*/);
  } else {
    assert(0);
    return false;
  }
}

bool set_weights(Parameter parameter,
                 FFModel &model,
                 std::vector<int> const &dims,
                 py::array &full_array) {
  py::buffer_info info = full_array.request();
  if (info.format == "f") {
    assert(parameter->parallel_tensor != nullptr);
    return parameter->parallel_tensor->set_tensor<float>(
        &model, dims, static_cast<float *>(info.ptr));
  } else {
    assert(0);
    return false;
  }
}

//-------- FFModel --------

Tensor create_tensor(FFModel &model,
                     std::vector<int> const &dims,
                     DataType data_type,
                     bool create_grad) {
  Tensor tensor = NULL;
  int num_dims = dims.size();
  if (num_dims == 2) {
    tensor =
        model.create_tensor<2>(dims.data(), data_type, NULL, 0, create_grad);
  } else if (num_dims == 3) {
    tensor =
        model.create_tensor<3>(dims.data(), data_type, NULL, 0, create_grad);
  } else if (num_dims == 4) {
    tensor =
        model.create_tensor<4>(dims.data(), data_type, NULL, 0, create_grad);
#if MAX_TENSOR_DIM >= 5
  } else if (num_dims == 5) {
    tensor =
        model.create_tensor<5>(dims.data(), data_type, NULL, 0, create_grad);
#endif
  } else {
    assert(0);
  }
  return tensor;
}

SingleDataLoader *create_data_loader_ptr(FFModel &model,
                                         Tensor &batch_tensor,
                                         py::array &full_array) {
  py::buffer_info info = full_array.request();
  DataType dtype;
  if (info.format == "f") {
    dtype = DataType::DT_FLOAT;
  } else if (info.format == "i") {
    dtype = DataType::DT_INT32;
  } else if (info.format == "l") {
    dtype = DataType::DT_INT64;
  }

  size_t num_samples = info.shape[0];
  ParallelTensor parallel_batch_tensor;
  model.get_parallel_tensor_from_tensor(batch_tensor, parallel_batch_tensor);
  return new SingleDataLoader(
      model, parallel_batch_tensor, info.ptr, num_samples, dtype);
}

SingleDataLoader *create_data_loader_attach(FFModel &model,
                                            Tensor &batch_tensor,
                                            py::array &full_array) {
  py::buffer_info info = full_array.request();
  DataType dtype;
  if (info.format == "f") {
    dtype = DataType::DT_FLOAT;
  } else if (info.format == "i") {
    dtype = DataType::DT_INT32;
  } else if (info.format == "l") {
    dtype = DataType::DT_INT64;
  }

  int num_dims = info.shape.size();
  ParallelDim dims[MAX_TENSOR_DIM];
  for (int i = 0; i < num_dims; i++) {
    dims[i].size = info.shape[i];
  }
  ParallelTensor full_tensor;
  if (num_dims == 2) {
    full_tensor = model.create_parallel_tensor<2>(dims, dtype);
  } else if (num_dims == 3) {
    full_tensor = model.create_parallel_tensor<3>(dims, dtype);
  } else if (num_dims == 4) {
    full_tensor = model.create_parallel_tensor<4>(dims, dtype);
#if MAX_TENSOR_DIM >= 5
  } else if (num_dims == 5) {
    full_tensor = model.create_parallel_tensor<5>(dims, dtype);
#endif
  } else {
    assert(0);
  }
  model.map_tensor(full_tensor, NULL /*parallel_op*/);
  ssize_t num_samples = info.shape[0];
  full_tensor->attach_raw_ptr(model.config, info.ptr, true);
  ParallelTensor parallel_batch_tensor;
  model.get_parallel_tensor_from_tensor(batch_tensor, parallel_batch_tensor);
  SingleDataLoader *dataloader = new SingleDataLoader(
      model, parallel_batch_tensor, full_tensor, num_samples, dtype);
  full_tensor->detach_raw_ptr(model.config);
  return dataloader;
}

SingleDataLoader *create_data_loader(FFModel &model,
                                     Tensor batch_tensor,
                                     py::array &full_array) {
  if (model.config.enable_control_replication) {
    assert(model.config.python_data_loader_type != 1);
    return create_data_loader_ptr(model, batch_tensor, full_array);
  } else {
    if (model.config.python_data_loader_type == 1) {
      return create_data_loader_attach(model, batch_tensor, full_array);
    } else {
      return create_data_loader_ptr(model, batch_tensor, full_array);
    }
  }
}

Tensor concat(FFModel &model,
              std::vector<Tensor> const &tensors,
              int axis,
              char const *name) {
  int size = tensors.size();
  return model.concat(size, tensors.data(), axis, name);
}

std::vector<Tensor> split(FFModel &model,
                          Tensor const &input,
                          std::vector<int> const &split,
                          int axis,
                          char const *name) {
  std::vector<Tensor> outputs;
  outputs.resize(split.size());
  model.split(input, outputs.data(), split, axis, name);
  return outputs;
}

} // namespace

PYBIND11_MODULE(flexflow_pybind11_internal, m) {
  m.attr("cuda_enabled") = true;

  m.def("begin_flexflow_task", &begin_flexflow_task);
  m.def("finish_flexflow_task", &finish_flexflow_task);

  py::enum_<ActiMode>(m, "ActiMode")
      .value("AC_MODE_NONE", ActiMode::AC_MODE_NONE)
      .value("AC_MODE_RELU", ActiMode::AC_MODE_RELU);

  py::enum_<CompMode>(m, "CompMode")
      .value("TRAINING", CompMode::COMP_MODE_TRAINING)
      .value("INFERENCE", CompMode::COMP_MODE_INFERENCE);

  py::enum_<DataType>(m, "DataType")
      .value("DT_FLOAT", DataType::DT_FLOAT)
      .value("DT_DOUBLE", DataType::DT_DOUBLE)
      .value("DT_INT32", DataType::DT_INT32)
      .value("DT_INT64", DataType::DT_INT64)
      .value("DT_BOOLEAN", DataType::DT_BOOLEAN);

  py::class_<Initializer>(m, "Initializer");

  py::class_<GlorotUniform, Initializer>(m, "GlorotUniformInitializer")
      .def(py::init<int>(), "seed"_a);

  py::class_<UniformInitializer, Initializer>(m, "UniformInitializer")
      .def(py::init<int, float, float>(), "seed"_a, "min"_a, "max"_a);

  py::class_<ZeroInitializer, Initializer>(m, "ZeroInitializer")
      .def(py::init());

  py::class_<Op>(m, "Op")
      .def_readonly("num_weights", &Op::numWeights)
      .def("get_parameter_by_id", [](Op &op, int id) { return op.weights[id]; })
      .def("get_input_tensor_by_id",
           [](Op &op, int id) { return op.inputs[id]; });

  py::class_<Optimizer>(m, "Optimizer");

  py::class_<SGDOptimizer, Optimizer>(m, "SGDOptimizer")
      .def(py::init<FFModel const *, double, double, bool, double>(),
           "model"_a,
           "lr"_a = 0.01f,
           "momentum"_a = 0.0f,
           "nesterov"_a = false,
           "weight_decay"_a = 0.0f)
      .def("set_learning_rate",
           [](SGDOptimizer &optimizer, double lr) { optimizer.lr = lr; });

  py::class_<AdamOptimizer, Optimizer>(m, "AdamOptimizer")
      .def(py::init<FFModel const *, double, double, double, double, double>(),
           "model"_a,
           "alpha"_a = 0.001f,
           "beta1"_a = 0.9f,
           "beta2"_a = 0.999f,
           "weight_decay"_a = 0.0f,
           "epsilon"_a = 1e-8)
      .def("set_learning_rate",
           [](AdamOptimizer &optimizer, double lr) { optimizer.alpha = lr; });

  py::class_<NetConfig>(m, "NetConfig")
      .def(py::init())
      .def_readonly("dataset_path", &NetConfig::dataset_path);

  py::class_<SingleDataLoader>(m, "SingleDataLoader")
      .def(py::init<FFModel &, ParallelTensor, ParallelTensor, int, DataType>(),
           "ffmodel"_a,
           "input"_a,
           "full_input"_a,
           "num_samples"_a,
           "data_type"_a)
      .def_readonly("num_samples", &SingleDataLoader::num_samples)
      .def("reset", &SingleDataLoader::reset)
      .def("next_batch", &SingleDataLoader::next_batch);

  py::class_<TensorBase>(m, "TensorBase")
      .def_readonly("data_type", &TensorBase::data_type)
      .def_property_readonly("dims",
                             [](TensorBase &t) {
                               std::vector<int> dims(t.num_dims);
                               for (int i = 0; i < t.num_dims; i++) {
                                 dims[i] = t.dims[i];
                               }
                               std::reverse(dims.begin(), dims.end());
                               return dims;
                             })
      .def_readonly("num_dims", &TensorBase::num_dims)
      //.def("inline_map", [](TensorBase &t, FFConfig &config) {
      // t.inline_map(config); }) .def("inline_unmap", [](TensorBase &t,
      // FFConfig &config) { t.inline_unmap(config); })
      .def("get_array", &get_array, py::return_value_policy::move)
      .def("set_tensor", &set_tensor, "ffmodel"_a, "np_array"_a)
      .def(
          "attach_numpy_array", &attach_numpy_array, "ffconfig"_a, "np_array"_a)
      .def("detach_numpy_array", &detach_numpy_array, "ffconfig"_a);

  // py::class_<Tensor, TensorBase* >(m, "Tensor");
  // py::class_<Tensor>(m, "Tensor");
  py::class_<Parameter>(m, "Parameter")
      .def("_get_weights", &get_weights, "ffmodel"_a, "full_array"_a)
      .def("_set_weights", &set_weights, "ffmodel"_a, "dims"_a, "full_array"_a);

  py::class_<FFConfig>(m, "FFConfig")
      .def(py::init())
      .def_readonly("batch_size", &FFConfig::batchSize)
      .def_readonly("epochs", &FFConfig::epochs)
      .def_readonly("num_nodes", &FFConfig::numNodes)
      .def_readonly("workers_per_node", &FFConfig::workersPerNode)
      .def("begin_trace",
           [](FFConfig &config, int trace_id) {
             config.lg_hlr->begin_trace(config.lg_ctx, trace_id);
           })
      .def("end_trace",
           [](FFConfig &config, int trace_id) {
             config.lg_hlr->end_trace(config.lg_ctx, trace_id);
           })
      .def("get_current_time", &get_current_time);

  py::enum_<LossType>(m, "LossType")
      .value("LOSS_CATEGORICAL_CROSSENTROPY",
             LossType::LOSS_CATEGORICAL_CROSSENTROPY)
      .value("LOSS_SPARSE_CATEGORICAL_CROSSENTROPY",
             LossType::LOSS_SPARSE_CATEGORICAL_CROSSENTROPY)
      .value("LOSS_MEAN_SQUARED_ERROR_AVG_REDUCE",
             LossType::LOSS_MEAN_SQUARED_ERROR_AVG_REDUCE)
      .value("LOSS_MEAN_SQUARED_ERROR_SUM_REDUCE",
             LossType::LOSS_MEAN_SQUARED_ERROR_SUM_REDUCE);

  py::enum_<MetricsType>(m, "MetricsType")
      .value("METRICS_ACCURACY", MetricsType::METRICS_ACCURACY)
      .value("METRICS_CATEGORICAL_CROSSENTROPY",
             MetricsType::METRICS_CATEGORICAL_CROSSENTROPY)
      .value("METRICS_SPARSE_CATEGORICAL_CROSSENTROPY",
             MetricsType::METRICS_SPARSE_CATEGORICAL_CROSSENTROPY)
      .value("METRICS_MEAN_SQUARED_ERROR",
             MetricsType::METRICS_MEAN_SQUARED_ERROR)
      .value("METRICS_ROOT_MEAN_SQUARED_ERROR",
             MetricsType::METRICS_ROOT_MEAN_SQUARED_ERROR)
      .value("METRICS_MEAN_ABSOLUTE_ERROR",
             MetricsType::METRICS_MEAN_ABSOLUTE_ERROR);

  py::enum_<PoolType>(m, "PoolType")
      .value("POOL_MAX", PoolType::POOL_MAX)
      .value("POOL_AVG", PoolType::POOL_AVG);

  py::enum_<ParameterSyncType>(m, "ParameterSyncType")
      .value("NONE", ParameterSyncType::NONE)
      .value("PS", ParameterSyncType::PS)
      .value("NCCL", ParameterSyncType::NCCL);

  py::class_<PerfMetrics>(m, "PerfMetrics")
      .def("get_accuracy", [](PerfMetrics &m) {
        return m.train_correct * 100.0f / m.train_all;
      });

  py::class_<FFModel>(m, "FFModel")
      .def(py::init<FFConfig &>())
      .def_readonly("label_tensor", &FFModel::label_tensor)
      .def_readwrite("optimizer", &FFModel::optimizer)
      .def("_compile",
           static_cast<void (FFModel::*)(
               LossType, std::vector<MetricsType> const &, CompMode)>(
               &FFModel::compile),
           "loss_type"_a,
           "metrics"_a,
           "comp_mode"_a)
      .def("create_data_loader",
           &create_data_loader,
           "batch_tensor"_a,
           "full_array"_a)
      .def("create_tensor",
           &create_tensor,
           "dims"_a,
           "data_type"_a,
           "create_grad"_a = true)
      .def("get_layer_by_id", [](FFModel &m, int id) { return m.layers[id]; })
      .def("get_last_layer", [](FFModel &m) { return m.layers.back(); })
      .def("get_perf_metrics",
           [](FFModel &m) {
             return m.current_metrics.get_result<PerfMetrics>();
           })
      //.def("init_layers", &FFModel::init_layers)
      .def("reset_metrics", &FFModel::reset_metrics)
      .def("compute_metrics", &FFModel::compute_metrics)
      // Training
      .def("forward", &FFModel::forward, "seq_length"_a = -1)
      .def("zero_gradients", &FFModel::zero_gradients)
      .def("backward", &FFModel::backward, "seq_length"_a = -1)
      .def("update", &FFModel::update)
      // Arithmetic operators
      .def("exp", &FFModel::exp, "x"_a, "name"_a = nullptr)
      .def("add",
           &FFModel::add,
           "x"_a,
           "y"_a,
           "inplace_a"_a = false,
           "name"_a = nullptr)
      .def("subtract",
           &FFModel::subtract,
           "x"_a,
           "y"_a,
           "inplace_a"_a = false,
           "name"_a = nullptr)
      .def("multiply",
           &FFModel::multiply,
           "x"_a,
           "y"_a,
           "inplace_a"_a = false,
           "name"_a = nullptr)
      .def("divide",
           &FFModel::divide,
           "x"_a,
           "y"_a,
           "inplace_a"_a = false,
           "name"_a = nullptr)
      // Scalar arithmetic operators
      .def("scalar_multiply",
           &FFModel::scalar_multiply,
           "x"_a,
           "scalar"_a,
           "inplace"_a = true,
           "name"_a = nullptr)
      .def("scalar_add",
           &FFModel::scalar_add,
           "x"_a,
           "scalar"_a,
           "inplace"_a = true,
           "name"_a = nullptr)
      .def("scalar_sub",
           &FFModel::scalar_add,
           "x"_a,
           "scalar"_a,
           "inplace"_a = true,
           "name"_a = nullptr)
      .def("scalar_truediv",
           &FFModel::scalar_add,
           "x"_a,
           "scalar"_a,
           "inplace"_a = true,
           "name"_a = nullptr)
      // Activations
      .def(
          "relu", &FFModel::relu, "x"_a, "inplace"_a = true, "name"_a = nullptr)
      .def("identity", &FFModel::identity, "x"_a, "name"_a = nullptr)
      .def("gelu", &FFModel::identity, "x"_a, "name"_a = nullptr)
      .def("sigmoid", &FFModel::sigmoid, "x"_a, "name"_a = nullptr)
      .def("tanh", &FFModel::tanh, "x"_a, "name"_a = nullptr)
      .def("elu", &FFModel::elu, "x"_a, "inplace"_a = true, "name"_a = nullptr)
      // Layers
      .def("embedding",
           &FFModel::embedding,
           "input"_a,
           "num_embeddings"_a,
           "embedding_dim"_a,
           "aggr"_a,
           "datatype"_a,
           "shared_op"_a = nullptr,
           "kernel_initializer"_a = nullptr,
           "name"_a = nullptr)
      .def("conv2d",
           &FFModel::conv2d,
           "input"_a,
           "out_channels"_a,
           "kernel_h"_a,
           "kernel_w"_a,
           "stride_h"_a,
           "stride_w"_a,
           "padding_h"_a,
           "padding_w"_a,
           "activation"_a = ActiMode::AC_MODE_NONE,
           "groups"_a = 1,
           "use_bias"_a = true,
           "shared_op"_a = nullptr,
           "kernel_initializer"_a = nullptr,
           "bias_initializer"_a = nullptr,
           "name"_a = nullptr)
      .def("dropout",
           &FFModel::dropout,
           "input"_a,
           "rate"_a,
           "seed"_a = 0,
           "name"_a = nullptr)
      .def("pool2d",
           &FFModel::pool2d,
           "input"_a,
           "kernel_h"_a,
           "kernel_w"_a,
           "stride_h"_a,
           "stride_w"_a,
           "padding_h"_a,
           "padding_w"_a,
           "pool_type"_a = PoolType::POOL_MAX,
           "activation"_a = ActiMode::AC_MODE_NONE,
           "name"_a = nullptr)
      .def("batch_norm",
           &FFModel::batch_norm,
           "input"_a,
           "relu"_a = true,
           "name"_a = nullptr)
      .def("layer_norm",
           &FFModel::layer_norm,
           "input"_a,
           "axes"_a,
           "elementwise_affine"_a,
           "eps"_a,
           "name"_a = nullptr)
      .def("batch_matmul",
           &FFModel::batch_matmul,
           "A"_a,
           "B"_a,
           "a_seq_length_dim"_a = -1,
           "b_seq_length_dim"_a = -1,
           "name"_a = nullptr)
      .def("dense",
           &FFModel::dense,
           "input"_a,
           "out_dim"_a,
           "activation"_a = ActiMode::AC_MODE_NONE,
           "use_bias"_a = true,
           "data_type"_a = DataType::DT_FLOAT,
           "shared_op"_a = nullptr,
           "kernel_initializer"_a = nullptr,
           "bias_initializer"_a = nullptr,
           "name"_a = nullptr)
      .def("flat", &FFModel::flat, "input"_a, "name"_a = nullptr)
      .def("softmax",
           &FFModel::softmax,
           "input"_a,
           "axis"_a = -1,
           "name"_a = nullptr)
      .def("transpose",
           &FFModel::transpose,
           "input"_a,
           "perm"_a,
           "name"_a = nullptr)
      .def("reshape",
           &FFModel::reshape,
           "input"_a,
           "shape"_a,
           "name"_a = nullptr)
      .def(
          "reverse", &FFModel::reverse, "input"_a, "axis"_a, "name"_a = nullptr)
      .def("multihead_attention",
           &FFModel::multihead_attention,
           "query"_a,
           "key"_a,
           "value"_a,
           "embed_dim"_a,
           "num_heads"_a,
           "kdim"_a = 0,
           "vdim"_a = 0,
           "dropout"_a = 0.0f,
           "bias"_a = true,
           "add_bias_k"_a = false,
           "add_zero_attn"_a = false,
           "kernel_initializer"_a = nullptr,
           "name"_a = nullptr)
      .def("concat", &concat, "tensors"_a, "axis"_a, "name"_a = nullptr)
      .def("split", &split, "input"_a, "split"_a, "axis"_a, "name"_a = nullptr)
      // Others
      .def("print_layers", &FFModel::print_layers, "id"_a = -1);
}
