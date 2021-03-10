#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "config.h"
#include "ffconst.h"
#include "metrics_functions.h"
#include "model.h"
#include "tensor.h"
#include "python/flexflow_dataloader.h"

namespace py = pybind11;
using py::literals::operator""_a;

namespace {

double get_current_time(FFConfig &config)
{
  config.lg_hlr->issue_execution_fence(config.lg_ctx);
  TimingLauncher timer(MEASURE_MICRO_SECONDS);
  Future future = config.lg_hlr->issue_timing_measurement(config.lg_ctx, timer);
  future.get_void_result();
  double ts_start = Realm::Clock::current_time_in_microseconds();
  return ts_start;
}


Tensor *create_tensor(FFModel &model, const std::vector<int> &dims, DataType data_type, bool create_grad)
{
  Tensor *tensor = new Tensor();
  int num_dims = dims.size();
  if (num_dims == 2) {
    *tensor = model.create_tensor<2>(dims.data(), data_type, NULL, create_grad);
  } else if (num_dims == 3) {
    *tensor = model.create_tensor<3>(dims.data(), data_type, NULL, create_grad);
  } else if (num_dims == 4) {
    *tensor = model.create_tensor<4>(dims.data(), data_type, NULL, create_grad);
#if MAX_TENSOR_DIM >= 5
  } else if (num_dims == 5) {
     *tensor = model.create_tensor<5>(dims.data(), data_type, NULL, create_grad);
#endif
  } else {
    assert(0);
  }
  return tensor;
}

SingleDataLoader *create_data_loader(FFModel &model, Tensor &batch_tensor, py::array full_array) {
  py::buffer_info info = full_array.request();
  DataType dtype;
  if (info.format == "f") {
    dtype = DataType::DT_FLOAT;
  } else if (info.format == "i") {
    dtype = DataType::DT_INT32;
  }

  ssize_t num_samples = info.shape[0];
  return new SingleDataLoader(model, batch_tensor, info.ptr, num_samples, dtype);
}

void fit(FFModel &model, SingleDataLoader &x, SingleDataLoader &y, int epochs)
{
  int num_samples = y.num_samples;
  int batch_size = model.config.batchSize;
  static int tracing_id = 200;
  tracing_id ++;
  for (int epoch = 0; epoch < epochs; epoch++) {
    x.reset();
    y.reset();
    model.reset_metrics();
    int iterations = num_samples / batch_size;
    for (int iter = 0; iter < iterations; iter++) {
      x.next_batch(model);
      y.next_batch(model);
      model.config.lg_hlr->begin_trace(model.config.lg_ctx, tracing_id);
      model.forward();
      model.zero_gradients();
      model.backward();
      model.update();
      model.config.lg_hlr->end_trace(model.config.lg_ctx, tracing_id);
    }
  }
}

}

PYBIND11_MODULE(flexflow_bindings, m) {
  m.attr("cuda_enabled") =
      true;

  py::enum_<ActiMode>(m, "ActiMode")
      .value("AC_MODE_NONE", ActiMode::AC_MODE_NONE)
      .value("AC_MODE_RELU", ActiMode::AC_MODE_RELU);

  py::class_<Initializer>(m, "Initializer");

  py::class_<GlorotUniform, Initializer>(m, "GlorotUniformInitializer")
      .def(py::init<int>(), "seed"_a);

  py::class_<UniformInitializer, Initializer>(m, "UniformInitializer")
      .def(py::init<int, float, float>(), "seed"_a, "min"_a, "max"_a);

  py::class_<ZeroInitializer, Initializer>(m, "ZeroInitializer")
      .def(py::init());

  py::class_<Op>(m, "Op");

  py::class_<Optimizer>(m, "Optimizer");

  py::class_<SGDOptimizer, Optimizer>(m, "SGDOptimizer")
      .def(py::init<const FFModel*, double, double, bool, double>(), "model"_a, "lr"_a = 0.01f, "momentum"_a = 0.0f, "nesterov"_a = false, "weight_decay"_a = 0.0f);

  py::class_<NetConfig>(m, "NetConfig")
      .def(py::init())
      .def_readonly("dataset_path", &NetConfig::dataset_path);

  py::class_<SingleDataLoader>(m, "SingleDataLoader")
      .def_readonly("num_samples", &SingleDataLoader::num_samples)
      .def("reset", &SingleDataLoader::reset)
      .def("next_batch", &SingleDataLoader::next_batch);

  py::class_<Tensor>(m, "Tensor")
      .def_property_readonly("dims", [](Tensor &t) { return std::vector<int>(t.adim, &t.adim[t.numDim]); });

  py::class_<FFConfig>(m, "FFConfig")
      .def(py::init())
      .def_readonly("batch_size", &FFConfig::batchSize)
      .def_readonly("epochs", &FFConfig::epochs)
      .def_readonly("num_nodes", &FFConfig::numNodes)
      .def_readonly("workers_per_node", &FFConfig::workersPerNode)
      .def("begin_trace", [](FFConfig &config, int trace_id) { config.lg_hlr->begin_trace(config.lg_ctx, trace_id); })
      .def("end_trace", [](FFConfig &config, int trace_id) { config.lg_hlr->end_trace(config.lg_ctx, trace_id); })
      .def("get_current_time", &get_current_time);

  py::enum_<CompMode>(m, "CompMode")
      .value("TRAINING", CompMode::COMP_MODE_TRAINING)
      .value("INFERENCE", CompMode::COMP_MODE_INFERENCE);

  py::enum_<DataType>(m, "DataType")
      .value("DT_FLOAT", DataType::DT_FLOAT)
      .value("DT_DOUBLE", DataType::DT_DOUBLE)
      .value("DT_INT32", DataType::DT_INT32)
      .value("DT_INT64", DataType::DT_INT64)
      .value("DT_BOOLEAN", DataType::DT_BOOLEAN);

  py::enum_<LossType>(m, "LossType")
      .value("LOSS_CATEGORICAL_CROSSENTROPY", LossType::LOSS_CATEGORICAL_CROSSENTROPY)
      .value("LOSS_SPARSE_CATEGORICAL_CROSSENTROPY", LossType::LOSS_SPARSE_CATEGORICAL_CROSSENTROPY)
      .value("LOSS_MEAN_SQUARED_ERROR_AVG_REDUCE", LossType::LOSS_MEAN_SQUARED_ERROR_AVG_REDUCE)
      .value("LOSS_MEAN_SQUARED_ERROR_SUM_REDUCE", LossType::LOSS_MEAN_SQUARED_ERROR_SUM_REDUCE);

  py::enum_<MetricsType>(m, "MetricsType")
      .value("METRICS_ACCURACY", MetricsType::METRICS_ACCURACY)
      .value("METRICS_CATEGORICAL_CROSSENTROPY", MetricsType::METRICS_CATEGORICAL_CROSSENTROPY)
      .value("METRICS_SPARSE_CATEGORICAL_CROSSENTROPY", MetricsType::METRICS_SPARSE_CATEGORICAL_CROSSENTROPY)
      .value("METRICS_MEAN_SQUARED_ERROR", MetricsType::METRICS_MEAN_SQUARED_ERROR)
      .value("METRICS_ROOT_MEAN_SQUARED_ERROR", MetricsType::METRICS_ROOT_MEAN_SQUARED_ERROR)
      .value("METRICS_MEAN_ABSOLUTE_ERROR", MetricsType::METRICS_MEAN_ABSOLUTE_ERROR);

  py::enum_<PoolType>(m, "PoolType")
      .value("POOL_MAX", PoolType::POOL_MAX)
      .value("POOL_AVG", PoolType::POOL_AVG);

  py::class_<PerfMetrics>(m, "PerfMetrics")
      .def("get_accuracy", [](PerfMetrics &m){ return m.train_correct * 100.0f / m.train_all; });

  py::class_<FFModel>(m, "FFModel")
      .def(py::init<FFConfig &>())
      .def_readonly("label_tensor", &FFModel::label_tensor)
      .def_readwrite("optimizer", &FFModel::optimizer)
      .def("compile", static_cast<void (FFModel::*)(LossType, const std::vector<MetricsType>&, CompMode)>(&FFModel::compile), "loss_type"_a, "metrics"_a, "comp_mode"_a = CompMode::COMP_MODE_TRAINING)
      .def("create_data_loader", &create_data_loader, "batch_tensor"_a, "full_array"_a)
      .def("create_tensor", &create_tensor, "dims"_a, "data_type"_a, "create_grad"_a = true)
      .def("get_perf_metrics", [](FFModel &m) { return m.current_metrics.get_result<PerfMetrics>(); })
      .def("init_layers", &FFModel::init_layers)
      .def("reset_metrics", &FFModel::reset_metrics)
      // Training
      .def("fit", &fit, "x"_a, "y"_a, "epochs"_a = 1)
      .def("forward", &FFModel::forward)
      .def("zero_gradients", &FFModel::zero_gradients)
      .def("backward", &FFModel::backward)
      .def("update", &FFModel::update)
      // Arithmetic operators
      .def("exp", &FFModel::exp, "x"_a, "name"_a = nullptr)
      .def("add", &FFModel::add, "x"_a, "y"_a, "name"_a = nullptr)
      .def("subtract", &FFModel::subtract, "x"_a, "y"_a, "name"_a = nullptr)
      .def("multiply", &FFModel::multiply, "x"_a, "y"_a, "name"_a = nullptr)
      .def("divide", &FFModel::divide, "x"_a, "y"_a, "name"_a = nullptr)
      // Activations
      .def("relu", &FFModel::relu, "x"_a, "name"_a = nullptr)
      .def("sigmoid", &FFModel::sigmoid, "x"_a, "name"_a = nullptr)
      .def("tanh", &FFModel::tanh, "x"_a, "name"_a = nullptr)
      .def("elu", &FFModel::elu, "x"_a, "name"_a = nullptr)
      // Layers
      .def("conv2d", &FFModel::conv2d, "input"_a, "out_channels"_a, "kernel_h"_a, "kernel_w"_a, "stride_h"_a, "stride_w"_a, "padding_h"_a, "padding_w"_a, "activation"_a = ActiMode::AC_MODE_NONE, "groups"_a = 1, "use_bias"_a = true, "shared_op"_a = nullptr, "kernel_initializer"_a = nullptr, "bias_initializer"_a = nullptr, "name"_a = nullptr)
      .def("dropout", &FFModel::dropout, "input"_a, "rate"_a, "seed"_a = 0, "name"_a = nullptr)
      .def("pool2d", &FFModel::pool2d, "input"_a, "kernel_h"_a, "kernel_w"_a, "stride_h"_a, "stride_w"_a, "padding_h"_a, "padding_w"_a, "pool_type"_a = PoolType::POOL_MAX, "activation"_a = ActiMode::AC_MODE_NONE, "name"_a = nullptr)
      .def("batch_norm", &FFModel::batch_norm, "input"_a, "relu"_a = true, "name"_a = nullptr)
      .def("batch_matmul", &FFModel::batch_matmul, "A"_a, "B"_a, "a_seq_length_dim"_a = -1, "b_seq_length_dim"_a = -1)
      .def("dense", &FFModel::dense, "input"_a, "out_dim"_a, "activation"_a = ActiMode::AC_MODE_NONE, "use_bias"_a = true, "shared_op"_a = nullptr, "kernel_initializer"_a = nullptr, "bias_initializer"_a = nullptr, "name"_a = nullptr)
      .def("flat", &FFModel::flat, "input"_a, "name"_a = nullptr)
      .def("softmax", &FFModel::softmax, "input"_a, "name"_a = nullptr)
      .def("transpose", &FFModel::transpose, "input"_a, "perm"_a, "name"_a = nullptr)
      .def("reshape", &FFModel::reshape, "input"_a, "shape"_a, "name"_a = nullptr)
      .def("reverse", &FFModel::reverse, "input"_a, "axis"_a, "name"_a = nullptr)
      .def("multihead_attention", &FFModel::multihead_attention, "query"_a, "key"_a, "value"_a, "embed_dim"_a, "num_heads"_a, "kdim"_a = 0, "vdim"_a = 0, "dropout"_a = 0.0f, "bias"_a = true, "add_bias_k"_a = false, "add_zero_attn"_a = false, "kernel_initializer"_a = nullptr, "name"_a = nullptr);

}
