/* Copyright 2022 NVIDIA CORPORATION
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

#include "gtest/gtest.h"

#include "onnx_parser.h"
#include "operators/binary.h"
#include "operators/conv2d.h"
#include "operators/pool2d.h"
#include "operators/softmax.h"
#include "operators/unary.h"

namespace {

namespace tbl = triton::backend::legion;

#define CHECK_GENERAL_OPERATOR_ATTRIBUTES(                                \
    op__, op_type__, model__, strategy__, num_input__, num_weight__,      \
    num_output__)                                                         \
  do {                                                                    \
    EXPECT_EQ(op__->op_type, op_type__);                                  \
    EXPECT_TRUE(                                                          \
        op__->model == reinterpret_cast<tbl::LegionModelState*>(model__)) \
        << "Expect: " << model__ << "; Got: " << op__->model;             \
    EXPECT_TRUE(op__->strategy == strategy__)                             \
        << "Expect: " << strategy__ << "; Got: " << op__->strategy;       \
    EXPECT_EQ(op__->num_inputs, num_input__);                             \
    EXPECT_EQ(op__->num_weights, num_weight__);                           \
    EXPECT_EQ(op__->num_outputs, num_output__);                           \
  } while (false)

#define CHECK_GENERAL_TENSOR_ATTRIBUTES(                                     \
    t__, owner__, is_weight__, dtype__, dims_vec__)                          \
  do {                                                                       \
    if (is_weight__) {                                                       \
      EXPECT_TRUE(dynamic_cast<tbl::Weights*>(t__) != nullptr)               \
          << "Expect tensor to be a Weights instance";                       \
    } else {                                                                 \
      EXPECT_FALSE(dynamic_cast<tbl::Weights*>(t__) != nullptr)              \
          << "Expect tensor not to be a Weights instance";                   \
    }                                                                        \
    EXPECT_TRUE(t__->owner == owner__)                                       \
        << ((owner__ == nullptr) ? "Expect tensor not owned by the operator" \
                                 : "Expect tensor owned by the operator");   \
    EXPECT_EQ(t__->type, dtype__);                                           \
    EXPECT_EQ(t__->bounds, dims_vec__);                                      \
  } while (false)

class OnnxParserSingleNodeSingleProcessorTest : public ::testing::Test {
 public:
  OnnxParserSingleNodeSingleProcessorTest()
      : layer_strategy_(0, 0, nullptr), local_cpus_(1)
  {
    // Set up layer strategy and function for finding local processor
    // (LegionTritonRuntime::FindLocalProcessors).
    // For tests in this fixture there should be only one CPU processor
    // which is "local" to the machine.
    // Note that the layer strategy doesn't specify dims as it depends on
    // the operator it describes, the test should set it properly or have
    // a fake operator implementation that is indenpendent to LayerStrategy::dim
    local_cpus_[0].id = 1;
    find_local_processor_fn_ = [this](Realm::Processor::Kind kind)
        -> const std::vector<Realm::Processor>& {
      switch (kind) {
        case Realm::Processor::LOC_PROC:
          return local_cpus_;
        case Realm::Processor::TOC_PROC:
          return local_cpus_;
        default:
          throw std::invalid_argument("Unknown processor kind");
      }
      return local_cpus_;
    };
    layer_strategy_.kind = Realm::Processor::LOC_PROC;
    layer_strategy_.nProcs = 1;
    layer_strategy_.local_processors[0] = local_cpus_[0];
    layer_strategy_.global_processors.push_back(local_cpus_[0]);
  }

  tbl::LayerStrategy layer_strategy_;
  std::function<const std::vector<Realm::Processor>&(Realm::Processor::Kind)>
      find_local_processor_fn_;
  std::vector<Realm::Processor> local_gpus_;
  std::vector<Realm::Processor> local_cpus_;
};

TEST_F(OnnxParserSingleNodeSingleProcessorTest, ParseAdd)
{
  std::vector<tbl::Tensor*> model_stub;
  std::vector<std::pair<std::string, tbl::Tensor*>> inputs;
  std::vector<std::pair<std::string, tbl::Tensor*>> outputs;
  std::vector<tbl::Operator*> layers;
  tbl::PartitionStrategy strategy(
      reinterpret_cast<tbl::LegionModelState*>(&model_stub),
      {reinterpret_cast<const tbl::LayerStrategy*>(&layer_strategy_)});

  auto err = tbl::OnnxParser::LoadModel(
      find_local_processor_fn_,
      reinterpret_cast<tbl::LegionModelState*>(&model_stub), &strategy,
      "data/add.onnx", &inputs, &outputs, &layers);
  ASSERT_TRUE(err == nullptr) << TRITONSERVER_ErrorMessage(err);

  ASSERT_EQ(model_stub.size(), 3) << "Expect 3 tensors are parsed";
  ASSERT_EQ(layers.size(), 1) << "Expect 1 layer is parsed";

  auto generated_op = dynamic_cast<tbl::BinaryOperator*>(layers[0]);
  ASSERT_TRUE(generated_op != nullptr)
      << "Expect the operator to be a Binary instance";

  CHECK_GENERAL_OPERATOR_ATTRIBUTES(
      generated_op, tbl::OperatorType::OP_EW_ADD, &model_stub, &layer_strategy_,
      2, 0, 1);

  ASSERT_EQ(inputs.size(), 2) << "Expect 2 inputs are parsed";
  for (size_t i = 0; i <= 1; i++) {
    ASSERT_TRUE(inputs[i].second == model_stub[i]);
    CHECK_GENERAL_TENSOR_ATTRIBUTES(
        inputs[i].second, nullptr, false, tbl::DataType::DT_FLOAT,
        std::vector<size_t>({4, 2}));
  }

  auto output = model_stub[2];
  CHECK_GENERAL_TENSOR_ATTRIBUTES(
      output, generated_op, false, tbl::DataType::DT_FLOAT,
      std::vector<size_t>({4, 2}));
}

TEST_F(OnnxParserSingleNodeSingleProcessorTest, ParseSub)
{
  std::vector<tbl::Tensor*> model_stub;
  std::vector<std::pair<std::string, tbl::Tensor*>> inputs;
  std::vector<std::pair<std::string, tbl::Tensor*>> outputs;
  std::vector<tbl::Operator*> layers;
  tbl::PartitionStrategy strategy(
      reinterpret_cast<tbl::LegionModelState*>(&model_stub),
      {reinterpret_cast<const tbl::LayerStrategy*>(&layer_strategy_)});

  auto err = tbl::OnnxParser::LoadModel(
      find_local_processor_fn_,
      reinterpret_cast<tbl::LegionModelState*>(&model_stub), &strategy,
      "data/sub.onnx", &inputs, &outputs, &layers);
  ASSERT_TRUE(err == nullptr) << TRITONSERVER_ErrorMessage(err);

  ASSERT_EQ(model_stub.size(), 3) << "Expect 3 tensors are parsed";
  ASSERT_EQ(layers.size(), 1) << "Expect 1 layer is parsed";

  auto generated_op = dynamic_cast<tbl::BinaryOperator*>(layers[0]);
  ASSERT_TRUE(generated_op != nullptr)
      << "Expect the operator to be a Binary instance";

  CHECK_GENERAL_OPERATOR_ATTRIBUTES(
      generated_op, tbl::OperatorType::OP_EW_SUB, &model_stub, &layer_strategy_,
      2, 0, 1);

  ASSERT_EQ(inputs.size(), 2) << "Expect 2 inputs are parsed";
  for (size_t i = 0; i <= 1; i++) {
    ASSERT_TRUE(inputs[i].second == model_stub[i]);
    CHECK_GENERAL_TENSOR_ATTRIBUTES(
        inputs[i].second, nullptr, false, tbl::DataType::DT_FLOAT,
        std::vector<size_t>({4, 2}));
  }

  auto output = model_stub[2];
  CHECK_GENERAL_TENSOR_ATTRIBUTES(
      output, generated_op, false, tbl::DataType::DT_FLOAT,
      std::vector<size_t>({4, 2}));
}

TEST_F(OnnxParserSingleNodeSingleProcessorTest, ParseMul)
{
  std::vector<tbl::Tensor*> model_stub;
  std::vector<std::pair<std::string, tbl::Tensor*>> inputs;
  std::vector<std::pair<std::string, tbl::Tensor*>> outputs;
  std::vector<tbl::Operator*> layers;
  tbl::PartitionStrategy strategy(
      reinterpret_cast<tbl::LegionModelState*>(&model_stub),
      {reinterpret_cast<const tbl::LayerStrategy*>(&layer_strategy_)});

  auto err = tbl::OnnxParser::LoadModel(
      find_local_processor_fn_,
      reinterpret_cast<tbl::LegionModelState*>(&model_stub), &strategy,
      "data/mul.onnx", &inputs, &outputs, &layers);
  ASSERT_TRUE(err == nullptr) << TRITONSERVER_ErrorMessage(err);

  ASSERT_EQ(model_stub.size(), 3) << "Expect 3 tensors are parsed";
  ASSERT_EQ(layers.size(), 1) << "Expect 1 layer is parsed";

  auto generated_op = dynamic_cast<tbl::BinaryOperator*>(layers[0]);
  ASSERT_TRUE(generated_op != nullptr)
      << "Expect the operator to be a Binary instance";

  CHECK_GENERAL_OPERATOR_ATTRIBUTES(
      generated_op, tbl::OperatorType::OP_EW_MUL, &model_stub, &layer_strategy_,
      2, 0, 1);

  ASSERT_EQ(inputs.size(), 2) << "Expect 2 inputs are parsed";
  for (size_t i = 0; i <= 1; i++) {
    ASSERT_TRUE(inputs[i].second == model_stub[i]);
    CHECK_GENERAL_TENSOR_ATTRIBUTES(
        inputs[i].second, nullptr, false, tbl::DataType::DT_FLOAT,
        std::vector<size_t>({4, 2}));
  }

  auto output = model_stub[2];
  CHECK_GENERAL_TENSOR_ATTRIBUTES(
      output, generated_op, false, tbl::DataType::DT_FLOAT,
      std::vector<size_t>({4, 2}));
}

TEST_F(OnnxParserSingleNodeSingleProcessorTest, ParseAvgPool)
{
  std::vector<tbl::Tensor*> model_stub;
  tbl::PartitionStrategy strategy(
      reinterpret_cast<tbl::LegionModelState*>(&model_stub),
      {reinterpret_cast<const tbl::LayerStrategy*>(&layer_strategy_)});
  std::vector<std::pair<std::string, tbl::Tensor*>> inputs;
  std::vector<std::pair<std::string, tbl::Tensor*>> outputs;
  std::vector<tbl::Operator*> layers;

  auto err = tbl::OnnxParser::LoadModel(
      find_local_processor_fn_,
      reinterpret_cast<tbl::LegionModelState*>(&model_stub), &strategy,
      "data/avg_pool.onnx", &inputs, &outputs, &layers);
  ASSERT_TRUE(err == nullptr) << TRITONSERVER_ErrorMessage(err);

  ASSERT_EQ(model_stub.size(), 2) << "Expect 2 tensors are parsed";
  ASSERT_EQ(layers.size(), 1) << "Expect 1 layer is parsed";
  auto generated_op = dynamic_cast<tbl::Pool2D*>(layers[0]);
  ASSERT_TRUE(generated_op != nullptr)
      << "Expect the operator to be a Pool2D instance";

  CHECK_GENERAL_OPERATOR_ATTRIBUTES(
      generated_op, tbl::OperatorType::OP_POOL2D, &model_stub, &layer_strategy_,
      1, 0, 1);
  EXPECT_EQ(generated_op->activation, tbl::ActivationMode::AC_MODE_NONE);
  EXPECT_EQ(generated_op->pool_type, tbl::PoolType::POOL_AVG);
  EXPECT_EQ(generated_op->kernel_h, 2);
  EXPECT_EQ(generated_op->kernel_w, 2);
  EXPECT_EQ(generated_op->stride_h, 1);
  EXPECT_EQ(generated_op->stride_w, 1);
  EXPECT_EQ(generated_op->padding_h, 0);
  EXPECT_EQ(generated_op->padding_w, 0);

  // Check associated tensors
  ASSERT_EQ(inputs.size(), 1) << "Expect 1 input is parsed";
  ASSERT_TRUE(inputs[0].second == model_stub[0]);
  CHECK_GENERAL_TENSOR_ATTRIBUTES(
      inputs[0].second, nullptr, false, tbl::DataType::DT_FLOAT,
      std::vector<size_t>({1, 3, 30, 30}));

  auto output = model_stub[1];
  CHECK_GENERAL_TENSOR_ATTRIBUTES(
      output, generated_op, false, tbl::DataType::DT_FLOAT,
      std::vector<size_t>({1, 3, 29, 29}));
}

TEST_F(OnnxParserSingleNodeSingleProcessorTest, ParseAvgPoolAutoPad)
{
  std::vector<tbl::Tensor*> model_stub;
  tbl::PartitionStrategy strategy(
      reinterpret_cast<tbl::LegionModelState*>(&model_stub),
      {reinterpret_cast<const tbl::LayerStrategy*>(&layer_strategy_)});
  std::vector<std::pair<std::string, tbl::Tensor*>> inputs;
  std::vector<std::pair<std::string, tbl::Tensor*>> outputs;
  std::vector<tbl::Operator*> layers;

  auto err = tbl::OnnxParser::LoadModel(
      find_local_processor_fn_,
      reinterpret_cast<tbl::LegionModelState*>(&model_stub), &strategy,
      "data/avg_pool_autopad.onnx", &inputs, &outputs, &layers);
  auto expected_err = std::string(
      "Unsupported attribute value 'SAME_LOWER' for attribute 'auto_pad' in "
      "'AveragePool' layer named '', currently supported value is 'NOTSET'");
  ASSERT_TRUE(err != nullptr) << "Unexpected successful model load";
  ASSERT_TRUE(TRITONSERVER_ERROR_UNSUPPORTED == TRITONSERVER_ErrorCode(err))
      << "Wrong error type" << std::endl
      << "Actual error type: " << TRITONSERVER_ErrorCodeString(err) << std::endl
      << "Expected error type: "
      << "Unsupported";
  ASSERT_TRUE(expected_err == TRITONSERVER_ErrorMessage(err))
      << "Wrong error message" << std::endl
      << "Actual error message: " << TRITONSERVER_ErrorMessage(err) << std::endl
      << "Expected error message: " << expected_err;
}

TEST_F(OnnxParserSingleNodeSingleProcessorTest, ParseAvgPoolCeil)
{
  std::vector<tbl::Tensor*> model_stub;
  tbl::PartitionStrategy strategy(
      reinterpret_cast<tbl::LegionModelState*>(&model_stub),
      {reinterpret_cast<const tbl::LayerStrategy*>(&layer_strategy_)});
  std::vector<std::pair<std::string, tbl::Tensor*>> inputs;
  std::vector<std::pair<std::string, tbl::Tensor*>> outputs;
  std::vector<tbl::Operator*> layers;

  auto err = tbl::OnnxParser::LoadModel(
      find_local_processor_fn_,
      reinterpret_cast<tbl::LegionModelState*>(&model_stub), &strategy,
      "data/avg_pool_ceil.onnx", &inputs, &outputs, &layers);
  auto expected_err = std::string(
      "Unsupported attribute value for attribute 'ceil_mode' in 'AveragePool' "
      "layer named '', currently supported value is 0");
  ASSERT_TRUE(err != nullptr) << "Unexpected successful model load";
  ASSERT_TRUE(TRITONSERVER_ERROR_UNSUPPORTED == TRITONSERVER_ErrorCode(err))
      << "Wrong error type" << std::endl
      << "Actual error type: " << TRITONSERVER_ErrorCodeString(err) << std::endl
      << "Expected error type: "
      << "Unsupported";
  ASSERT_TRUE(expected_err == TRITONSERVER_ErrorMessage(err))
      << "Wrong error message" << std::endl
      << "Actual error message: " << TRITONSERVER_ErrorMessage(err) << std::endl
      << "Expected error message: " << expected_err;
}

TEST_F(OnnxParserSingleNodeSingleProcessorTest, ParseAvgPoolCountIncludePad)
{
  std::vector<tbl::Tensor*> model_stub;
  tbl::PartitionStrategy strategy(
      reinterpret_cast<tbl::LegionModelState*>(&model_stub),
      {reinterpret_cast<const tbl::LayerStrategy*>(&layer_strategy_)});
  std::vector<std::pair<std::string, tbl::Tensor*>> inputs;
  std::vector<std::pair<std::string, tbl::Tensor*>> outputs;
  std::vector<tbl::Operator*> layers;

  auto err = tbl::OnnxParser::LoadModel(
      find_local_processor_fn_,
      reinterpret_cast<tbl::LegionModelState*>(&model_stub), &strategy,
      "data/avg_pool_count_include_pad.onnx", &inputs, &outputs, &layers);
  auto expected_err = std::string(
      "Unsupported attribute value for attribute 'count_include_pad' in "
      "'AveragePool' layer named '', currently supported value is 0");
  ASSERT_TRUE(err != nullptr) << "Unexpected successful model load";
  ASSERT_TRUE(TRITONSERVER_ERROR_UNSUPPORTED == TRITONSERVER_ErrorCode(err))
      << "Wrong error type" << std::endl
      << "Actual error type: " << TRITONSERVER_ErrorCodeString(err) << std::endl
      << "Expected error type: "
      << "Unsupported";
  ASSERT_TRUE(expected_err == TRITONSERVER_ErrorMessage(err))
      << "Wrong error message" << std::endl
      << "Actual error message: " << TRITONSERVER_ErrorMessage(err) << std::endl
      << "Expected error message: " << expected_err;
}

TEST_F(OnnxParserSingleNodeSingleProcessorTest, ParseAvgPoolPad)
{
  std::vector<tbl::Tensor*> model_stub;
  tbl::PartitionStrategy strategy(
      reinterpret_cast<tbl::LegionModelState*>(&model_stub),
      {reinterpret_cast<const tbl::LayerStrategy*>(&layer_strategy_)});
  std::vector<std::pair<std::string, tbl::Tensor*>> inputs;
  std::vector<std::pair<std::string, tbl::Tensor*>> outputs;
  std::vector<tbl::Operator*> layers;

  auto err = tbl::OnnxParser::LoadModel(
      find_local_processor_fn_,
      reinterpret_cast<tbl::LegionModelState*>(&model_stub), &strategy,
      "data/avg_pool_pad.onnx", &inputs, &outputs, &layers);
  ASSERT_TRUE(err == nullptr) << TRITONSERVER_ErrorMessage(err);

  ASSERT_EQ(model_stub.size(), 2) << "Expect 2 tensors are parsed";
  ASSERT_EQ(layers.size(), 1) << "Expect 1 layer is parsed";
  auto generated_op = dynamic_cast<tbl::Pool2D*>(layers[0]);
  ASSERT_TRUE(generated_op != nullptr)
      << "Expect the operator to be a Pool2D instance";

  CHECK_GENERAL_OPERATOR_ATTRIBUTES(
      generated_op, tbl::OperatorType::OP_POOL2D, &model_stub, &layer_strategy_,
      1, 0, 1);
  EXPECT_EQ(generated_op->activation, tbl::ActivationMode::AC_MODE_NONE);
  EXPECT_EQ(generated_op->pool_type, tbl::PoolType::POOL_AVG);
  EXPECT_EQ(generated_op->kernel_h, 2);
  EXPECT_EQ(generated_op->kernel_w, 2);
  EXPECT_EQ(generated_op->stride_h, 3);
  EXPECT_EQ(generated_op->stride_w, 3);
  EXPECT_EQ(generated_op->padding_h, 1);
  EXPECT_EQ(generated_op->padding_w, 1);

  // Check associated tensors
  ASSERT_EQ(inputs.size(), 1) << "Expect 1 input is parsed";
  ASSERT_TRUE(inputs[0].second == model_stub[0]);
  CHECK_GENERAL_TENSOR_ATTRIBUTES(
      inputs[0].second, nullptr, false, tbl::DataType::DT_FLOAT,
      std::vector<size_t>({1, 3, 30, 30}));

  auto output = model_stub[1];
  CHECK_GENERAL_TENSOR_ATTRIBUTES(
      output, generated_op, false, tbl::DataType::DT_FLOAT,
      std::vector<size_t>({1, 3, 11, 11}));
}

TEST_F(OnnxParserSingleNodeSingleProcessorTest, ParseCast)
{
  std::vector<tbl::Tensor*> model_stub;
  tbl::PartitionStrategy strategy(
      reinterpret_cast<tbl::LegionModelState*>(&model_stub),
      {reinterpret_cast<const tbl::LayerStrategy*>(&layer_strategy_)});
  std::vector<std::pair<std::string, tbl::Tensor*>> inputs;
  std::vector<std::pair<std::string, tbl::Tensor*>> outputs;
  std::vector<tbl::Operator*> layers;

  auto err = tbl::OnnxParser::LoadModel(
      find_local_processor_fn_,
      reinterpret_cast<tbl::LegionModelState*>(&model_stub), &strategy,
      "data/cast.onnx", &inputs, &outputs, &layers);
  ASSERT_TRUE(err == nullptr) << TRITONSERVER_ErrorMessage(err);

  ASSERT_EQ(model_stub.size(), 2) << "Expect 2 tensors are parsed";
  ASSERT_EQ(layers.size(), 1) << "Expect 1 layer is parsed";
  auto generated_op = dynamic_cast<tbl::UnaryOperator*>(layers[0]);
  ASSERT_TRUE(generated_op != nullptr)
      << "Expect the operator to be a UnaryOperator instance";

  CHECK_GENERAL_OPERATOR_ATTRIBUTES(
      generated_op, tbl::OperatorType::OP_CAST, &model_stub, &layer_strategy_,
      1, 0, 1);
  EXPECT_EQ(generated_op->scalar_type, tbl::DataType::DT_FLOAT);
  EXPECT_EQ(generated_op->inplace, false);

  // Check associated tensors
  ASSERT_EQ(inputs.size(), 1) << "Expect 1 input is parsed";
  ASSERT_TRUE(inputs[0].second == model_stub[0]);
  CHECK_GENERAL_TENSOR_ATTRIBUTES(
      inputs[0].second, nullptr, false, tbl::DataType::DT_FLOAT,
      std::vector<size_t>({1, 3}));
  auto output = model_stub[1];
  CHECK_GENERAL_TENSOR_ATTRIBUTES(
      output, generated_op, false, tbl::DataType::DT_DOUBLE,
      std::vector<size_t>({1, 3}));
}

TEST_F(OnnxParserSingleNodeSingleProcessorTest, ParseConv2D)
{
  // Data section
  std::vector<float> bias_data = {0, 1};
  std::vector<std::vector<std::vector<std::vector<float>>>> weight_data = {
      {{{0, 1, 2}, {3, 4, 5}, {6, 7, 8}}},
      {{{9, 10, 11}, {12, 13, 14}, {15, 16, 17}}}};
  std::vector<tbl::Tensor*> model_stub;
  tbl::PartitionStrategy strategy(
      reinterpret_cast<tbl::LegionModelState*>(&model_stub),
      {reinterpret_cast<const tbl::LayerStrategy*>(&layer_strategy_)});
  std::vector<std::pair<std::string, tbl::Tensor*>> inputs;
  std::vector<std::pair<std::string, tbl::Tensor*>> outputs;
  std::vector<tbl::Operator*> layers;

  auto err = tbl::OnnxParser::LoadModel(
      find_local_processor_fn_,
      reinterpret_cast<tbl::LegionModelState*>(&model_stub), &strategy,
      "data/conv2d_with_bias.onnx", &inputs, &outputs, &layers);
  ASSERT_TRUE(err == nullptr) << TRITONSERVER_ErrorMessage(err);

  ASSERT_EQ(model_stub.size(), 4) << "Expect 4 tensors are parsed";
  ASSERT_EQ(layers.size(), 1) << "Expect 1 layer is parsed";
  auto generated_op = dynamic_cast<tbl::Conv2D*>(layers[0]);
  ASSERT_TRUE(generated_op != nullptr)
      << "Expect the operator to be a Conv2D instance";

  CHECK_GENERAL_OPERATOR_ATTRIBUTES(
      generated_op, tbl::OperatorType::OP_CONV2D, &model_stub, &layer_strategy_,
      1, 2, 1);
  // [gluo FIXME] expected value are set based on knowledge of the model file,
  // should add sanity check before running test that the files are intact
  // Check op specific attributes
  EXPECT_EQ(generated_op->activation, tbl::ActivationMode::AC_MODE_NONE);
  EXPECT_EQ(generated_op->in_channels, 1);
  EXPECT_EQ(generated_op->out_channels, 2);
  EXPECT_EQ(generated_op->kernel_h, 3);
  EXPECT_EQ(generated_op->kernel_w, 3);
  EXPECT_EQ(generated_op->stride_h, 1);
  EXPECT_EQ(generated_op->stride_w, 1);
  EXPECT_EQ(generated_op->padding_h, 0);
  EXPECT_EQ(generated_op->padding_w, 0);
  EXPECT_EQ(generated_op->groups, 1);
  EXPECT_EQ(generated_op->use_bias, true);

  // Check associated tensors
  ASSERT_EQ(inputs.size(), 1) << "Expect 1 input is parsed";
  ASSERT_TRUE(inputs[0].second == model_stub[0]);
  CHECK_GENERAL_TENSOR_ATTRIBUTES(
      inputs[0].second, nullptr, false, tbl::DataType::DT_FLOAT,
      std::vector<size_t>({4, 1, 5, 5}));

  {
    auto weight = model_stub[1];
    CHECK_GENERAL_TENSOR_ATTRIBUTES(
        weight, generated_op, true, tbl::DataType::DT_FLOAT,
        std::vector<size_t>({2, 1, 3, 3}));
    auto bound =
        generated_op->GetWeightBounds(layer_strategy_.local_processors[0]);
    const float* data_allocation = reinterpret_cast<const float*>(
        dynamic_cast<tbl::Weights*>(weight)->local_allocation[0]);
    for (size_t oc = bound.lo[0]; oc <= bound.hi[0]; ++oc) {
      for (size_t ic = bound.lo[1]; ic <= bound.hi[1]; ++ic) {
        for (size_t kh = bound.lo[2]; kh <= bound.hi[2]; ++kh) {
          for (size_t kw = bound.lo[3]; kw <= bound.hi[3]; ++kw) {
            EXPECT_EQ(weight_data[oc][ic][kh][kw], *data_allocation)
                << "Mismatched value at weight entry (" << oc << ", " << ic
                << ", " << kh << ", " << kw << ")";
            ++data_allocation;
          }
        }
      }
    }
  }

  {
    auto bias = model_stub[2];
    CHECK_GENERAL_TENSOR_ATTRIBUTES(
        bias, generated_op, true, tbl::DataType::DT_FLOAT,
        std::vector<size_t>({2}));
    auto bound =
        generated_op->GetBiasBounds(layer_strategy_.local_processors[0]);
    const float* data_allocation = reinterpret_cast<const float*>(
        dynamic_cast<tbl::Weights*>(bias)->local_allocation[0]);
    for (size_t idx = bound.lo[0]; idx <= bound.hi[0]; ++idx) {
      EXPECT_EQ(bias_data[idx], *data_allocation)
          << "Mismatched value at weight entry (" << idx << ")";
      ++data_allocation;
    }
  }

  auto output = model_stub[3];
  CHECK_GENERAL_TENSOR_ATTRIBUTES(
      output, generated_op, false, tbl::DataType::DT_FLOAT,
      std::vector<size_t>({4, 2, 3, 3}));
}

TEST_F(OnnxParserSingleNodeSingleProcessorTest, ParseIdentity)
{
  std::vector<tbl::Tensor*> model_stub;
  tbl::PartitionStrategy strategy(
      reinterpret_cast<tbl::LegionModelState*>(&model_stub),
      {reinterpret_cast<const tbl::LayerStrategy*>(&layer_strategy_)});
  std::vector<std::pair<std::string, tbl::Tensor*>> inputs;
  std::vector<std::pair<std::string, tbl::Tensor*>> outputs;
  std::vector<tbl::Operator*> layers;

  auto err = tbl::OnnxParser::LoadModel(
      find_local_processor_fn_,
      reinterpret_cast<tbl::LegionModelState*>(&model_stub), &strategy,
      "data/identity.onnx", &inputs, &outputs, &layers);
  ASSERT_TRUE(err == nullptr) << TRITONSERVER_ErrorMessage(err);

  ASSERT_EQ(model_stub.size(), 2) << "Expect 2 tensors are parsed";
  ASSERT_EQ(layers.size(), 1) << "Expect 1 layer is parsed";
  auto generated_op = dynamic_cast<tbl::UnaryOperator*>(layers[0]);
  ASSERT_TRUE(generated_op != nullptr)
      << "Expect the operator to be a UnaryOperator instance";

  CHECK_GENERAL_OPERATOR_ATTRIBUTES(
      generated_op, tbl::OperatorType::OP_IDENTITY, &model_stub,
      &layer_strategy_, 1, 0, 1);
  // [gluo FIXME] expected value are set based on knowledge of the model file,
  // should add sanity check before running test that the files are intact
  // Check op specific attributes
  EXPECT_EQ(generated_op->scalar_type, tbl::DataType::DT_FLOAT);
  EXPECT_EQ(generated_op->inplace, false);

  // Check associated tensors
  ASSERT_EQ(inputs.size(), 1) << "Expect 1 input is parsed";
  ASSERT_TRUE(inputs[0].second == model_stub[0]);
  CHECK_GENERAL_TENSOR_ATTRIBUTES(
      inputs[0].second, nullptr, false, tbl::DataType::DT_FLOAT,
      std::vector<size_t>({4, 1, 5, 5}));
  auto output = model_stub[1];
  CHECK_GENERAL_TENSOR_ATTRIBUTES(
      output, generated_op, false, tbl::DataType::DT_FLOAT,
      std::vector<size_t>({4, 1, 5, 5}));
}

TEST_F(OnnxParserSingleNodeSingleProcessorTest, ParseMaxPool)
{
  std::vector<tbl::Tensor*> model_stub;
  tbl::PartitionStrategy strategy(
      reinterpret_cast<tbl::LegionModelState*>(&model_stub),
      {reinterpret_cast<const tbl::LayerStrategy*>(&layer_strategy_)});
  std::vector<std::pair<std::string, tbl::Tensor*>> inputs;
  std::vector<std::pair<std::string, tbl::Tensor*>> outputs;
  std::vector<tbl::Operator*> layers;

  auto err = tbl::OnnxParser::LoadModel(
      find_local_processor_fn_,
      reinterpret_cast<tbl::LegionModelState*>(&model_stub), &strategy,
      "data/max_pool.onnx", &inputs, &outputs, &layers);
  ASSERT_TRUE(err == nullptr) << TRITONSERVER_ErrorMessage(err);

  ASSERT_EQ(model_stub.size(), 2) << "Expect 2 tensors are parsed";
  ASSERT_EQ(layers.size(), 1) << "Expect 1 layer is parsed";
  auto generated_op = dynamic_cast<tbl::Pool2D*>(layers[0]);
  ASSERT_TRUE(generated_op != nullptr)
      << "Expect the operator to be a Pool2D instance";

  CHECK_GENERAL_OPERATOR_ATTRIBUTES(
      generated_op, tbl::OperatorType::OP_POOL2D, &model_stub, &layer_strategy_,
      1, 0, 1);
  EXPECT_EQ(generated_op->activation, tbl::ActivationMode::AC_MODE_NONE);
  EXPECT_EQ(generated_op->pool_type, tbl::PoolType::POOL_MAX);
  EXPECT_EQ(generated_op->kernel_h, 5);
  EXPECT_EQ(generated_op->kernel_w, 5);
  EXPECT_EQ(generated_op->stride_h, 2);
  EXPECT_EQ(generated_op->stride_w, 2);
  EXPECT_EQ(generated_op->padding_h, 2);
  EXPECT_EQ(generated_op->padding_w, 2);

  // Check associated tensors
  ASSERT_EQ(inputs.size(), 1) << "Expect 1 input is parsed";
  ASSERT_TRUE(inputs[0].second == model_stub[0]);
  CHECK_GENERAL_TENSOR_ATTRIBUTES(
      inputs[0].second, nullptr, false, tbl::DataType::DT_FLOAT,
      std::vector<size_t>({1, 1, 5, 5}));

  auto output = model_stub[1];
  CHECK_GENERAL_TENSOR_ATTRIBUTES(
      output, generated_op, false, tbl::DataType::DT_FLOAT,
      std::vector<size_t>({1, 1, 3, 3}));
}

TEST_F(OnnxParserSingleNodeSingleProcessorTest, ParseMaxPoolAutoPad)
{
  std::vector<tbl::Tensor*> model_stub;
  tbl::PartitionStrategy strategy(
      reinterpret_cast<tbl::LegionModelState*>(&model_stub),
      {reinterpret_cast<const tbl::LayerStrategy*>(&layer_strategy_)});
  std::vector<std::pair<std::string, tbl::Tensor*>> inputs;
  std::vector<std::pair<std::string, tbl::Tensor*>> outputs;
  std::vector<tbl::Operator*> layers;

  auto err = tbl::OnnxParser::LoadModel(
      find_local_processor_fn_,
      reinterpret_cast<tbl::LegionModelState*>(&model_stub), &strategy,
      "data/max_pool_autopad.onnx", &inputs, &outputs, &layers);
  auto expected_err = std::string(
      "Unsupported attribute value 'SAME_UPPER' for attribute 'auto_pad' in "
      "'MaxPool' "
      "layer named '', currently supported value is 'NOTSET'");
  ASSERT_TRUE(err != nullptr) << "Unexpected successful model load";
  ASSERT_TRUE(TRITONSERVER_ERROR_UNSUPPORTED == TRITONSERVER_ErrorCode(err))
      << "Wrong error type" << std::endl
      << "Actual error type: " << TRITONSERVER_ErrorCodeString(err) << std::endl
      << "Expected error type: "
      << "Unsupported";
  ASSERT_TRUE(expected_err == TRITONSERVER_ErrorMessage(err))
      << "Wrong error message" << std::endl
      << "Actual error message: " << TRITONSERVER_ErrorMessage(err) << std::endl
      << "Expected error message: " << expected_err;
}

TEST_F(OnnxParserSingleNodeSingleProcessorTest, ParseMaxPoolCeil)
{
  std::vector<tbl::Tensor*> model_stub;
  tbl::PartitionStrategy strategy(
      reinterpret_cast<tbl::LegionModelState*>(&model_stub),
      {reinterpret_cast<const tbl::LayerStrategy*>(&layer_strategy_)});
  std::vector<std::pair<std::string, tbl::Tensor*>> inputs;
  std::vector<std::pair<std::string, tbl::Tensor*>> outputs;
  std::vector<tbl::Operator*> layers;

  auto err = tbl::OnnxParser::LoadModel(
      find_local_processor_fn_,
      reinterpret_cast<tbl::LegionModelState*>(&model_stub), &strategy,
      "data/max_pool_ceil.onnx", &inputs, &outputs, &layers);
  auto expected_err = std::string(
      "Unsupported attribute value for attribute 'ceil_mode' in 'MaxPool' "
      "layer named '', currently supported value is 0");
  ASSERT_TRUE(err != nullptr) << "Unexpected successful model load";
  ASSERT_TRUE(TRITONSERVER_ERROR_UNSUPPORTED == TRITONSERVER_ErrorCode(err))
      << "Wrong error type" << std::endl
      << "Actual error type: " << TRITONSERVER_ErrorCodeString(err) << std::endl
      << "Expected error type: "
      << "Unsupported";
  ASSERT_TRUE(expected_err == TRITONSERVER_ErrorMessage(err))
      << "Wrong error message" << std::endl
      << "Actual error message: " << TRITONSERVER_ErrorMessage(err) << std::endl
      << "Expected error message: " << expected_err;
}

TEST_F(OnnxParserSingleNodeSingleProcessorTest, ParseMaxPoolDilations)
{
  std::vector<tbl::Tensor*> model_stub;
  tbl::PartitionStrategy strategy(
      reinterpret_cast<tbl::LegionModelState*>(&model_stub),
      {reinterpret_cast<const tbl::LayerStrategy*>(&layer_strategy_)});
  std::vector<std::pair<std::string, tbl::Tensor*>> inputs;
  std::vector<std::pair<std::string, tbl::Tensor*>> outputs;
  std::vector<tbl::Operator*> layers;

  auto err = tbl::OnnxParser::LoadModel(
      find_local_processor_fn_,
      reinterpret_cast<tbl::LegionModelState*>(&model_stub), &strategy,
      "data/max_pool_dilations.onnx", &inputs, &outputs, &layers);
  auto expected_err = std::string(
      "Unsupported attribute value for attribute 'dilations' in 'MaxPool' "
      "layer named '', each of the attribute value must be 1");
  ASSERT_TRUE(err != nullptr) << "Unexpected successful model load";
  ASSERT_TRUE(TRITONSERVER_ERROR_UNSUPPORTED == TRITONSERVER_ErrorCode(err))
      << "Wrong error type" << std::endl
      << "Actual error type: " << TRITONSERVER_ErrorCodeString(err) << std::endl
      << "Expected error type: "
      << "Unsupported";
  ASSERT_TRUE(expected_err == TRITONSERVER_ErrorMessage(err))
      << "Wrong error message" << std::endl
      << "Actual error message: " << TRITONSERVER_ErrorMessage(err) << std::endl
      << "Expected error message: " << expected_err;
}

TEST_F(OnnxParserSingleNodeSingleProcessorTest, ParseMaxPoolOrder)
{
  std::vector<tbl::Tensor*> model_stub;
  tbl::PartitionStrategy strategy(
      reinterpret_cast<tbl::LegionModelState*>(&model_stub),
      {reinterpret_cast<const tbl::LayerStrategy*>(&layer_strategy_)});
  std::vector<std::pair<std::string, tbl::Tensor*>> inputs;
  std::vector<std::pair<std::string, tbl::Tensor*>> outputs;
  std::vector<tbl::Operator*> layers;

  auto err = tbl::OnnxParser::LoadModel(
      find_local_processor_fn_,
      reinterpret_cast<tbl::LegionModelState*>(&model_stub), &strategy,
      "data/max_pool_order.onnx", &inputs, &outputs, &layers);
  auto expected_err = std::string(
      "Unsupported attribute value for attribute 'storage_order' in 'MaxPool' "
      "layer named '', currently supported value is 0");
  ASSERT_TRUE(err != nullptr) << "Unexpected successful model load";
  ASSERT_TRUE(TRITONSERVER_ERROR_UNSUPPORTED == TRITONSERVER_ErrorCode(err))
      << "Wrong error type" << std::endl
      << "Actual error type: " << TRITONSERVER_ErrorCodeString(err) << std::endl
      << "Expected error type: "
      << "Unsupported";
  ASSERT_TRUE(expected_err == TRITONSERVER_ErrorMessage(err))
      << "Wrong error message" << std::endl
      << "Actual error message: " << TRITONSERVER_ErrorMessage(err) << std::endl
      << "Expected error message: " << expected_err;
}

TEST_F(OnnxParserSingleNodeSingleProcessorTest, ParseReciprocal)
{
  std::vector<tbl::Tensor*> model_stub;
  tbl::PartitionStrategy strategy(
      reinterpret_cast<tbl::LegionModelState*>(&model_stub),
      {reinterpret_cast<const tbl::LayerStrategy*>(&layer_strategy_)});
  std::vector<std::pair<std::string, tbl::Tensor*>> inputs;
  std::vector<std::pair<std::string, tbl::Tensor*>> outputs;
  std::vector<tbl::Operator*> layers;

  auto err = tbl::OnnxParser::LoadModel(
      find_local_processor_fn_,
      reinterpret_cast<tbl::LegionModelState*>(&model_stub), &strategy,
      "data/reciprocal.onnx", &inputs, &outputs, &layers);
  ASSERT_TRUE(err == nullptr) << TRITONSERVER_ErrorMessage(err);

  ASSERT_EQ(model_stub.size(), 2) << "Expect 2 tensors are parsed";
  ASSERT_EQ(layers.size(), 1) << "Expect 1 layer is parsed";
  auto generated_op = dynamic_cast<tbl::UnaryOperator*>(layers[0]);
  ASSERT_TRUE(generated_op != nullptr)
      << "Expect the operator to be a UnaryOperator instance";

  CHECK_GENERAL_OPERATOR_ATTRIBUTES(
      generated_op, tbl::OperatorType::OP_RECIPROCAL, &model_stub,
      &layer_strategy_, 1, 0, 1);
  // [gluo FIXME] expected value are set based on knowledge of the model file,
  // should add sanity check before running test that the files are intact
  // Check op specific attributes
  EXPECT_EQ(generated_op->scalar_type, tbl::DataType::DT_FLOAT);
  EXPECT_EQ(generated_op->inplace, false);

  // Check associated tensors
  ASSERT_EQ(inputs.size(), 1) << "Expect 1 input is parsed";
  ASSERT_TRUE(inputs[0].second == model_stub[0]);
  CHECK_GENERAL_TENSOR_ATTRIBUTES(
      inputs[0].second, nullptr, false, tbl::DataType::DT_FLOAT,
      std::vector<size_t>({1, 3}));
  auto output = model_stub[1];
  CHECK_GENERAL_TENSOR_ATTRIBUTES(
      output, generated_op, false, tbl::DataType::DT_FLOAT,
      std::vector<size_t>({1, 3}));
}

TEST_F(OnnxParserSingleNodeSingleProcessorTest, ParseSoftmax)
{
  std::vector<tbl::Tensor*> model_stub;
  tbl::PartitionStrategy strategy(
      reinterpret_cast<tbl::LegionModelState*>(&model_stub),
      {reinterpret_cast<const tbl::LayerStrategy*>(&layer_strategy_)});
  std::vector<std::pair<std::string, tbl::Tensor*>> inputs;
  std::vector<std::pair<std::string, tbl::Tensor*>> outputs;
  std::vector<tbl::Operator*> layers;

  auto err = tbl::OnnxParser::LoadModel(
      find_local_processor_fn_,
      reinterpret_cast<tbl::LegionModelState*>(&model_stub), &strategy,
      "data/softmax.onnx", &inputs, &outputs, &layers);
  ASSERT_TRUE(err == nullptr) << TRITONSERVER_ErrorMessage(err);

  ASSERT_EQ(model_stub.size(), 2) << "Expect 2 tensors are parsed";
  ASSERT_EQ(layers.size(), 1) << "Expect 1 layer is parsed";
  auto generated_op = dynamic_cast<tbl::Softmax*>(layers[0]);
  ASSERT_TRUE(generated_op != nullptr)
      << "Expect the operator to be a Softmax instance";

  CHECK_GENERAL_OPERATOR_ATTRIBUTES(
      generated_op, tbl::OperatorType::OP_SOFTMAX, &model_stub,
      &layer_strategy_, 1, 0, 1);
  EXPECT_EQ(generated_op->dim, 0);

  // // Check associated tensors
  ASSERT_EQ(inputs.size(), 1) << "Expect 1 input is parsed";
  ASSERT_TRUE(inputs[0].second == model_stub[0]);
  CHECK_GENERAL_TENSOR_ATTRIBUTES(
      inputs[0].second, nullptr, false, tbl::DataType::DT_FLOAT,
      std::vector<size_t>({3, 1}));
  auto output = model_stub[1];
  CHECK_GENERAL_TENSOR_ATTRIBUTES(
      output, generated_op, false, tbl::DataType::DT_FLOAT,
      std::vector<size_t>({3, 1}));
}

TEST_F(OnnxParserSingleNodeSingleProcessorTest, ParseSoftmaxDefaultAxis)
{
  std::vector<tbl::Tensor*> model_stub;
  tbl::PartitionStrategy strategy(
      reinterpret_cast<tbl::LegionModelState*>(&model_stub),
      {reinterpret_cast<const tbl::LayerStrategy*>(&layer_strategy_)});
  std::vector<std::pair<std::string, tbl::Tensor*>> inputs;
  std::vector<std::pair<std::string, tbl::Tensor*>> outputs;
  std::vector<tbl::Operator*> layers;

  auto err = tbl::OnnxParser::LoadModel(
      find_local_processor_fn_,
      reinterpret_cast<tbl::LegionModelState*>(&model_stub), &strategy,
      "data/softmax_default_axis.onnx", &inputs, &outputs, &layers);
  ASSERT_TRUE(err == nullptr) << TRITONSERVER_ErrorMessage(err);

  auto generated_op = dynamic_cast<tbl::Softmax*>(layers[0]);
  ASSERT_TRUE(generated_op != nullptr)
      << "Expect the operator to be a Softmax instance";
  CHECK_GENERAL_OPERATOR_ATTRIBUTES(
      generated_op, tbl::OperatorType::OP_SOFTMAX, &model_stub,
      &layer_strategy_, 1, 0, 1);
  EXPECT_EQ(generated_op->dim, 1);
}

TEST_F(OnnxParserSingleNodeSingleProcessorTest, ParseSoftmaxNegativeAxis)
{
  std::vector<tbl::Tensor*> model_stub;
  tbl::PartitionStrategy strategy(
      reinterpret_cast<tbl::LegionModelState*>(&model_stub),
      {reinterpret_cast<const tbl::LayerStrategy*>(&layer_strategy_)});
  std::vector<std::pair<std::string, tbl::Tensor*>> inputs;
  std::vector<std::pair<std::string, tbl::Tensor*>> outputs;
  std::vector<tbl::Operator*> layers;

  auto err = tbl::OnnxParser::LoadModel(
      find_local_processor_fn_,
      reinterpret_cast<tbl::LegionModelState*>(&model_stub), &strategy,
      "data/softmax_negative_axis.onnx", &inputs, &outputs, &layers);
  ASSERT_TRUE(err == nullptr) << TRITONSERVER_ErrorMessage(err);

  auto generated_op = dynamic_cast<tbl::Softmax*>(layers[0]);
  ASSERT_TRUE(generated_op != nullptr)
      << "Expect the operator to be a Softmax instance";
  CHECK_GENERAL_OPERATOR_ATTRIBUTES(
      generated_op, tbl::OperatorType::OP_SOFTMAX, &model_stub,
      &layer_strategy_, 1, 0, 1);
  EXPECT_EQ(generated_op->dim, 0);
}

TEST_F(OnnxParserSingleNodeSingleProcessorTest, ParseSqrt)
{
  std::vector<tbl::Tensor*> model_stub;
  tbl::PartitionStrategy strategy(
      reinterpret_cast<tbl::LegionModelState*>(&model_stub),
      {reinterpret_cast<const tbl::LayerStrategy*>(&layer_strategy_)});
  std::vector<std::pair<std::string, tbl::Tensor*>> inputs;
  std::vector<std::pair<std::string, tbl::Tensor*>> outputs;
  std::vector<tbl::Operator*> layers;

  auto err = tbl::OnnxParser::LoadModel(
      find_local_processor_fn_,
      reinterpret_cast<tbl::LegionModelState*>(&model_stub), &strategy,
      "data/sqrt.onnx", &inputs, &outputs, &layers);
  ASSERT_TRUE(err == nullptr) << TRITONSERVER_ErrorMessage(err);

  ASSERT_EQ(model_stub.size(), 2) << "Expect 2 tensors are parsed";
  ASSERT_EQ(layers.size(), 1) << "Expect 1 layer is parsed";
  auto generated_op = dynamic_cast<tbl::UnaryOperator*>(layers[0]);
  ASSERT_TRUE(generated_op != nullptr)
      << "Expect the operator to be a UnaryOperator instance";

  CHECK_GENERAL_OPERATOR_ATTRIBUTES(
      generated_op, tbl::OperatorType::OP_SQRT, &model_stub, &layer_strategy_,
      1, 0, 1);
  EXPECT_EQ(generated_op->scalar_type, tbl::DataType::DT_FLOAT);
  EXPECT_EQ(generated_op->inplace, false);

  // Check associated tensors
  ASSERT_EQ(inputs.size(), 1) << "Expect 1 input is parsed";
  ASSERT_TRUE(inputs[0].second == model_stub[0]);
  CHECK_GENERAL_TENSOR_ATTRIBUTES(
      inputs[0].second, nullptr, false, tbl::DataType::DT_FLOAT,
      std::vector<size_t>({3, 1}));
  auto output = model_stub[1];
  CHECK_GENERAL_TENSOR_ATTRIBUTES(
      output, generated_op, false, tbl::DataType::DT_FLOAT,
      std::vector<size_t>({3, 1}));
}

TEST_F(OnnxParserSingleNodeSingleProcessorTest, ParseTanh)
{
  std::vector<tbl::Tensor*> model_stub;
  tbl::PartitionStrategy strategy(
      reinterpret_cast<tbl::LegionModelState*>(&model_stub),
      {reinterpret_cast<const tbl::LayerStrategy*>(&layer_strategy_)});
  std::vector<std::pair<std::string, tbl::Tensor*>> inputs;
  std::vector<std::pair<std::string, tbl::Tensor*>> outputs;
  std::vector<tbl::Operator*> layers;

  auto err = tbl::OnnxParser::LoadModel(
      find_local_processor_fn_,
      reinterpret_cast<tbl::LegionModelState*>(&model_stub), &strategy,
      "data/tanh.onnx", &inputs, &outputs, &layers);
  ASSERT_TRUE(err == nullptr) << TRITONSERVER_ErrorMessage(err);

  ASSERT_EQ(model_stub.size(), 2) << "Expect 2 tensors are parsed";
  ASSERT_EQ(layers.size(), 1) << "Expect 1 layer is parsed";
  auto generated_op = dynamic_cast<tbl::UnaryOperator*>(layers[0]);
  ASSERT_TRUE(generated_op != nullptr)
      << "Expect the operator to be a UnaryOperator instance";

  CHECK_GENERAL_OPERATOR_ATTRIBUTES(
      generated_op, tbl::OperatorType::OP_TANH, &model_stub, &layer_strategy_,
      1, 0, 1);
  EXPECT_EQ(generated_op->scalar_type, tbl::DataType::DT_FLOAT);
  EXPECT_EQ(generated_op->inplace, false);

  // Check associated tensors
  ASSERT_EQ(inputs.size(), 1) << "Expect 1 input is parsed";
  ASSERT_TRUE(inputs[0].second == model_stub[0]);
  CHECK_GENERAL_TENSOR_ATTRIBUTES(
      inputs[0].second, nullptr, false, tbl::DataType::DT_FLOAT,
      std::vector<size_t>({3, 1}));
  auto output = model_stub[1];
  CHECK_GENERAL_TENSOR_ATTRIBUTES(
      output, generated_op, false, tbl::DataType::DT_FLOAT,
      std::vector<size_t>({3, 1}));
}


}  // namespace

int
main(int argc, char** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
