#include <stdio.h>
#include <iostream>
#include <sstream>
#include <cstdlib>
#include <vector>
#include <assert.h>
#include <queue>
#include <map>
#include <set>
#include <cmath>
#include <time.h>
#include "cnn.h"

#define MAX_NUM_DIMS 4
#define MAX_NUM_WORKERS 64
#define MAX_NUM_PARTS 64
#define MAX_NUM_OPS 1000
#define MAX_NUM_CONFIGS 20
#define L2_FACTOR -0.0001
#define DATA_XFER_FACTOR 0.01
#define NOT_USE_SIMULATE_OPT
//#define VERBOSE

const int SRC_LENGTH = 5;
const int DST_LENGTH = 5;
const int LSTM_PER_NODE_LENGTH = 8;
const int NUM_LAYERS = 2;
const int VOCAB_SIZE = 32000;
const int LOCAL_BATCH_SIZE = 64;
const int EMBEDDING_SIZE = 1024;
const int HIDDEN_SIZE = 1024;
const int NUM_NODES = 2;
const int WORKERS_PER_NODE = 4;
const int NUM_WORKERS = NUM_NODES * WORKERS_PER_NODE; // NUM_WORKERS <= MAX_NUM_WORKERS
const int NUM_PARTITIONS = NUM_NODES * WORKERS_PER_NODE; // NUM_PARTITIONS <= MAX_NUM_PARTS
const int BATCH_SIZE = NUM_PARTITIONS * LOCAL_BATCH_SIZE;
const float INTRA_NODE_BANDWIDTH = 4 * 1024 * 1024;
const float CROSS_NODE_BANDWIDTH = 1 * 1024 * 1024;

using namespace std;

class OpConfig {
public:
  int nDims, nParts, dim[MAX_NUM_DIMS], map[MAX_NUM_PARTS];
};

struct Rect {
  int nDims, lo[MAX_NUM_DIMS], hi[MAX_NUM_DIMS];
};

class Op;

class Tensor {
public:
  Tensor(void)
  : nDims(0), owner(NULL), idx(-1) {}
  Tensor(int _nDims, int* _dim, Op* _owner, int _idx)
  : nDims(_nDims), owner(_owner), idx(_idx) {
    for (int i = 0; i < nDims; i++)
      dim[i] = _dim[i];
  }

  Tensor(int _nDims, int dim0, int dim1, Op* _owner, int _idx)
  : nDims(_nDims), owner(_owner), idx(_idx) {
    assert(nDims == 2);
    dim[0] = dim0; dim[1] = dim1;
  }

  Tensor(int _nDims, int dim0, int dim1, int dim2, int dim3, Op* _owner, int _idx)
  : nDims(_nDims), owner(_owner), idx(_idx) {
    assert(nDims == 4);
    dim[0] = dim0; dim[1] = dim1; dim[2] = dim2; dim[3] = dim3;
  }
public:
  Op* owner;
  int idx, nDims, dim[MAX_NUM_DIMS];
};

int op_global_guid = 0;
Op* guidToOp[MAX_NUM_OPS];
std::vector<std::vector<Op*> > parameters;

class Op {
public:
  Op(std::string _name)
  : name(_name) {
    guid = op_global_guid ++;
    numInputs = 0;
    assert(guid < MAX_NUM_OPS);
    guidToOp[guid] = this;
  }

  void add_input_tensor(Tensor x) {
    numInputs ++;
    inputTensors.push_back(x);
    if (x.owner != NULL) {
      preOps.push_back(x.owner);
      x.owner->nextOps.push_back(this);
      x.owner->nextOpTensors.push_back(x);
    }
  }

  virtual float compute(OpConfig c) = 0;

  virtual float update(const std::vector<OpConfig>& vec) = 0;

  virtual Rect get_tensor_shape(OpConfig c, int idx, Tensor t, bool is_input) = 0;
  //virtual float xfer(Tensor t, OpConfig config, OpConfig preConfig) = 0;
  virtual OpConfig get_random_config() = 0;
public:
  int guid, numInputs;
  std::string name;
  std::vector<Tensor> inputTensors, outputTensors, nextOpTensors;
  std::vector<Op*> nextOps, preOps;
};

float dpCompTime = 0.0f, mpCompTime = 0.0f, bestCompTime = 0.0f;
float totalDataXfer = 0.0f;

class Conv2D : public Op {
public:
  Conv2D(int _outputSize, int _kernelH, int _kernelW, int _strideH, int _strideW,
         int _paddingH, int _paddingW, Tensor x, std::string name)
  : outputSize(_outputSize), kernelH(_kernelH), kernelW(_kernelW), strideH(_strideH),
    strideW(_strideW), paddingH(_paddingH), paddingW(_paddingW), Op(name) {
    assert(x.nDims == 4);
    batchSize = x.dim[0];
    inputSize = x.dim[1];
    inputHeight = x.dim[2];
    inputWidth = x.dim[3];
    outputHeight = 1 + (inputHeight + 2 * paddingH - kernelH) / strideH;
    outputWidth = 1 + (inputWidth + 2 * paddingW - kernelW) / strideW;
    assert(outputHeight > 0);
    assert(outputWidth > 0);
    add_input_tensor(x);
    Tensor y(4, batchSize, outputSize, outputHeight, outputWidth, this, 0);
    printf("Conv2D(%s):	input[%d %d %d %d] output[%d %d %d %d] kernel(%d %d) stride(%d %d) padding(%d %d)\n",
           name.c_str(), batchSize, inputSize, inputHeight, inputWidth,
           batchSize, outputSize, outputHeight, outputWidth, kernelH, kernelW,
           strideH, strideW, paddingH, paddingW);
    outputTensors.push_back(y);
    nConfigs = 0;
    for (int i = 1; i <= NUM_PARTITIONS; i*=2)
      config_x[nConfigs++] = i;
    for (int i = 0; i < nConfigs; i++) {
      int b = batchSize / config_x[i];
      computeTime[i] = measure_conv2d_time(b, inputSize, inputHeight, inputWidth, 
                                           outputSize, outputHeight, outputWidth,
                                           kernelH, kernelW, strideH, strideW,
                                           paddingH, paddingW);
    }
    dpCompTime += computeTime[nConfigs-1] * config_x[nConfigs-1];
    mpCompTime += computeTime[0] * config_x[0];
    float bestV = computeTime[0] * config_x[0];
    for (int i = 0; i < nConfigs; i++)
      bestV = std::min(bestV, computeTime[i] * config_x[i]);
    bestCompTime += bestV;
  }
  float compute(OpConfig c);
  float update(const std::vector<OpConfig>& vec);
  Rect get_tensor_shape(OpConfig c, int idx, Tensor t, bool is_input);
  OpConfig get_random_config();
private:
  int batchSize, inputSize, outputSize, inputWidth, inputHeight, outputWidth, outputHeight;
  int kernelH, kernelW, strideH, strideW, paddingH, paddingW;
  int nConfigs, config_x[MAX_NUM_CONFIGS];
  float computeTime[MAX_NUM_CONFIGS];
};

float Conv2D::compute(OpConfig c) {
  assert(c.nDims == 1);
  int idx = 0;
  for (; idx < nConfigs; idx++)
    if (config_x[idx] == c.dim[0])
      break;
  assert(idx < nConfigs);
  return computeTime[idx];
};

float Conv2D::update(const std::vector<OpConfig>& vec) {
  int used[NUM_WORKERS];
  assert(vec.size() == 1);
  OpConfig config = vec[0];
  assert(config.nDims == 1);
  for (int i = 0; i < NUM_WORKERS; i++)
    used[i] = 0;
  assert(config.dim[0] == config.nParts);
  for (int i = 0; i < config.dim[0]; i++)
    used[config.map[i]] ++;
  int intra_cnt = 0, cross_cnt = 0, l2_cnt = 0;
  for (int i = 0; i < NUM_NODES; i++) {
    int cnt = 0;
    for (int j = 0; j < WORKERS_PER_NODE; j++) {
      if (used[i * WORKERS_PER_NODE + j] > 0) cnt++;
      l2_cnt += used[i * WORKERS_PER_NODE + j] * used[i * WORKERS_PER_NODE + j];
    }
    if (cnt > intra_cnt) intra_cnt = cnt;
    if (cnt > 0) cross_cnt ++;
  }
  float xfer = kernelH * kernelW * inputSize * outputSize * sizeof(float);
  //float ret = xfer * cnt / INTRA_NODE_BANDWIDTH + l2_cnt * L2_FACTOR;
  assert(cross_cnt > 0);
  assert(intra_cnt > 0);
  //printf("Conv2D::intra_cnt = %d cross_cnt = %d\n", intra_cnt, cross_cnt);
  //printf("Conv2D::intra_xfer = %.4lf cross_xfer = %.4lf\n", xfer * (intra_cnt-1) / INTRA_NODE_BANDWIDTH, xfer * (cross_cnt-1) / CROSS_NODE_BANDWIDTH);
  return xfer * (intra_cnt-1) / INTRA_NODE_BANDWIDTH + xfer * (cross_cnt-1) / CROSS_NODE_BANDWIDTH;
}

Rect Conv2D::get_tensor_shape(OpConfig config, int idx, Tensor t, bool is_input)
{
  assert(config.nDims == 1);
  assert(t.nDims == 4); // Assume 4-D tensors
  assert(idx < config.dim[0]);
  int extent = t.dim[0] / config.dim[0];
  Rect r;
  r.nDims = 4;
  r.lo[0] = extent * idx; r.hi[0] = extent * (idx + 1) - 1;
  r.lo[1] = 0; r.hi[1] = t.dim[1] - 1;
  r.lo[2] = 0; r.hi[2] = t.dim[2] - 1;
  r.lo[3] = 0; r.hi[3] = t.dim[3] - 1;
  return r;
}

OpConfig Conv2D::get_random_config()
{
  OpConfig config;
  config.nDims = 1;
  int idx = std::rand() % nConfigs;
  //int idx = nConfigs - 1;
  config.nParts = config_x[idx];
  config.dim[0] = config.nParts;
  for (int i = 0; i < config.nParts; i++)
    config.map[i] = std::rand() % NUM_WORKERS;
  return config;
}

class Pool2D : public Op {
public:
  Pool2D(int _kernelH, int _kernelW, int _strideH, int _strideW,
         int _paddingH, int _paddingW, Tensor x, std::string name)
  : kernelH(_kernelH), kernelW(_kernelW), strideH(_strideH), strideW(_strideW),
    paddingH(_paddingH), paddingW(_paddingW), Op(name) {
    assert(x.nDims == 4);
    batchSize = x.dim[0];
    outputSize = x.dim[1];
    inputHeight = x.dim[2];
    inputWidth = x.dim[3];
    outputHeight = 1 + (inputHeight + 2 * paddingH - kernelH) / strideH;
    outputWidth = 1 + (inputWidth + 2 * paddingW - kernelW) / strideW;
    assert(outputHeight > 0);
    assert(outputWidth > 0);
    add_input_tensor(x);
    Tensor y(4, batchSize, outputSize, outputHeight, outputWidth, this, 0);
    outputTensors.push_back(y);
    nConfigs = 0;
    for (int i = 1; i <= NUM_PARTITIONS; i*=2)
      config_x[nConfigs++] = i;
    for (int i = 0; i < nConfigs; i++) {
      int b = batchSize / config_x[i];
      computeTime[i] = measure_pool2d_time(b, outputSize, inputHeight, inputWidth,
                                           outputHeight, outputWidth,
                                           kernelH, kernelW, strideH, strideW,
                                           paddingH, paddingW);
    }
    dpCompTime += computeTime[nConfigs-1] * config_x[nConfigs-1];
    mpCompTime += computeTime[0] * config_x[0];
    float bestV = computeTime[0] * config_x[0];
    for (int i = 0; i < nConfigs; i++)
      bestV = std::min(bestV, computeTime[i] * config_x[i]);
    bestCompTime += bestV;
  }
  float compute(OpConfig c);
  float update(const std::vector<OpConfig>& vec);
  Rect get_tensor_shape(OpConfig c, int idx, Tensor t, bool is_input);
  OpConfig get_random_config();
private:
  int batchSize, inputSize, outputSize, inputWidth, inputHeight, outputWidth, outputHeight;
  int kernelH, kernelW, strideH, strideW, paddingH, paddingW;
  int nConfigs, config_x[MAX_NUM_CONFIGS];
  float computeTime[MAX_NUM_CONFIGS];
};

float Pool2D::compute(OpConfig c) {
  assert(c.nDims == 1);
  int idx = 0;
  for (; idx < nConfigs; idx++)
    if (config_x[idx] == c.dim[0])
      break;
  assert(idx < nConfigs);
  return computeTime[idx];
}

float Pool2D::update(const std::vector<OpConfig>& vec) {
  return 0;
}

Rect Pool2D::get_tensor_shape(OpConfig config, int idx, Tensor t, bool is_input)
{
  assert(config.nDims == 1);
  assert(t.nDims == 4); // Assume 4-D tensors
  assert(idx < config.dim[0]);
  int extent = t.dim[0] / config.dim[0];
  Rect r;
  r.nDims = 4;
  r.lo[0] = extent * idx; r.hi[0] = extent * (idx + 1) - 1;
  r.lo[1] = 0; r.hi[1] = t.dim[1] - 1;
  r.lo[2] = 0; r.hi[2] = t.dim[2] - 1;
  r.lo[3] = 0; r.hi[3] = t.dim[3] - 1;
  return r;
}

OpConfig Pool2D::get_random_config()
{
  OpConfig config;
  config.nDims = 1;
  int idx = std::rand() % nConfigs;
  //int idx = nConfigs - 1;
  config.nParts = config_x[idx];
  config.dim[0] = config.nParts;
  for (int i = 0; i < config.nParts; i++)
    config.map[i] = std::rand() % NUM_WORKERS;
  return config;
}

class Concat : public Op {
public:
  Concat(int num, Tensor* x, std::string name)
  : Op(name) {
    assert(num > 0);
    Tensor y = x[0];
    for (int i = 1; i < num; i++) {
      assert(y.nDims == x[i].nDims);
      assert(y.dim[0] == x[i].dim[0]);
      for (int j = 2; j < y.nDims; j++)
        assert(y.dim[j] == x[i].dim[j]);
      y.dim[1] += x[i].dim[1];
    }
    for (int i = 0; i < num; i++)
      add_input_tensor(x[i]);
    outputTensors.push_back(y);
    nConfigs = 0;
    for (int i = 1; i <= NUM_PARTITIONS; i*=2)
      config_x[nConfigs++] = i;
  }
  float compute(OpConfig c);
  float update(const std::vector<OpConfig>& vec);
  Rect get_tensor_shape(OpConfig c, int idx, Tensor t, bool is_input);
  OpConfig get_random_config();
private:
  int nConfigs, config_x[MAX_NUM_CONFIGS];
};

float Concat::compute(OpConfig c) {
  return 0.02 / c.nParts;
}

float Concat::update(const std::vector<OpConfig>& vec) {
  return 0;
}

Rect Concat::get_tensor_shape(OpConfig config, int idx, Tensor t, bool is_input) {
  assert(config.nDims == 1);
  assert(t.nDims == 4); // Assume 4-D tensors
  assert(idx < config.dim[0]);
  int extent = t.dim[0] / config.dim[0];
  Rect r;
  r.nDims = 4;
  r.lo[0] = extent * idx; r.hi[0] = extent * (idx + 1) - 1;
  r.lo[1] = 0; r.hi[1] = t.dim[1] - 1;
  r.lo[2] = 0; r.hi[2] = t.dim[2] - 1;
  r.lo[3] = 0; r.hi[3] = t.dim[3] - 1;
  return r;
}

OpConfig Concat::get_random_config()
{
  OpConfig config;
  config.nDims = 1;
  int idx = std::rand() % nConfigs;
  //int idx = nConfigs - 1;
  config.nParts = config_x[idx];
  config.dim[0] = config.nParts;
  for (int i = 0; i < config.nParts; i++)
    config.map[i] = std::rand() % NUM_WORKERS;
  return config;
}

class Flat : public Op {
public:
  Flat(Tensor x, std::string name) : Op(name) {
    assert(x.nDims == 4);
    outputSize = x.dim[1] * x.dim[2] * x.dim[3];
    add_input_tensor(x);
    Tensor y(2, x.dim[0], outputSize, this, 0);
    outputTensors.push_back(y);
    nConfigs = 0;
    for (int i = 1; i <= NUM_PARTITIONS; i*=2)
      config_x[nConfigs++] = i;
  }
  float compute(OpConfig c);
  float update(const std::vector<OpConfig>& vec);
  Rect get_tensor_shape(OpConfig c, int idx, Tensor t, bool is_input);
  OpConfig get_random_config();
private:
  int outputSize;
  int nConfigs, config_x[MAX_NUM_CONFIGS];
};

float Flat::compute(OpConfig c) {
  return 0;
}

float Flat::update(const std::vector<OpConfig>& vec) {
  return 0;
}

Rect Flat::get_tensor_shape(OpConfig config, int idx, Tensor t, bool is_input) {
  assert(config.nDims == 1);
  if (is_input) {
    assert(t.nDims == 4);
    assert(idx < config.dim[0]);
    int extent = t.dim[0] / config.dim[0];
    Rect r;
    r.nDims = 4;
    r.lo[0] = extent * idx; r.hi[0] = extent * (idx + 1) - 1;
    r.lo[1] = 0; r.hi[1] = t.dim[1] - 1;
    r.lo[2] = 0; r.hi[2] = t.dim[2] - 1;
    r.lo[3] = 0; r.hi[3] = t.dim[3] - 1;
    return r;
  } else {
    assert(t.nDims == 2);
    assert(idx < config.dim[0]);
    int extent = t.dim[0] / config.dim[0];
    Rect r;
    r.nDims = 2;
    r.lo[0] = extent * idx; r.hi[0] = extent * (idx + 1) - 1;
    r.lo[1] = 0; r.hi[1] = t.dim[1] - 1;
    return r;
  }
}

OpConfig Flat::get_random_config()
{
  OpConfig config;
  config.nDims = 1;
  int idx = std::rand() % nConfigs;
  //int idx = nConfigs - 1;
  config.nParts = config_x[idx];
  config.dim[0] = config.nParts;
  for (int i = 0; i < config.nParts; i++)
    config.map[i] = std::rand() % NUM_WORKERS;
  return config;
}

class LSTM : public Op {
public:
  LSTM(int _batchSize, int _inputSize, int _hiddenSize, Tensor x, Tensor hx, Tensor cx, std::string name)
  : batchSize(_batchSize), inputSize(_inputSize), hiddenSize(_hiddenSize), Op(name) {
    assert(x.nDims == 2);
    assert(x.dim[0] == batchSize);
    assert(x.dim[1] == inputSize);
    assert(hx.nDims == 2);
    assert(hx.dim[0] == batchSize);
    assert(hx.dim[1] == hiddenSize);
    assert(cx.nDims == 2);
    assert(cx.dim[0] == batchSize);
    assert(cx.dim[1] == hiddenSize);
    add_input_tensor(x);
    add_input_tensor(hx);
    add_input_tensor(cx);
    assert(numInputs == 3);
    Tensor hy(2, batchSize, hiddenSize, this, 0);
    Tensor cy(2, batchSize, hiddenSize, this, 1);
    outputTensors.push_back(hy);
    outputTensors.push_back(cy);
    nConfigs = 0;
    for (int i = 1; i <= NUM_PARTITIONS; i*=2)
      config_x[nConfigs++] = i;
    for (int i = 0; i < nConfigs; i++) {
      int b = batchSize / config_x[i];
      computeTime[i] = measure_lstm_time(1, LSTM_PER_NODE_LENGTH, b, inputSize, hiddenSize);
    }
    dpCompTime += computeTime[nConfigs-1] * config_x[nConfigs-1];
    mpCompTime += computeTime[0] * config_x[0];
    float bestV = computeTime[0] * config_x[0];
    for (int i = 0; i < nConfigs; i++)
      bestV = std::min(bestV, computeTime[i] * config_x[i]);
    bestCompTime += bestV;
    printf("		dpCompTime(%.2lf) bestCompTime(%.2lf)\n", computeTime[nConfigs-1] * config_x[nConfigs-1], bestV);
  }
  float compute(OpConfig c);
  float update(const std::vector<OpConfig>& vec);
  Rect get_tensor_shape(OpConfig c, int idx, Tensor t, bool is_input);
  //float xfer(Tensor t, OpConfig config, OpConfig preConfig);
  OpConfig get_random_config();
private:
  int batchSize, inputSize, hiddenSize;
  // params for configs
  int nConfigs, config_x[MAX_NUM_CONFIGS];
  float computeTime[MAX_NUM_CONFIGS];
};

float LSTM::compute(OpConfig c) {
  assert(c.nDims == 1);
  int idx = 0;
  for (; idx < nConfigs; idx++)
    if (config_x[idx] == c.dim[0])
      break;
  assert(idx < nConfigs);
  return computeTime[idx];
};

float LSTM::update(const std::vector<OpConfig>& vec)
{
  int used[NUM_WORKERS];
  float xfer = hiddenSize * inputSize * 8 * sizeof(float);
  for (int i = 0; i < NUM_WORKERS; i++)
    used[i] = 0;
  for (int i = 0; i < vec.size(); i++) {
    for (int j = 0; j < vec[i].nParts; j++)
      used[vec[i].map[j]] += 1;
  }
  int intra_cnt = 0, cross_cnt = 0, l2_cnt = 0;
  for (int i = 0; i < NUM_NODES; i++) {
    int cnt = 0;
    for (int j = 0; j < WORKERS_PER_NODE; j++) {
      if (used[i * WORKERS_PER_NODE + j] > 0) cnt++;
      l2_cnt += used[i * WORKERS_PER_NODE + j] * used[i * WORKERS_PER_NODE + j];
    }
    if (cnt > intra_cnt) intra_cnt = cnt;
    if (cnt > 0)
      totalDataXfer += (cnt - 1) * xfer;
    if (cnt > 0) cross_cnt ++;
  }
  assert(intra_cnt > 0);
  assert(cross_cnt > 0);
  totalDataXfer += (cross_cnt - 1) * xfer;
  //printf("intra(%.2lf) cross(%.2lf) epsilon(%.2lf)\n", xfer * (intra_cnt-1) / INTRA_NODE_BANDWIDTH, xfer * (cross_cnt-1) / CROSS_NODE_BANDWIDTH, l2_cnt * L2_FACTOR);
  float ret = xfer * (intra_cnt-1) / INTRA_NODE_BANDWIDTH
              + xfer * (cross_cnt-1) / CROSS_NODE_BANDWIDTH
              + l2_cnt * L2_FACTOR;
  //printf("LSTM::update: %.2lf\n", ret);
  return ret;
};

Rect LSTM::get_tensor_shape(OpConfig config, int idx, Tensor t, bool is_input)
{
  assert(config.nDims == 1);
  assert(t.nDims == 2); // Assume 2-D tensors
  assert(idx < config.dim[0]);
  int extent = t.dim[0] / config.dim[0];
  Rect r;
  r.nDims = 2;
  r.lo[0] = extent * idx; r.hi[0] = extent * (idx + 1) - 1;
  r.lo[1] = 0; r.hi[1] = t.dim[1] - 1;
  return r;
}

OpConfig LSTM::get_random_config()
{
  OpConfig config;
  config.nDims = 1;
  int idx = std::rand() % nConfigs;
  //int idx = nConfigs - 1;
  config.nParts = config_x[idx];
  config.dim[0] = config.nParts;
  for (int i = 0; i < config.nParts; i++)
    config.map[i] = std::rand() % NUM_WORKERS;
  return config;
}

class Softmax : public Op {
public:
  Softmax(int _batchSize, int _inputSize, int _outputSize, bool softmax, bool lstm_linear, Tensor x, std::string name)
  : batchSize(_batchSize), inputSize(_inputSize), outputSize(_outputSize), Op(name) {
    assert(x.nDims == 2);
    assert(x.dim[0] == batchSize);
    assert(x.dim[1] == inputSize);
    add_input_tensor(x);
    assert(numInputs == 1);
    Tensor y(2, batchSize, outputSize, this, 0);
    outputTensors.push_back(y);
    nConfigs = 0;
    //FIXME: for now only consider i * j == NUM_PARTITIONS
    for (int i = 1; i <= NUM_PARTITIONS; i*=2)
      for (int j = NUM_PARTITIONS / i; i * j <= NUM_PARTITIONS; j*=2) {
        config_x[nConfigs] = i; config_y[nConfigs] = j;
        nConfigs ++;
      }
    for (int i = 0; i < nConfigs; i++) {
      int batch = batchSize / config_x[i];
      if (lstm_linear) batch = batch * LSTM_PER_NODE_LENGTH;
      int output = outputSize / config_y[i];
      computeTime[i] = measure_linear_time(batch, inputSize, output, softmax);
    }
    dpCompTime += computeTime[nConfigs-1] * config_x[nConfigs-1] * config_y[nConfigs-1];
    mpCompTime += computeTime[0] * config_x[0] * config_y[0];
    float bestV = computeTime[0] * config_x[0] * config_y[0];
    for (int i = 0; i < nConfigs; i++)
      bestV = std::min(bestV, computeTime[i] * config_x[i] * config_y[i]);
    bestCompTime += bestV;
    printf("		dpCompTime(%.2lf) bestCompTime(%.2lf)\n", computeTime[nConfigs-1] * config_x[nConfigs-1] * config_y[nConfigs-1], bestV);
  }

  float compute(OpConfig c);
  float update(const std::vector<OpConfig>& c);
  Rect get_tensor_shape(OpConfig c, int idx, Tensor t, bool is_input);
  //float xfer(Tensor t, OpConfig config, OpConfig preConfig);
  OpConfig get_random_config();
private:
  int batchSize, inputSize, outputSize;
  // params for config
  int nConfigs, config_x[MAX_NUM_CONFIGS], config_y[MAX_NUM_CONFIGS];
  float computeTime[MAX_NUM_CONFIGS];
};

float Softmax::compute(OpConfig c) {
  //return 3.56 / c.nParts;
  assert(c.nDims == 2);
  int idx = 0;
  for (; idx < nConfigs; idx++)
    if (config_x[idx] == c.dim[0] && config_y[idx] == c.dim[1])
      break;
  assert(idx < nConfigs);
  return computeTime[idx];
};

float Softmax::update(const std::vector<OpConfig>& vec) {
  int used[NUM_WORKERS][NUM_PARTITIONS];
  for (int i = 0; i < NUM_WORKERS; i++)
    for (int j = 0; j < NUM_PARTITIONS; j++)
      used[i][j] = 0;
  float xfer = ((float)inputSize) * outputSize / NUM_PARTITIONS * sizeof(float);
  for (int i = 0; i < vec.size(); i++) {
    for (int j = 0; j < vec[i].dim[0]; j++)
      for (int k = 0; k < vec[i].dim[1]; k++)
        for (int l = 0; l < vec[i].dim[0]; l++) {
          int gpuID = vec[i].map[j*vec[i].dim[1] + k];
          int parID = k * vec[i].dim[0] + l;
          used[gpuID][parID] ++;
        }
  }
  float ret = 0.0f;
  for (int k = 0; k < NUM_PARTITIONS; k++) {
    int intra_cnt = 0, cross_cnt = 0, l2_cnt = 0;
    for (int i = 0; i < NUM_NODES; i++) {
      int cnt = 0;
      for (int j = 0; j < WORKERS_PER_NODE; j++) {
        if (used[i*WORKERS_PER_NODE+j][k] > 0) cnt++;
        l2_cnt += used[i * WORKERS_PER_NODE + j][k] * used[i * WORKERS_PER_NODE + j][k];
      }
      if (cnt > intra_cnt) intra_cnt = cnt;
      //FIXME: uncomment us
      //if (cnt > 0)
      //  totalDataXfer += (cnt - 1) * xfer;
      if (cnt > 0) cross_cnt ++;
    }
    assert(intra_cnt > 0);
    assert(cross_cnt > 0);
    totalDataXfer += (cross_cnt-1) * xfer;
    //printf("Linear[%d]: intra_cnt = %d cross_cnt = %d\n", k, intra_cnt, cross_cnt);
    //printf("Linear::intra_xfer = %.4lf cross_xfer = %.4lf\n", xfer * (intra_cnt-1)/ INTRA_NODE_BANDWIDTH, xfer * (cross_cnt-1) / CROSS_NODE_BANDWIDTH);
    ret += (xfer * (intra_cnt-1)) / INTRA_NODE_BANDWIDTH
           + (xfer * (cross_cnt-1))/ CROSS_NODE_BANDWIDTH
           + L2_FACTOR * l2_cnt;
  }
  //printf("Softmax::update = %.2lf\n", ret);
  return ret;
};

Rect Softmax::get_tensor_shape(OpConfig config, int idx, Tensor t, bool is_input)
{
  Rect r;
  int extent_n, extent_c, trans_n, trans_c;
  if (is_input) {
    extent_n = t.dim[0] / config.dim[0];
    extent_c = t.dim[1];
    trans_n = idx / config.dim[1];
    trans_c = 0;
  } else {
    Rect r;
    extent_n = t.dim[0] / config.dim[0];
    extent_c = t.dim[1] / config.dim[1];
    trans_n = idx / config.dim[1];
    trans_c = idx % config.dim[1];
  }
  r.nDims = t.nDims;
  assert(r.nDims == 2);
  r.lo[1] = trans_c * extent_c; r.hi[1] = (trans_c + 1) * extent_c - 1;
  r.lo[0] = trans_n * extent_n; r.hi[0] = (trans_n + 1) * extent_n - 1;
  return r;
}

/*
float Softmax::xfer(Tensor t, OpConfig config, OpConfig preConfig, int idx) {
  assert(numInputs == 1);
  assert(config.nParts == NUM_PARTITIONS);
  assert(preConfig.nParts == NUM_PARTITIONS);
  assert(t.nDims == 2);
  float xfer = (float)(t.dim[0] / config.dim[0]) * t.dim[1];
  assert(false);
};
*/

OpConfig Softmax::get_random_config() {
  OpConfig config;
  config.nDims = 2;
  int idx = std::rand() % nConfigs;
  config.dim[0] = config_x[idx];
  config.dim[1] = config_y[idx];
  config.nParts = config.dim[0] * config.dim[1];
  for (int i = 0; i < config.nParts; i++)
    config.map[i] = std::rand() % NUM_WORKERS;
  return config;
}

class Embed : public Op {
public:
  Embed(int _batchSize, int _vocabSize, int _outputSize, std::string name)
  : batchSize(_batchSize), vocabSize(_vocabSize), outputSize(_outputSize), Op(name) {
    assert(numInputs == 0);
    Tensor y(2, batchSize, outputSize, this, 0);
    outputTensors.push_back(y);
    nConfigs = 0;
    for (int i = 1; i <= NUM_PARTITIONS; i*=2)
      config_x[nConfigs++] = i;
  }

  float compute(OpConfig c);
  float update(const std::vector<OpConfig>& c);
  Rect get_tensor_shape(OpConfig c, int idx, Tensor t, bool is_input);
  //float xfer(Tensor t, OpConfig config, OpConfig, preConfig, int idx);
  OpConfig get_random_config();
private:
  int batchSize, vocabSize, outputSize;
  int nConfigs, config_x[MAX_NUM_CONFIGS];
};

float Embed::compute(OpConfig c) {
  return 0.04f / c.nParts + 0.01;
}

float Embed::update(const std::vector<OpConfig>& vec) {
  int used[NUM_WORKERS];
  float xfer = vocabSize * outputSize * sizeof(float);
  for (int i = 0; i < NUM_WORKERS; i++)
    used[i] = 0;
  for (int i = 0; i < vec.size(); i++) {
    for (int j = 0; j < vec[i].nParts; j++)
      used[vec[i].map[j]] += 1;
  }
  int intra_cnt = 0, cross_cnt = 0, l2_cnt = 0;
  for (int i = 0; i < NUM_NODES; i++) {
    int cnt = 0;
    for (int j = 0; j < WORKERS_PER_NODE; j++) {
      if (used[i*WORKERS_PER_NODE + j] > 0) cnt++;
      l2_cnt += used[i * WORKERS_PER_NODE + j] * used[i * WORKERS_PER_NODE + j];
    }
    if (cnt > intra_cnt) intra_cnt = cnt;
    if (cnt > 0)
      totalDataXfer += (cnt-1) * xfer;
    if (cnt > 0) cross_cnt++;
  }
  totalDataXfer += (cross_cnt-1) * xfer;
  assert(intra_cnt > 0);
  assert(cross_cnt > 0);
  float ret = xfer * (intra_cnt - 1) / INTRA_NODE_BANDWIDTH
              + xfer * (cross_cnt - 1) / CROSS_NODE_BANDWIDTH
              + l2_cnt * L2_FACTOR;
  //printf("Embed::update: %.2lf\n", ret);
  return ret;
}

Rect Embed::get_tensor_shape(OpConfig config, int idx, Tensor t, bool is_input)
{
  assert(t.nDims == 2);
  assert(config.nDims == 1);
  Rect r;
  r.nDims = 2;
  r.lo[1] = 0; r.hi[1] = t.dim[1] - 1;
  int extent = t.dim[0] / config.dim[0];
  r.lo[0] = extent * idx; r.hi[0] = extent * (idx + 1) - 1;
  return r;
}

/*
float Embed::xfer(Tensor t, OpConfig config, OpConfig preConfig, int idx) {
  assert(t.nDims == 2);
  assert(config.nParts == NUM_PARTITIONS);
  assert(preConfig.nParts == NUM_PARTITIONS);
  assert(config.dim[1] == 1);
  assert(preConfig.dim[1] = 1);
  float xfer = (t.dim[0] / preConfig.dim[0]) * t.dim[1];
  if (config.map[idx] == preConfig.map[idx])
    return 0.0f;
  else
    return xfer;
}
*/

OpConfig Embed::get_random_config()
{
  OpConfig config;
  config.nDims = 1;
  int idx = std::rand() % nConfigs;
  //int idx = nConfigs - 1;
  config.nParts = config_x[idx];
  config.dim[0] = config.nParts;
  for (int i = 0; i < config.nParts; i++)
    config.map[i] = std::rand() % NUM_WORKERS;
  return config;
}

int task_global_guid = 0;
class Task;
struct Edge {
  Task* task;
};
class Task {
public:
  Task(int _deviceId, float _computeTime)
  : workerId(_deviceId), counter(0), computeTime(_computeTime),
    readyTime(0.0f), previous(NULL), next(NULL) {
    guid = task_global_guid ++;
  }
  Task(int _deviceId, float _computeTime, int _guid)
  : workerId(_deviceId), counter(0), computeTime(_computeTime),
    readyTime(0.0f), previous(NULL), next(NULL) {
    printf("new guid = %d\n", _guid);
    guid = _guid;
  }
  void add_next_task(Task* next) {
    Edge nextEdge;
    nextEdge.task = next;
    nextTasks.push_back(nextEdge);
    Edge preEdge;
    preEdge.task = this;
    next->preTasks.push_back(preEdge);
  }

  void remove_next_task(Task* next) {
    while (true) {
      int idx = 0;
      for (idx = 0; idx < nextTasks.size(); idx++)
        if (nextTasks[idx].task == next) break;
      if (idx == nextTasks.size()) break;
      nextTasks.erase(nextTasks.begin() + idx);
    }
  }

  void remove_pre_task(Task* pre) {
    while (true) {
      int idx = 0;
      for (idx = 0; idx < preTasks.size(); idx++)
        if (preTasks[idx].task == pre) break;
      if (idx == preTasks.size()) break;
      preTasks.erase(preTasks.begin() + idx);
    }
  }

  void compute_ready_time() {
    readyTime = 0.0f;
    for (int i = 0; i < preTasks.size(); i++) {
      Task* pre = preTasks[i].task;
      readyTime = std::max(readyTime, pre->startTime + pre->computeTime);
    }
  }
  float readyTime, startTime, computeTime;
  int workerId, counter, guid;
  std::vector<Edge> preTasks, nextTasks;
  Task *previous, *next;
};

typedef Task* TaskPtr;

struct TaskCompare {
  bool operator() (const TaskPtr &lhs, const TaskPtr &rhs) const {
    //printf("lhs(%d %.2lf) rhs(%d %.2lf)\n", lhs->guid, lhs->readyTime, rhs->guid, rhs->readyTime);
    if (lhs->readyTime != rhs->readyTime)
      return (lhs->readyTime < rhs->readyTime);
    else
      return (lhs->guid < rhs->guid);
  }
};

float intersect(Rect a, Rect b)
{
  assert(a.nDims == b.nDims);
  float ret = 1.0f;
  for (int i = 0; i < a.nDims; i++) {
    int w = std::min(a.hi[i], b.hi[i]) - std::max(a.lo[i], b.lo[i]) + 1;
    w = std::max(w, 0);
    ret = ret * w;
  }
  return ret;
}

inline float bandwidth(int gpu1, int gpu2)
{
  int node1 = gpu1 % WORKERS_PER_NODE;
  int node2 = gpu2 % WORKERS_PER_NODE;
  if (gpu1 == gpu2)
    return 1024 * 1024 * 1024;
  else if (node1 == node2)
    return INTRA_NODE_BANDWIDTH;//NVLink Bandwidth
  else
    return CROSS_NODE_BANDWIDTH;// IB Bandwidth
}

Task* tasks[MAX_NUM_OPS][NUM_PARTITIONS];
Task* commTasks[MAX_NUM_OPS * NUM_PARTITIONS * NUM_PARTITIONS];
float simulate_time(const std::map<Op*, OpConfig>& global, bool print = false)
{
  // We need to reset task_global_guid
  task_global_guid = 0;
  int comm_task_id = 0;
  std::set<Task*, TaskCompare> readyQueue;
  totalDataXfer = 0.0f;
  float accCompTime = 0.0f;
  for (int i = 0; i < op_global_guid; i++) {
    Op* op = guidToOp[i];
    OpConfig config = global.find(op)->second;
    for (int j = 0; j < config.nParts; j++) {
      tasks[i][j] = new Task(config.map[j], op->compute(config));
      accCompTime += tasks[i][j]->computeTime;
    }
    // Build dependencies
    for (int j = 0; j < op->inputTensors.size(); j++) {
      Tensor t = op->inputTensors[j];
      Op* preOp = t.owner;
      if (preOp == NULL) continue;
      OpConfig preConfig = global.find(preOp)->second;
      for (int dstId = 0; dstId < config.nParts; dstId++) {
        Rect dstR = op->get_tensor_shape(config, dstId, t, true/*is_input*/);
        Task* dstT = tasks[i][dstId];
        //printf("dstOp(%d) idx(%d)\n", op->guid, dstId);
        for (int srcId = 0; srcId < preConfig.nParts; srcId++) {
          Rect srcR = preOp->get_tensor_shape(preConfig, srcId, t, false/*is_input*/);
          Task* srcT = tasks[preOp->guid][srcId];
          //printf("srcOp(%d) idx(%d)\n", preOp->guid, srcId);
          // Make sure we are adding links in order
          assert(preOp->guid < i);
          if (intersect(srcR, dstR) > 0) {
            // Add dependency between srcT -> dstT
            float cost = 0.0f;
            if (srcT->workerId != dstT->workerId) {
              cost = intersect(srcR, dstR) * LSTM_PER_NODE_LENGTH * 4 / bandwidth(srcT->workerId, dstT->workerId);
              totalDataXfer += intersect(srcR, dstR) * LSTM_PER_NODE_LENGTH * 4;
              int comm_device_id = (srcId + 1) * MAX_NUM_WORKERS + dstId;
              Task* task = new Task(comm_device_id, cost);
              commTasks[comm_task_id++] = task;
              srcT->add_next_task(task);
              task->add_next_task(dstT);
            } else {
              srcT->add_next_task(dstT);
            }
          }
        }
      }
    }
    // If No dependencies, add tasks to readyQueues
    for (int j = 0; j < config.nParts; j++)
      if (tasks[i][j]->preTasks.size() == 0) {
        readyQueue.insert(tasks[i][j]);
      }
  }
  std::vector<Task*> allTasks;
  float gpuTime[MAX_NUM_WORKERS * (MAX_NUM_WORKERS + 1)];
  for (int i = 0; i < MAX_NUM_WORKERS * (MAX_NUM_WORKERS + 1); i++) {
    gpuTime[i] = 0.0f;
  }
  while (!readyQueue.empty()) {
    // Find the task with earliest start time
    Task* t = *readyQueue.begin();
    allTasks.push_back(t);
    readyQueue.erase(readyQueue.begin());
    int gpuId = t->workerId;
    gpuTime[gpuId] = std::max(gpuTime[gpuId], t->readyTime);
    t->startTime = gpuTime[gpuId];
    gpuTime[gpuId] += t->computeTime;
    for (int i = 0; i < t->nextTasks.size(); i++) {
      Task* next = t->nextTasks[i].task;
      float nextReadyTime = t->startTime + t->computeTime;
      next->readyTime = std::max(next->readyTime, nextReadyTime);
      next->counter ++;
      if (next->counter == next->preTasks.size()) {
        // The task is ready
        readyQueue.insert(next);
      }
    }
    if (print)
      printf("[%zu] gpu(%.2lf %.2lf %.2lf %.2lf)\n", allTasks.size(), gpuTime[0], gpuTime[1], gpuTime[2], gpuTime[3]);
  }
  float totalTime = 0.0f;
  for (int i = 0; i < MAX_NUM_WORKERS; i++)
    totalTime = std::max(totalTime, gpuTime[i]);

  assert(allTasks.size() == task_global_guid);
  // Add update cost
  for (int i = 0; i < parameters.size(); i++) {
    assert(parameters[i].size() > 0);
    std::vector<Op*> opList = parameters[i];
    std::vector<OpConfig> configs;
    std::map<Op*, OpConfig>::const_iterator it;
    for (int j = 0; j < opList.size(); j++) {
      it = global.find(opList[j]);
      configs.push_back(it->second);
    }
    float updateCost = opList[0]->update(configs);
    totalTime += updateCost;
  }
#ifdef VERBOSE
  for (int i = 0; i < op_global_guid; i++) {
    OpConfig c = global.find(guidToOp[i])->second;
    printf("c[%d] dim(%d %d) map(%d %d %d %d)\n", i, c.dim[0], c.dim[1], c.map[0], c.map[1], c.map[2], c.map[3]);
  }
  for (int i = 0; i < op_global_guid; i++) {
    OpConfig config = global.find(guidToOp[i])->second;
    for (int j = 0; j < config.nParts; j++) {
      Task* t = tasks[i][j];
      printf("t[%d] ready(%.2lf) start(%.2lf) compute(%.2lf) next(", t->guid, t->readyTime, t->startTime, t->computeTime);
      for (int k = 0; k < t->nextTasks.size(); k++)
        printf("%d %.2lf, ", t->nextTasks[k].task->guid, t->nextTasks[k].cost);
      printf(")\n");
    }
  }
  for (int i = 0; i < allTasks.size(); i++)
    for (int j = i+1; j < allTasks.size(); j++)
      if (allTasks[i]->guid == allTasks[j]->guid) {
        assert(allTasks[i]->startTime + allTasks[i]->computeTime <= allTasks[j]->startTime);
      }
#endif
#ifdef NOT_USE_SIMULATE_OPT
  for (int i = 0; i < op_global_guid; i++) {
    OpConfig config = global.find(guidToOp[i])->second;
    for (int j = 0; j < config.nParts; j++)
      delete tasks[i][j];
  }
  for (int i = 0; i < comm_task_id; i++)
    delete commTasks[i];
#endif

  if (print) {
    printf("totalTime = %.2lf\n", totalTime);
    printf("totalDataXfer = %.2lf\n", totalDataXfer);
    printf("totalCompTime = %.2lf\n", accCompTime);
  }

  //return totalTime + costDataXfer * DATA_XFER_FACTOR;
  return totalTime;
}

void generate_init_config(std::map<Op*, OpConfig>& global)
{
  for (int i = 0; i < op_global_guid; i++) {
    global[guidToOp[i]] = guidToOp[i]->get_random_config();
  }
}

Op* rewrite(const std::map<Op*, OpConfig>& current,
             std::map<Op*, OpConfig>& next)
{
  next = current;
  int opId = std::rand() % op_global_guid;
  next[guidToOp[opId]] = guidToOp[opId]->get_random_config();
  return guidToOp[opId];
}

long long current_time(void)
{
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  long long t = (1000000LL * ts.tv_sec) + (ts.tv_nsec / 1000);
  return t;
}

Tensor add_conv_layer(Tensor t, int outputSize, int kernelX, int kernelY,
                      int strideX, int strideY, int paddingX, int paddingY,
                      std::string name)
{
  Conv2D* conv = new Conv2D(outputSize, kernelX, kernelY, strideX, strideY,
                            paddingX, paddingY, t, name);
  return conv->outputTensors[0];
}

Tensor add_pool_layer(Tensor t, int kernelX, int kernelY, int strideX, int strideY,
                      int paddingX, int paddingY, std::string name)
{
  Pool2D* pool = new Pool2D(kernelX, kernelY, strideX, strideY, paddingX, paddingY,
                            t, name);
  return pool->outputTensors[0];
}

Tensor add_flat_layer(Tensor t, std::string name)
{
  Flat* flat = new Flat(t, name);
  assert(flat->outputTensors.size() == 1);
  return flat->outputTensors[0];
}

Tensor add_linear_layer(Tensor t, int outputSize, bool softmaxLayer, std::string name)
{
  Softmax* softmax = new Softmax(t.dim[0], t.dim[1], outputSize, softmaxLayer, false/*lstm_linear*/, t, name);
  assert(softmax->outputTensors.size() == 1);
  return softmax->outputTensors[0];
}

Tensor add_concat_layer(int n, Tensor* tensors, std::string name)
{
  Concat* concat = new Concat(n, tensors, name);
  assert(concat->outputTensors.size() == 1);
  return concat->outputTensors[0];
}

void build_alexnet_model()
{
  init_cudnn();
  Tensor x(4, BATCH_SIZE, 3, 224, 224, NULL, 0);
  Tensor t = add_conv_layer(x, 64, 11, 11, 4, 4, 2, 2, "conv1");
  t = add_pool_layer(t, 3, 3, 2, 2, 0, 0, "pool2");
  t = add_conv_layer(t, 192, 5, 5, 1, 1, 2, 2, "conv3");
  t = add_pool_layer(t, 3, 3, 2, 2, 0, 0, "pool4");
  t = add_conv_layer(t, 384, 3, 3, 1, 1, 1, 1, "conv5");
  t = add_conv_layer(t, 256, 3, 3, 1, 1, 1, 1, "conv6");
  t = add_conv_layer(t, 256, 3, 3, 1, 1, 1, 1, "conv7");
  t = add_pool_layer(t, 3, 3, 2, 2, 0, 0, "pool8");
  t = add_flat_layer(t, "flat");
  t = add_linear_layer(t, 4096, false, "linear9");
  t = add_linear_layer(t, 4096, false, "linear10");
  t = add_linear_layer(t, 1000, true, "linear11");
  std::vector<Op*> opList;
  for (int i = 0; i < op_global_guid; i++) {
    opList.clear();
    opList.push_back(guidToOp[i]);
    parameters.push_back(opList);
  }
}

Tensor InceptionA(Tensor input, int pool_features, std::string prefix)
{
  Tensor t1 = add_conv_layer(input, 64, 1, 1, 1, 1, 0, 0, prefix + "conv1");
  Tensor t2 = add_conv_layer(input, 48, 1, 1, 1, 1, 0, 0, prefix + "conv2a");
  t2 = add_conv_layer(t2, 64, 5, 5, 1, 1, 2, 2, prefix + "conv2b");
  Tensor t3 = add_conv_layer(input, 64, 1, 1, 1, 1, 0, 0, prefix + "conv3a");
  t3 = add_conv_layer(t3, 96, 3, 3, 1, 1, 1, 1, prefix + "conv3b");
  t3 = add_conv_layer(t3, 96, 3, 3, 1, 1, 1, 1, prefix + "conv3c");
  Tensor t4 = add_pool_layer(input, 3, 3, 1, 1, 1, 1, prefix + "pool4a");
  t4 = add_conv_layer(t4, pool_features, 1, 1, 1, 1, 0, 0, prefix + "conv4b");
  Tensor concat[4];
  concat[0] = t1; concat[1] = t2; concat[2] = t3; concat[3] = t4;
  Tensor output = add_concat_layer(4, concat, prefix + "concat");
  return output;
}

Tensor InceptionB(Tensor input, std::string prefix)
{
  Tensor t1 = add_conv_layer(input, 384, 3, 3, 2, 2, 0, 0, prefix + "conv1");
  Tensor t2 = add_conv_layer(input, 64, 1, 1, 1, 1, 0, 0, prefix + "conv2a");
  t2 = add_conv_layer(t2, 96, 3, 3, 1, 1, 1, 1, prefix + "conv2b");
  t2 = add_conv_layer(t2, 96, 3, 3, 2, 2, 0, 0, prefix + "conv2c");
  Tensor t3 = add_pool_layer(input, 3, 3, 2, 2, 0, 0, prefix + "pool3");
  Tensor concat[3];
  concat[0] = t1; concat[1] = t2; concat[2] = t3;
  Tensor output = add_concat_layer(3, concat, prefix + "concat");
  return output;
}

Tensor InceptionC(Tensor input, int channels, std::string prefix)
{
  Tensor t1 = add_conv_layer(input, 192, 1, 1, 1, 1, 0, 0, prefix + "conv1");
  Tensor t2 = add_conv_layer(input, channels, 1, 1, 1, 1, 0, 0, prefix + "conv2a");
  t2 = add_conv_layer(t2, channels, 1, 7, 1, 1, 0, 3, prefix + "conv2b");
  t2 = add_conv_layer(t2, 192, 7, 1, 1, 1, 3, 0, prefix + "conv2c");
  Tensor t3 = add_conv_layer(input, channels, 1, 1, 1, 1, 0, 0, prefix + "conv3a");
  t3 = add_conv_layer(t3, channels, 7, 1, 1, 1, 3, 0, prefix + "conv3b");
  t3 = add_conv_layer(t3, channels, 1, 7, 1, 1, 0, 3, prefix + "conv3c");
  t3 = add_conv_layer(t3, channels, 7, 1, 1, 1, 3, 0, prefix + "conv3d");
  t3 = add_conv_layer(t3, 192, 1, 7, 1, 1, 0, 3, prefix + "conv3e");
  Tensor t4 = add_pool_layer(input, 3, 3, 1, 1, 1, 1, prefix + "pool4a");
  t4 = add_conv_layer(t4, 192, 1, 1, 1, 1, 0, 0, prefix + "conv4b");
  Tensor concat[4];
  concat[0] = t1; concat[1] = t2; concat[2] = t3; concat[3] = t4;
  Tensor output = add_concat_layer(4, concat, prefix + "concat");
  return output;
}

Tensor InceptionD(Tensor input, std::string prefix)
{
  Tensor t1 = add_conv_layer(input, 192, 1, 1, 1, 1, 0, 0, prefix + "conv1a");
  t1 = add_conv_layer(t1, 320, 3, 3, 2, 2, 0, 0, prefix + "conv1b");
  Tensor t2 = add_conv_layer(input, 192, 1, 1, 1, 1, 0, 0, prefix + "conv2a");
  t2 = add_conv_layer(t2, 192, 1, 7, 1, 1, 0, 3, prefix + "conv2b");
  t2 = add_conv_layer(t2, 192, 7, 1, 1, 1, 3, 0, prefix + "conv2c");
  t2 = add_conv_layer(t2, 192, 3, 3, 2, 2, 0, 0, prefix + "conv2d");
  Tensor t3 = add_pool_layer(input, 3, 3, 2, 2, 0, 0, prefix + "pool3");
  Tensor concat[3];
  concat[0] = t1; concat[1] = t2; concat[2] = t3;
  Tensor output = add_concat_layer(3, concat, prefix + "concat");
  return output;
}

Tensor InceptionE(Tensor input, std::string prefix)
{
  Tensor t1 = add_conv_layer(input, 320, 1, 1, 1, 1, 0, 0, prefix + "conv1");
  Tensor t2i = add_conv_layer(input, 384, 1, 1, 1, 1, 0, 0, prefix + "conv2a");
  Tensor t2 = add_conv_layer(t2i, 384, 1, 3, 1, 1, 0, 1, prefix + "conv2b");
  Tensor t3 = add_conv_layer(t2i, 384, 3, 1, 1, 1, 1, 0, prefix + "conv3a");
  Tensor t3i = add_conv_layer(input, 448, 1, 1, 1, 1, 0, 0, prefix + "conv3b");
  t3i = add_conv_layer(t3i, 384, 3, 3, 1, 1, 1, 1, prefix + "conv3c");
  Tensor t4 = add_conv_layer(t3i, 384, 1, 3, 1, 1, 0, 1, prefix + "conv4");
  Tensor t5 = add_conv_layer(t3i, 384, 3, 1, 1, 1, 1, 0, prefix + "conv5");
  Tensor t6 = add_pool_layer(input, 3, 3, 1, 1, 1, 1, prefix + "pool6a");
  t6 = add_conv_layer(t6, 192, 1, 1, 1, 1, 0, 0, prefix + "conv6b");
  Tensor concat[6];
  concat[0] = t1; concat[1] = t2; concat[2] = t3;
  concat[3] = t4; concat[4] = t5; concat[5] = t6;
  Tensor output = add_concat_layer(6, concat, prefix + "concat");
  return output;
}

void build_inception_model()
{
  init_cudnn();
  Tensor x(4, BATCH_SIZE, 3, 299, 299, NULL, 0);
  Tensor t = add_conv_layer(x, 32, 3, 3, 2, 2, 0, 0, "conv1");
  t = add_conv_layer(t, 32, 3, 3, 1, 1, 0, 0, "conv2");
  t = add_conv_layer(t, 64, 3, 3, 1, 1, 1, 1, "conv3");
  t = add_pool_layer(t, 3, 3, 2, 2, 0, 0, "pool4");
  t = add_conv_layer(t, 80, 1, 1, 1, 1, 0, 0, "conv5");
  t = add_conv_layer(t, 192, 3, 3, 1, 1, 0, 0, "conv6");
  t = add_pool_layer(t, 3, 3, 2, 2, 0, 0, "pool7");
  t = InceptionA(t, 32, "A1::");
  printf("End of InceptionA1\n");
  t = InceptionA(t, 64, "A2::");
  printf("End of InceptionA2\n");
  t = InceptionA(t, 64, "A3::");
  printf("End of InceptionA3\n");
  t = InceptionB(t, "B1::");
  printf("End of InceptionB\n");
  t = InceptionC(t, 128, "C1::");
  printf("End of InceptionC1\n");
  t = InceptionC(t, 160, "C2::");
  printf("End of InceptionC2\n");
  t = InceptionC(t, 160, "C3::");
  printf("End of InceptionC3\n");
  t = InceptionC(t, 192, "C4::");
  printf("End of InceptionC4\n");
  t = InceptionD(t, "D1::");
  printf("End of InceptionD\n");
  t = InceptionE(t, "E1::");
  printf("End of InceptionE1\n");
  t = InceptionE(t, "E2::");
  printf("End of InceptionE2\n");
  t = add_pool_layer(t, 8, 8, 1, 1, 0, 0, "pool");
  t = add_flat_layer(t, "flat");
  t = add_linear_layer(t, 2048, true, "linear");
  std::vector<Op*> opList;
  for (int i = 0; i < op_global_guid - 1; i++) {
    opList.clear();
    opList.push_back(guidToOp[i]);
    parameters.push_back(opList);
  }
}

Tensor DenseBlock(Tensor input, int numLayers, int growthRate, std::string prefix)
{
  Tensor t, last = input;
  for (int i = 0; i < numLayers; i++) {
    t = add_conv_layer(last, 4 * growthRate, 1, 1, 1, 1, 0, 0, prefix+"conv1");
    t = add_conv_layer(t, growthRate, 3, 3, 1, 1, 1, 1, prefix+"conv2");
    Tensor concat[2];
    concat[0] = last; concat[1] = t;
    last = add_concat_layer(2, concat, prefix+"concat");
  }
  return last;
}

Tensor Transition(Tensor input, int outputSize, std::string prefix)
{
  Tensor t = add_conv_layer(input, outputSize, 1, 1, 1, 1, 0, 0, prefix+"conv");
  t = add_pool_layer(t, 2, 2, 2, 2, 0, 0, prefix+"pool");
  return t;
}

void build_densenet_model()
{
  init_cudnn();
  Tensor x(4, BATCH_SIZE, 3, 224, 224, NULL, 0);
  Tensor t = add_conv_layer(x, 64, 7, 7, 2, 2, 3, 3, "conv1");
  t = add_pool_layer(t, 3, 3, 2, 2, 1, 1, "pool2");
  int numFeatures = 64;
  t = DenseBlock(t, 6, 32, "dense1::");
  numFeatures = (numFeatures + 32 * 6) / 2;
  t = Transition(t, numFeatures, "trans1::");
  t = DenseBlock(t, 12, 32, "dense2::");
  numFeatures = (numFeatures + 32 * 12) / 2;
  t = Transition(t, numFeatures, "trans2::");
  t = DenseBlock(t, 24, 32, "dense3::");
  numFeatures = (numFeatures + 32 * 24) / 2;
  t = Transition(t, numFeatures, "trans3::");
  t = DenseBlock(t, 16, 32, "dense4::");
  t = add_pool_layer(t, 7, 7, 1, 1, 0, 0, "pool");
  t = add_flat_layer(t, "flat");
  t = add_linear_layer(t, 1000, true, "linear");
  std::vector<Op*> opList;
  for (int i = 0; i < op_global_guid - 1; i++) {
    opList.clear();
    opList.push_back(guidToOp[i]);
    parameters.push_back(opList);
  }
}

Tensor BottleneckBlock(Tensor input, int outSize, int bnSize, int stride, std::string prefix)
{
  Tensor t = add_conv_layer(input, bnSize, 1, 1, 1, 1, 0, 0, prefix+"conv1");
  t = add_conv_layer(t, bnSize, 3, 3, stride, stride, 1, 1, prefix+"conv2");
  t = add_conv_layer(t, outSize, 1, 1, 1, 1, 0, 0, prefix+"conv3");
  return t;
}

void build_resnet_model()
{
  init_cudnn();
  Tensor x(4, BATCH_SIZE, 3, 224, 224, NULL, 0);
  Tensor t = add_conv_layer(x, 64, 7, 7, 2, 2, 3, 3, "conv1");
  t = add_pool_layer(t, 3, 3, 2, 2, 1, 1, "pool2");
  for (int i = 0; i < 3; i++)
    t = BottleneckBlock(t, 256, 64, 1, "Block1_");
  for (int i = 0; i < 4; i++) {
    int stride = (i==0) ? 2 : 1;
    t = BottleneckBlock(t, 512, 128, stride, "Block2_"+std::to_string(i)+"::");
  }
  for (int i = 0; i < 23; i++) {
    int stride = (i==0) ? 2 : 1;
    t = BottleneckBlock(t, 1024, 256, stride, "Block3_"+std::to_string(i)+"::");
  }
  for (int i = 0; i < 3; i++) {
    int stride = (i==0) ? 2 : 1;
    t = BottleneckBlock(t, 2048, 512, stride, "Block4_"+std::to_string(i)+"::");
  }
  t = add_pool_layer(t, 7, 7, 1, 1, 0, 0, "pool");
  t = add_flat_layer(t, "flat");
  t = add_linear_layer(t, 1000, true, "linear");
  std::vector<Op*> opList;
  for (int i = 0; i < op_global_guid - 1; i++) {
    opList.clear();
    opList.push_back(guidToOp[i]);
    parameters.push_back(opList);
  }
}

void build_nmt_model()
{
  init_cudnn();
  Tensor hx_init(2, BATCH_SIZE, HIDDEN_SIZE, NULL, 0);
  Tensor cx_init(2, BATCH_SIZE, HIDDEN_SIZE, NULL, 0);
  Embed* embed[SRC_LENGTH + DST_LENGTH];
  for (int i = 0; i < SRC_LENGTH + DST_LENGTH; i++)
    embed[i] = new Embed(BATCH_SIZE, VOCAB_SIZE, EMBEDDING_SIZE, "embed");

  LSTM* lstm[NUM_LAYERS][SRC_LENGTH + DST_LENGTH];
  for (int l = 0; l < NUM_LAYERS; l++)
    for (int i = 0; i < SRC_LENGTH + DST_LENGTH; i++) {
      if (l == 0)
        lstm[l][i] = new LSTM(BATCH_SIZE, EMBEDDING_SIZE, HIDDEN_SIZE, embed[i]->outputTensors[0],
                              i == 0 ? hx_init : lstm[l][i-1]->outputTensors[0],
                              i == 0 ? cx_init : lstm[l][i-1]->outputTensors[1], "encoder");
      else
        lstm[l][i] = new LSTM(BATCH_SIZE, HIDDEN_SIZE, HIDDEN_SIZE, lstm[l-1][i]->outputTensors[0],
                              i == 0 ? hx_init : lstm[l][i-1]->outputTensors[0],
                              i == 0 ? cx_init : lstm[l][i-1]->outputTensors[1], "decoder");
    }

  LSTM* attention[DST_LENGTH];
  for (int i = 0; i < DST_LENGTH; i++) {
    attention[i] = new LSTM(BATCH_SIZE, HIDDEN_SIZE, HIDDEN_SIZE, lstm[NUM_LAYERS-1][i+SRC_LENGTH]->outputTensors[0],
                            i == 0 ? hx_init : attention[i-1]->outputTensors[0],
                            i == 0 ? cx_init : attention[i-1]->outputTensors[1], "attention");
  }
  Softmax* softmax[DST_LENGTH];
  for (int i = 0; i < DST_LENGTH; i++)
    softmax[i] = new Softmax(BATCH_SIZE, HIDDEN_SIZE, VOCAB_SIZE, true/*softmax*/, true/*lstm_linear*/,
                             //lstm[NUM_LAYERS-1][i + SRC_LENGTH]->outputTensors[0], "linear");
                             attention[i]->outputTensors[0], "linear");
  int idx = 0;
  std::vector<Op*> opList;
  for (int l = 0; l <= NUM_LAYERS; l++) {
    opList.clear();
    for (int i = 0; i < SRC_LENGTH; i++)
      opList.push_back(guidToOp[idx++]);
    parameters.push_back(opList);
    opList.clear();
    for (int i = 0; i < DST_LENGTH; i++)
      opList.push_back(guidToOp[idx++]);
    parameters.push_back(opList);
  }
  opList.clear();
  for (int i = 0; i < DST_LENGTH; i++)
    opList.push_back(guidToOp[idx++]);
  parameters.push_back(opList);
  opList.clear();
  for (int i = 0; i < DST_LENGTH; i++)
    opList.push_back(guidToOp[idx++]);
  parameters.push_back(opList);
  assert(idx == op_global_guid);
}

void print_global_config(const std::map<Op*, OpConfig>& global)
{
  for (int i = 0; i < op_global_guid; i++) {
    OpConfig c = global.find(guidToOp[i])->second;
    printf("op[%d] %s:	dim(", i, guidToOp[i]->name.c_str());
    for (int j = 0; j < c.nDims; j++) printf("%d ", c.dim[j]);
    printf(") gpu(");
    for (int j = 0; j < c.nParts; j++) printf("%d ", c.map[j]);
    printf(")\n");
  }
}

int main()
{
  srand(time(NULL));
  build_nmt_model();
  printf("dpCompTime = %.2lf mpCompTime = %.2lf bestCompTime = %.2lf\n", dpCompTime, mpCompTime, bestCompTime);
  std::map<Op*, OpConfig> current, next, optimal;
  generate_init_config(current);
  for (int i = 0; i < op_global_guid; i++) {
    OpConfig config;
    config.nDims = (i >= op_global_guid - DST_LENGTH) ? 2 : 1;
    config.nParts = NUM_PARTITIONS;
    config.dim[0] = NUM_PARTITIONS;
    config.dim[1] = 1;
    for (int j = 0; j < config.nParts; j++)
      config.map[j] = j % NUM_WORKERS;
      //config.map[j] = (i / SRC_LENGTH) % NUM_WORKERS;
      //config.map[j] = (i / SRC_LENGTH) % WORKERS_PER_NODE + (j / WORKERS_PER_NODE) * WORKERS_PER_NODE;
    current[guidToOp[i]] = config;
  }
  optimal = current;
  float optimal_runtime = simulate_time(current, true), optimalDataXfer = 0.0f;
  float cur_runtime = optimal_runtime;
  long long start_time = current_time();
  int good_moves = 0, best_moves = 0;
  for (int i = 0; i <= 250000; i++) {
    Op* updOp = rewrite(current, next);
    float next_runtime = simulate_time(next);
    if (i % 10000 == 0) {
      printf("cur(%.2lf) next(%.2lf) best(%.2lf) optimalDataXfer(%.2lf)\n", cur_runtime, next_runtime, optimal_runtime, optimalDataXfer);
      long long end_time = current_time();
      printf("time = %lld us\n", (end_time - start_time) / (i+1));
      printf("best_moves (%d) good_moves (%d) i (%d)\n", best_moves, good_moves, i);
      print_global_config(optimal);
    }
    float rn = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
    float ratio = (next_runtime - cur_runtime) / cur_runtime;
    float diff = (next_runtime - cur_runtime);
    if (next_runtime < optimal_runtime) {
      best_moves ++;
      optimal_runtime = next_runtime;
      optimalDataXfer = totalDataXfer;
      optimal = next;
    }
    if (next_runtime < cur_runtime) {
      good_moves ++;
      current = next;
      cur_runtime = next_runtime;
    } else if (rn < std::exp(-5 * diff)) {
      current = next;
      cur_runtime = next_runtime;
    }
  }
}
