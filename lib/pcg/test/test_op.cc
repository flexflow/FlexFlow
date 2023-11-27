#include "doctest.h"
#include "pcg/file_format/v1/op.h"
#include "utils.h"
#include "utils/json.h"

using namespace FlexFlow;

#define TEST(m)                                                                \
  do {                                                                         \
    V1Op v1 = to_v1(Op::m);                                                    \
                                                                               \
    CHECK(from_v1(v1) == Op::m);                                               \
                                                                               \
    json j = v1;                                                               \
    CHECK(str(j) == "\"" #m "\"");                                             \
    /* TODO: Check deserialization.*/                                          \
  } while (0)

TEST_CASE("V1Op") {
  TEST(NOOP);
  TEST(INPUT);
  TEST(WEIGHT);
  TEST(CONV2D);
  TEST(DROPOUT);
  TEST(LINEAR);
  TEST(BATCHMATMUL);
  TEST(POOL2D);
  TEST(SCALAR_MULTIPLY);
  TEST(SCALAR_ADD);
  TEST(SCALAR_FLOOR_DIV);
  TEST(SCALAR_TRUE_DIV);
  TEST(SCALAR_SUB);
  TEST(RELU);
  TEST(IDENTITY);
  TEST(SIGMOID);
  TEST(TANH);
  TEST(ELU);
  TEST(FLAT);
  TEST(SOFTMAX);
  TEST(BATCHNORM);
  TEST(CONCAT);
  TEST(SPLIT);
  TEST(EMBEDDING);
  TEST(GROUP_BY);
  TEST(CACHE);
  TEST(AGGREGATE);
  TEST(AGG_SPEC);
  // TEST(OP_ELEMENTWISE)
  TEST(RESHAPE);
  TEST(REVERSE);
  TEST(TRANSPOSE);
  TEST(EW_ADD);
  TEST(EW_MUL);
  TEST(MATMUL);
  TEST(MUL);
  TEST(ENLARGE);
  TEST(SQUEEZE);
  TEST(UNSQUEEZE);
  TEST(EW_SUB);
  TEST(EW_DIV);
  TEST(EW_EQUAL);
  TEST(EW_GREATER);
  TEST(EW_LESS);
  TEST(EW_MAX);
  TEST(EW_MIN);
  TEST(REDUCE_ARGMAX);
  TEST(REDUCE_ARGMIN);
  TEST(REDUCE_MAX);
  TEST(REDUCE_MEAN);
  TEST(REDUCE_MIN);
  TEST(REDUCE_PROD);
  TEST(REDUCE_SUM);
  TEST(PAD);
  TEST(SHAPE);
  TEST(SIZE);
  TEST(TOPK);
  TEST(WHERE);
  TEST(CEIL);
  TEST(CAST);
  TEST(EXP);
  TEST(ROUND);
  TEST(LOG);
  TEST(LOGICAL_NOT);
  TEST(SQRT);
  TEST(SIN);
  TEST(COS);
  TEST(LEAKYRELU);
  TEST(SLICE);
  TEST(RESIZE);
  TEST(PRELU);
  TEST(GELU);
  TEST(MULTIHEAD_ATTENTION);
  TEST(FUSED);
  TEST(RSQRT);
  TEST(POW);
  TEST(MEAN);
  TEST(LAYERNORM);
  TEST(GATHER);
  TEST(BROADCAST);
  TEST(REPARTITION);
  TEST(COMBINE);
  TEST(REPLICATE);
  TEST(REDUCTION);
  TEST(BATCH);
  TEST(PIPELINE);
  TEST(FUSED_PARALLEL);
}
