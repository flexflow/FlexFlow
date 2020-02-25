#ifndef _FLEXFLOW_CONST_H_
#define _FLEXFLOW_CONST_H_

enum ActiMode {
  AC_MODE_NONE = 10,
  AC_MODE_RELU = 11,
  AC_MODE_SIGMOID = 12,
  AC_MODE_TANH = 13,
};

enum AggrMode {
  AGGR_MODE_NONE = 20,
  AGGR_MODE_SUM = 21,
  AGGR_MODE_AVG = 22,
};

enum PoolType {
  POOL_MAX = 30,
  POOL_AVG = 31,
};

enum DataType {
  DT_FLOAT = 40,
  DT_DOUBLE = 41,
  DT_INT32 = 42,
  DT_INT64 = 43,
  DT_BOOLEAN = 44,
};

#endif // _FLEXFLOW_CONST_H_