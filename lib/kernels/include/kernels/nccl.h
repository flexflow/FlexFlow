#ifndef _FLEXFLOW_KERNELS_INCLUDE_KERNELS_NCCL_H
#define _FLEXFLOW_KERNELS_INCLUDE_KERNELS_NCCL_H

#ifdef FF_USE_NCCL
#include <cstdio>
#include <cstdlib>
#include <nccl.h>

#define checkNCCL(cmd)                                                         \
  do {                                                                         \
    ncclResult_t r = cmd;                                                      \
    if (r != ncclSuccess) {                                                    \
      printf("Failed, NCCL error %s:%d '%s'\n",                                \
             __FILE__,                                                         \
             __LINE__,                                                         \
             ncclGetErrorString(r));                                           \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

#else
struct ncclUniqueId {};
struct ncclComm_t {};
#endif

namespace FlexFlow::Kernels::NCCL {

ncclUniqueId generate_unique_id();
ncclComm_t create_comm(ncclUniqueId const &, int num_ranks, int my_rank);

} // namespace FlexFlow::Kernels::NCCL

#endif
