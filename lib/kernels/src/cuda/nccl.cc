#include "kernels/nccl.h"
#include "utils/exception.h"
#include <cassert>

namespace FlexFlow {

ncclUniqueId generate_unique_id() {
#ifdef FF_USE_NCCL
  ncclUniqueId ncclId;
  checkNCCL(ncclGetUniqueId(&ncclId));
  return ncclId;
#else
  throw mk_runtime_error("FF_USE_NCCL is not defined");
#endif
}

ncclComm_t create_comm_raw(ncclUniqueId const &unique_id, int num_ranks,
                           int my_rank) {
#ifdef FF_USE_NCCL
  ncclComm_t ncclComm;
  assert(my_rank < num_ranks);
  checkNCCL(ncclCommInitRank(&ncclComm, num_ranks, unique_id, my_rank));
  // fprintf(stderr, "ncclComm(%p) allRanks(%d) myRank(%d) ncclId(%p)\n",
  //     ncclComm, allRanks, myRank, ncclId);
  return ncclComm;
#else
  throw mk_runtime_error("FF_USE_NCCL is not defined");
#endif
}

} // namespace FlexFlow
