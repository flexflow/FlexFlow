
#include "flexflow/utils/cuda_helper.h"
#include "mlp.h"

void DataLoader::load_input(Task const *task,
                            std::vector<PhysicalRegion> const &regions,
                            Context ctx,
                            Runtime *runtime) {
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  SampleIdxs *meta = (SampleIdxs *)task->local_args;
  TensorAccessorR<float, 5> acc_full_input(
      regions[0], task->regions[0], FID_DATA, ctx, runtime);
  TensorAccessorW<float, 5> acc_batch_input(regions[1],
                                            task->regions[1],
                                            FID_DATA,
                                            ctx,
                                            runtime,
                                            false /*readOutput*/);
  coord_t batch_size =
      acc_batch_input.rect.hi[3] - acc_batch_input.rect.lo[3] + 1;
  coord_t channels =
      acc_batch_input.rect.hi[2] - acc_batch_input.rect.lo[2] + 1;
  coord_t height = acc_batch_input.rect.hi[1] - acc_batch_input.rect.lo[1] + 1;
  coord_t width = acc_batch_input.rect.hi[0] - acc_batch_input.rect.lo[0] + 1;
  // FIXME: currently assume continous indices
  assert(batch_size == meta->num_samples);
  for (int i = 1; i < batch_size; i++)
    assert(meta->idxs[i] == meta->idxs[0] + i);
  coord_t start_idx = meta->idxs[0];
  float const *input_zc =
      acc_full_input.ptr + start_idx * channels * height * width;
  copy_kernel<<<GET_BLOCKS(acc_batch_input.rect.volume()), CUDA_NUM_THREADS>>>(
      acc_batch_input.ptr, input_zc, acc_batch_input.rect.volume());
  checkCUDA(cudaDeviceSynchronize());
}
