#include "accessor.h"
#include "model.h"
#include "cuda_helper.h"

template<typename DT, int dim>
TensorAccessorR<DT, dim>::TensorAccessorR(PhysicalRegion region,
                                          RegionRequirement req,
                                          FieldID fid,
                                          Context ctx,
                                          Runtime* runtime)
{
  const AccessorRO<DT, dim> acc(region, fid);
  rect = runtime->get_index_space_domain(
      ctx, req.region.get_index_space());
  assert(acc.accessor.is_dense_arbitrary(rect));
  ptr = acc.ptr(rect);
}

template<typename DT>
__global__
void zero_array(DT* ptr, coord_t size)
{
  CUDA_KERNEL_LOOP(i, size)
  {
    ptr[i] = 0;
  }
}

template<typename DT, int dim>
TensorAccessorW<DT, dim>::TensorAccessorW(PhysicalRegion region,
                                          RegionRequirement req,
                                          FieldID fid,
                                          Context ctx,
                                          Runtime* runtime,
                                          bool readOutput)
{
  rect = runtime->get_index_space_domain(
      ctx, req.region.get_index_space());
  if (readOutput) {
    const AccessorRW<DT, dim> acc(region, fid);
    assert(acc.accessor.is_dense_arbitrary(rect));
    ptr = acc.ptr(rect);
  } else {
    const AccessorWO<DT, dim> acc(region, fid);
    assert(acc.accessor.is_dense_arbitrary(rect));
    ptr = acc.ptr(rect);
    // FIXME: currently we zero init the region if not read output
    //assign_kernel<DT><<<GET_BLOCKS(rect.volume()), CUDA_NUM_THREADS>>>(
    //    ptr, rect.volume(), 0.0f);
    //checkCUDA(cudaDeviceSynchronize());
  }
}

template class TensorAccessorR<float, 1>;
template class TensorAccessorR<float, 2>;
template class TensorAccessorR<float, 3>;
template class TensorAccessorR<int32_t, 1>;
template class TensorAccessorR<int32_t, 2>;
template class TensorAccessorR<int32_t, 3>;
template class TensorAccessorR<int64_t, 1>;
template class TensorAccessorR<int64_t, 2>;
template class TensorAccessorR<int64_t, 3>;

template class TensorAccessorW<float, 1>;
template class TensorAccessorW<float, 2>;
template class TensorAccessorW<float, 3>;
template class TensorAccessorW<int32_t, 1>;
template class TensorAccessorW<int32_t, 2>;
template class TensorAccessorW<int32_t, 3>;
template class TensorAccessorW<int64_t, 1>;
template class TensorAccessorW<int64_t, 2>;
template class TensorAccessorW<int64_t, 3>;
