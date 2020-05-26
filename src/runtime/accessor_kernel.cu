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
    assign_kernel<DT><<<GET_BLOCKS(rect.volume()), CUDA_NUM_THREADS>>>(
        ptr, rect.volume(), 0.0f);
    checkCUDA(cudaDeviceSynchronize());
  }
}

template<typename DT>
const DT* helperGetTensorPointerR(PhysicalRegion region,
                                  RegionRequirement req,
                                  FieldID fid,
                                  Context ctx,
                                  Runtime* runtime)
{
  Domain domain = runtime->get_index_space_domain(
      ctx, req.region.get_index_space());
  switch (domain.get_dim()) {
    case 1:
    {
      TensorAccessorR<DT, 1> acc(region, req, fid, ctx, runtime);
      return acc.ptr;
    }
    case 2:
    {
      TensorAccessorR<DT, 2> acc(region, req, fid, ctx, runtime);
      return acc.ptr;
    }
    case 3:
    {
      TensorAccessorR<DT, 3> acc(region, req, fid, ctx, runtime);
      return acc.ptr;
    }
    case 4:
    {
      TensorAccessorR<DT, 4> acc(region, req, fid, ctx, runtime);
      return acc.ptr;
    }
    default:
    {
      fprintf(stderr, "Unsupported accessor dimension");
      assert(false);
      return NULL;
    }
  }
}

template<typename DT>
DT* helperGetTensorPointerWO(PhysicalRegion region,
                             RegionRequirement req,
                             FieldID fid,
                             Context ctx,
                             Runtime* runtime)
{
  Domain domain = runtime->get_index_space_domain(
      ctx, req.region.get_index_space());
  switch (domain.get_dim()) {
    case 1:
    {
      TensorAccessorW<DT, 1> acc(region, req, fid, ctx, runtime, false/*readOutput*/);
      return acc.ptr;
    }
    case 2:
    {
      TensorAccessorW<DT, 2> acc(region, req, fid, ctx, runtime, false/*readOutput*/);

      return acc.ptr;
    }
    case 3:
    {
      TensorAccessorW<DT, 3> acc(region, req, fid, ctx, runtime, false/*readOutput*/);
      return acc.ptr;
    }
    case 4:
    {
      TensorAccessorW<DT, 4> acc(region, req, fid, ctx, runtime, false/*readOutput*/);
      return acc.ptr;
    }
    default:
    {
      fprintf(stderr, "Unsupported accessor dimension");
      assert(false);
      return NULL;
    }
  }
}

template class TensorAccessorR<float, 1>;
template class TensorAccessorR<float, 2>;
template class TensorAccessorR<float, 3>;
template class TensorAccessorR<float, 4>;
template class TensorAccessorR<int32_t, 1>;
template class TensorAccessorR<int32_t, 2>;
template class TensorAccessorR<int32_t, 3>;
template class TensorAccessorR<int32_t, 4>;
template class TensorAccessorR<int64_t, 1>;
template class TensorAccessorR<int64_t, 2>;
template class TensorAccessorR<int64_t, 3>;
template class TensorAccessorR<int64_t, 4>;

template class TensorAccessorW<float, 1>;
template class TensorAccessorW<float, 2>;
template class TensorAccessorW<float, 3>;
template class TensorAccessorW<float, 4>;
template class TensorAccessorW<int32_t, 1>;
template class TensorAccessorW<int32_t, 2>;
template class TensorAccessorW<int32_t, 3>;
template class TensorAccessorW<int32_t, 4>;
template class TensorAccessorW<int64_t, 1>;
template class TensorAccessorW<int64_t, 2>;
template class TensorAccessorW<int64_t, 3>;
template class TensorAccessorW<int64_t, 4>;

template const float* helperGetTensorPointerR(
  PhysicalRegion region, RegionRequirement req, FieldID fid, Context ctx, Runtime* runtime);

template float* helperGetTensorPointerWO(
  PhysicalRegion region, RegionRequirement req, FieldID fid, Context ctx, Runtime* runtime);
