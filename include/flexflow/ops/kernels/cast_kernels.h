#ifndef _FLEXFLOW_OPS_KERNELS_CAST_KERNELS_H
#define _FLEXFLOW_OPS_KERNELS_CAST_KERNELS_H

#include "flexflow/device.h"
#include "flexflow/fftype.h"
#include "flexflow/op_meta.h"

namespace FlexFlow {

class Cast;

class CastMeta : public OpMeta {
public:
  CastMeta(FFHandler handle, Cast const *cast);
  DataType input_data_type, output_data_type;
};

namespace Kernels {
namespace Cast {
template <typename IDT, typename ODT>
void forward_kernel_wrapper(CastMeta const *m,
                            IDT const *input_ptr,
                            ODT *output_ptr,
                            size_t volume);

template <typename IDT, typename ODT>
void backward_kernel_wrapper(IDT const *src_ptr, ODT *dst_ptr, size_t volume);

namespace Internal {

template <typename IDT, typename ODT>
void forward_kernel(IDT const *input_ptr,
                    ODT *output_ptr,
                    size_t volume,
                    ffStream_t stream);
template <typename IDT, typename ODT>
void backward_kernel(IDT const *src_ptr,
                     ODT *dst_ptr,
                     size_t volume,
                     ffStream_t stream);
} // namespace Internal
} // namespace Cast
} // namespace Kernels
} // namespace FlexFlow

#endif // _FLEXFLOW_OPS_KERNELS_CAST_KERNELS_H
