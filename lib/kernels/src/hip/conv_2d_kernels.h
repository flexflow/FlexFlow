#ifndef _FLEXFLOW_KERNELS_HIP_CONV_2D_KERNELS_H
#define _FLEXFLOW_KERNELS_HIP_CONV_2D_KERNELS_H

#include "kernels/device.h"

namespace FlexFlow {
namespace Kernels {
namespace Conv2D {

miopenConvFwdAlgorithm_t selectConvolutionForwardAlgorithm(
    miopenHandle_t handle,
    const miopenTensorDescriptor_t xDesc,
    void const *x,
    const miopenTensorDescriptor_t wDesc,
    void const *w,
    const miopenConvolutionDescriptor_t convDesc,
    void *workSpace,
    size_t workSpaceSize,
    const miopenTensorDescriptor_t yDesc,
    void *y,
    float *time);

miopenConvBwdWeightsAlgorithm_t selectConvolutionBackwardFilterAlgorithm(
    miopenHandle_t handle,
    const miopenTensorDescriptor_t xDesc,
    void const *x,
    const miopenTensorDescriptor_t dyDesc,
    void const *dy,
    const miopenConvolutionDescriptor_t convDesc,
    void *workSpace,
    size_t workSpaceSize,
    const miopenTensorDescriptor_t dwDesc,
    void *dw,
    float *time);

miopenConvBwdDataAlgorithm_t selectConvolutionBackwardDataAlgorithm(
    miopenHandle_t handle,
    const miopenTensorDescriptor_t wDesc,
    void const *w,
    const miopenTensorDescriptor_t dyDesc,
    void const *dy,
    const miopenConvolutionDescriptor_t convDesc,
    void *workSpace,
    size_t workSpaceSize,
    const miopenTensorDescriptor_t dxDesc,
    void *dx,
    float *time);

} // namespace Conv2D
} // namespace Kernels
} // namespace FlexFlow

#endif
