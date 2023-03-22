#ifndef _FLEXFLOW_KERNELS_CUDA_CONV_2D_KERNELS_H
#define _FLEXFLOW_KERNELS_CUDA_CONV_2D_KERNELS_H

#include "kernels/device.h"

namespace FlexFlow {
namespace Kernels {
namespace Conv2D {
namespace Internal {

cudnnConvolutionFwdAlgo_t selectConvolutionForwardAlgorithm(
    cudnnHandle_t handle,
    const cudnnTensorDescriptor_t xDesc,
    void const *x,
    const cudnnFilterDescriptor_t wDesc,
    void const *w,
    const cudnnConvolutionDescriptor_t convDesc,
    void *workSpace,
    size_t workSpaceSize,
    const cudnnTensorDescriptor_t yDesc,
    void *y,
    float *time);

cudnnConvolutionBwdDataAlgo_t selectConvolutionBackwardDataAlgorithm(
    cudnnHandle_t handle,
    const cudnnFilterDescriptor_t wDesc,
    void const *w,
    const cudnnTensorDescriptor_t dyDesc,
    void const *dy,
    const cudnnConvolutionDescriptor_t convDesc,
    void *workSpace,
    size_t workSpaceSize,
    const cudnnTensorDescriptor_t dxDesc,
    void *dx,
    float *time);

cudnnConvolutionBwdFilterAlgo_t selectConvolutionBackwardFilterAlgorithm(
    cudnnHandle_t handle,
    const cudnnTensorDescriptor_t xDesc,
    void const *x,
    const cudnnTensorDescriptor_t dyDesc,
    void const *dy,
    const cudnnConvolutionDescriptor_t convDesc,
    void *workSpace,
    size_t workSpaceSize,
    const cudnnFilterDescriptor_t dwDesc,
    void *dw,
    float *time);

}
}
}
}

#endif 
