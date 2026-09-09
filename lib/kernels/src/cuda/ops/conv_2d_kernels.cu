#include "internal/device.h"
#include "kernels/conv_2d_kernels_gpu.h"
#include "op-attrs/ops/conv_2d.h"
#include "op-attrs/tensor_dims.h"
#include <vector>

namespace FlexFlow {

// Picks the fastest algorithm reported by cuDNN's heuristics that both
// succeeds and fits in the available workspace.
template <typename PerfResult>
static int select_algorithm_idx(std::vector<PerfResult> const &perf_results,
                                int num_results,
                                size_t workspace_size) {
  for (int i = 0; i < num_results; i++) {
    if (perf_results.at(i).status == CUDNN_STATUS_SUCCESS &&
        perf_results.at(i).memory <= workspace_size) {
      return i;
    }
  }

  PANIC("No cuDNN convolution algorithm fits within the available workspace",
        num_results,
        workspace_size);
}

Conv2DPerDeviceState conv_2d_gpu_init_kernel(PerDeviceFFHandle const &handle,
                                             Conv2DAttrs const &attrs,
                                             TensorShape const &input_shape,
                                             TensorShape const &output_shape) {
  // Applying an activation as part of Conv2D is not currently implemented. The
  // previous implementation hardcoded relu (regardless of which activation was
  // requested) and computed the backward pass by destructively modifying the
  // output gradient. If you need it, please create an issue.
  ASSERT(!attrs.activation.has_value(),
         "Conv2D does not currently support fused activations",
         attrs.activation);

  ASSERT(get_num_dims(input_shape.dims) == num_tensor_dims_t{4_n},
         "Conv2D expects 4-dimensional (i.e., NCHW) input tensors",
         input_shape);
  ASSERT(conv2d_get_output_shape(attrs, input_shape) == output_shape,
         "Conv2D output shape does not match the shape implied by its "
         "attributes and input shape",
         attrs,
         input_shape,
         output_shape);

  TensorShape filter_shape = conv2d_get_kernel_shape(attrs, input_shape);

  positive_int input_c = dim_at_idx(input_shape.dims, ff_dim_t{1_n});
  positive_int output_c = dim_at_idx(output_shape.dims, ff_dim_t{1_n});

  ASSERT(input_c % attrs.groups == 0,
         "Conv2D requires the number of input channels to be divisible by the "
         "number of groups",
         input_c,
         attrs.groups);

  ffTensorDescriptor_t inputTensor;
  ffTensorDescriptor_t biasTensor;
  ffTensorDescriptor_t outputTensor;
  ffFilterDescriptor_t filterDesc;
  ffConvolutionDescriptor_t convDesc;

  checkCUDNN(cudnnCreateTensorDescriptor(&inputTensor));
  checkCUDNN(cudnnCreateTensorDescriptor(&biasTensor));
  checkCUDNN(cudnnCreateTensorDescriptor(&outputTensor));
  checkCUDNN(cudnnCreateFilterDescriptor(&filterDesc));
  checkCUDNN(cudnnCreateConvolutionDescriptor(&convDesc));

  checkCUDNN(cudnnSetTensorDescriptorFromTensorShape(inputTensor, input_shape));
  checkCUDNN(
      cudnnSetTensorDescriptorFromTensorShape(outputTensor, output_shape));

  checkCUDNN(
      cudnnSetTensor4dDescriptor(biasTensor,
                                 CUDNN_TENSOR_NCHW,
                                 ff_to_cudnn_datatype(output_shape.data_type),
                                 /*n=*/1,
                                 /*c=*/output_c.int_from_positive_int(),
                                 /*h=*/1,
                                 /*w=*/1));

  checkCUDNN(cudnnSetFilter4dDescriptor(
      filterDesc,
      ff_to_cudnn_datatype(filter_shape.data_type),
      CUDNN_TENSOR_NCHW,
      dim_at_idx(filter_shape.dims, ff_dim_t{0_n}).int_from_positive_int(),
      dim_at_idx(filter_shape.dims, ff_dim_t{1_n}).int_from_positive_int(),
      dim_at_idx(filter_shape.dims, ff_dim_t{2_n}).int_from_positive_int(),
      dim_at_idx(filter_shape.dims, ff_dim_t{3_n}).int_from_positive_int()));

  checkCUDNN(
      cudnnSetConvolution2dDescriptor(convDesc,
                                      attrs.padding_h.unwrap_nonnegative(),
                                      attrs.padding_w.unwrap_nonnegative(),
                                      attrs.stride_h.int_from_positive_int(),
                                      attrs.stride_w.int_from_positive_int(),
                                      /*dilation_h=*/1,
                                      /*dilation_w=*/1,
                                      CUDNN_CROSS_CORRELATION,
                                      CUDNN_DATA_FLOAT));

  checkCUDNN(cudnnSetConvolutionGroupCount(
      convDesc, attrs.groups.int_from_positive_int()));

  // enable tensor core when possible
  if (handle.allowTensorOpMathConversion) {
    checkCUDNN(cudnnSetConvolutionMathType(
        convDesc, CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION));
  } else {
    checkCUDNN(cudnnSetConvolutionMathType(convDesc, CUDNN_TENSOR_OP_MATH));
  }

  {
    int n, c, h, w;
    checkCUDNN(cudnnGetConvolution2dForwardOutputDim(
        convDesc, inputTensor, filterDesc, &n, &c, &h, &w));
    ASSERT(dim_at_idx(output_shape.dims, ff_dim_t{0_n}) == positive_int{n});
    ASSERT(dim_at_idx(output_shape.dims, ff_dim_t{1_n}) == positive_int{c});
    ASSERT(dim_at_idx(output_shape.dims, ff_dim_t{2_n}) == positive_int{h});
    ASSERT(dim_at_idx(output_shape.dims, ff_dim_t{3_n}) == positive_int{w});
  }

  // NOTE: we use cuDNN's heuristics (rather than the cudnnFind*Ex family) to
  // select algorithms, as the latter requires (and destructively writes to)
  // real input, output and gradient buffers.
  ffConvolutionFwdAlgo_t fwdAlgo;
  {
    int max_num_results;
    checkCUDNN(cudnnGetConvolutionForwardAlgorithmMaxCount(handle.dnn,
                                                           &max_num_results));
    std::vector<cudnnConvolutionFwdAlgoPerf_t> perf_results(max_num_results);
    int num_results = 0;
    checkCUDNN(cudnnGetConvolutionForwardAlgorithm_v7(handle.dnn,
                                                      inputTensor,
                                                      filterDesc,
                                                      convDesc,
                                                      outputTensor,
                                                      max_num_results,
                                                      &num_results,
                                                      perf_results.data()));
    fwdAlgo = perf_results
                  .at(select_algorithm_idx(
                      perf_results, num_results, handle.workSpaceSize))
                  .algo;
  }

  ffConvolutionBwdFilterAlgo_t bwdFilterAlgo;
  {
    int max_num_results;
    checkCUDNN(cudnnGetConvolutionBackwardFilterAlgorithmMaxCount(
        handle.dnn, &max_num_results));
    std::vector<cudnnConvolutionBwdFilterAlgoPerf_t> perf_results(
        max_num_results);
    int num_results = 0;
    checkCUDNN(
        cudnnGetConvolutionBackwardFilterAlgorithm_v7(handle.dnn,
                                                      inputTensor,
                                                      outputTensor,
                                                      convDesc,
                                                      filterDesc,
                                                      max_num_results,
                                                      &num_results,
                                                      perf_results.data()));
    bwdFilterAlgo = perf_results
                        .at(select_algorithm_idx(
                            perf_results, num_results, handle.workSpaceSize))
                        .algo;
  }

  ffConvolutionBwdDataAlgo_t bwdDataAlgo;
  {
    int max_num_results;
    checkCUDNN(cudnnGetConvolutionBackwardDataAlgorithmMaxCount(
        handle.dnn, &max_num_results));
    std::vector<cudnnConvolutionBwdDataAlgoPerf_t> perf_results(
        max_num_results);
    int num_results = 0;
    checkCUDNN(
        cudnnGetConvolutionBackwardDataAlgorithm_v7(handle.dnn,
                                                    filterDesc,
                                                    outputTensor,
                                                    convDesc,
                                                    inputTensor,
                                                    max_num_results,
                                                    &num_results,
                                                    perf_results.data()));
    bwdDataAlgo = perf_results
                      .at(select_algorithm_idx(
                          perf_results, num_results, handle.workSpaceSize))
                      .algo;
  }

  return Conv2DPerDeviceState{
      /*inputTensor=*/inputTensor,
      /*biasTensor=*/biasTensor,
      /*outputTensor=*/outputTensor,
      /*filterDesc=*/filterDesc,
      /*convDesc=*/convDesc,
      /*fwdAlgo=*/fwdAlgo,
      /*bwdFilterAlgo=*/bwdFilterAlgo,
      /*bwdDataAlgo=*/bwdDataAlgo,
  };
}

void conv_2d_gpu_forward_kernel(
    cudaStream_t stream,
    PerDeviceFFHandle const &handle,
    Conv2DPerDeviceState const &per_device_state,
    Conv2DAttrs const &attrs,
    GenericTensorAccessorR const &input,
    GenericTensorAccessorR const &filter,
    std::optional<GenericTensorAccessorR> const &bias,
    GenericTensorAccessorW const &output) {
  ASSERT(bias.has_value() == attrs.use_bias);

  checkCUDNN(cudnnSetStream(handle.dnn, stream));

  float alpha = 1.0f, beta = 0.0f;
  checkCUDNN(cudnnConvolutionForward(handle.dnn,
                                     &alpha,
                                     per_device_state.inputTensor,
                                     input.ptr,
                                     per_device_state.filterDesc,
                                     filter.ptr,
                                     per_device_state.convDesc,
                                     per_device_state.fwdAlgo,
                                     handle.workSpace,
                                     handle.workSpaceSize,
                                     &beta,
                                     per_device_state.outputTensor,
                                     output.ptr));

  if (bias.has_value()) {
    checkCUDNN(cudnnAddTensor(handle.dnn,
                              &alpha,
                              per_device_state.biasTensor,
                              bias.value().ptr,
                              &alpha,
                              per_device_state.outputTensor,
                              output.ptr));
  }
}

void conv_2d_gpu_backward_kernel(
    cudaStream_t stream,
    PerDeviceFFHandle const &handle,
    Conv2DPerDeviceState const &per_device_state,
    Conv2DAttrs const &attrs,
    GenericTensorAccessorR const &output,
    GenericTensorAccessorR const &output_grad,
    GenericTensorAccessorR const &input,
    GenericTensorAccessorW const &input_grad,
    GenericTensorAccessorR const &filter,
    GenericTensorAccessorW const &filter_grad,
    std::optional<GenericTensorAccessorW> const &bias_grad) {
  ASSERT(bias_grad.has_value() == attrs.use_bias);

  checkCUDNN(cudnnSetStream(handle.dnn, stream));

  // NOTE: alpha is used for beta as well so that the gradients are accumulated
  // into rather than overwritten
  float alpha = 1.0f;

  checkCUDNN(cudnnConvolutionBackwardFilter(handle.dnn,
                                            &alpha,
                                            per_device_state.inputTensor,
                                            input.ptr,
                                            per_device_state.outputTensor,
                                            output_grad.ptr,
                                            per_device_state.convDesc,
                                            per_device_state.bwdFilterAlgo,
                                            handle.workSpace,
                                            handle.workSpaceSize,
                                            &alpha,
                                            per_device_state.filterDesc,
                                            filter_grad.ptr));

  if (bias_grad.has_value()) {
    checkCUDNN(cudnnConvolutionBackwardBias(handle.dnn,
                                            &alpha,
                                            per_device_state.outputTensor,
                                            output_grad.ptr,
                                            &alpha,
                                            per_device_state.biasTensor,
                                            bias_grad.value().ptr));
  }

  checkCUDNN(cudnnConvolutionBackwardData(handle.dnn,
                                          &alpha,
                                          per_device_state.filterDesc,
                                          filter.ptr,
                                          per_device_state.outputTensor,
                                          output_grad.ptr,
                                          per_device_state.convDesc,
                                          per_device_state.bwdDataAlgo,
                                          handle.workSpace,
                                          handle.workSpaceSize,
                                          &alpha,
                                          per_device_state.inputTensor,
                                          input_grad.ptr));
}

void conv_2d_gpu_cleanup_kernel(Conv2DPerDeviceState &per_device_state) {
  checkCUDNN(cudnnDestroyTensorDescriptor(per_device_state.inputTensor));
  checkCUDNN(cudnnDestroyTensorDescriptor(per_device_state.biasTensor));
  checkCUDNN(cudnnDestroyTensorDescriptor(per_device_state.outputTensor));
  checkCUDNN(cudnnDestroyFilterDescriptor(per_device_state.filterDesc));
  checkCUDNN(cudnnDestroyConvolutionDescriptor(per_device_state.convDesc));
}

} // namespace FlexFlow
