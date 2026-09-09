#include "kernels/batch_norm_kernels_gpu.h"
#include "internal/test_utils.h"
#include "kernels/create_accessor_with_contents.h"
#include "kernels/format_accessor_contents.h"
#include "test/utils/doctest/check_kv.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

static BatchNormAttrs make_attrs() {
  return BatchNormAttrs{
      /*relu=*/false,
      /*affine=*/true,
      /*eps=*/1e-5,
      /*momentum=*/0.1,
  };
}

// NCHW
static GenericTensorAccessorR make_input(Allocator &allocator) {
  return create_4d_accessor_r_with_contents<float>(
      {
          {
              {{1, 2}, {3, 4}},
              {{-1, -2}, {0.5, 0.25}},
          },
          {
              {{5, 6}, {7, 8}},
              {{2, 0.125}, {-3, 1}},
          },
      },
      allocator);
}

static GenericTensorAccessorR make_gamma(Allocator &allocator) {
  return create_1d_accessor_r_with_contents<float>({2, 0.5}, allocator);
}

static GenericTensorAccessorR make_beta(Allocator &allocator) {
  return create_1d_accessor_r_with_contents<float>({-1, 3}, allocator);
}

static GenericTensorAccessorR make_output(Allocator &allocator) {
  return create_4d_accessor_r_with_contents<float>(
      {
          {
              {{-4.055047512054443, -3.1821770668029785},
               {-2.3093061447143555, -1.4364354610443115}},
              {{2.760241985321045, 2.433763027191162},
               {3.249960422515869, 3.1683406829833984}},
          },
          {
              {{-0.5635647177696228, 0.3093060255050659},
               {1.1821768283843994, 2.0550475120544434}},
              {{3.7396788597106934, 3.127530813217163},
               {2.1072840690612793, 3.4131999015808105}},
          },
      },
      allocator);
}

TEST_SUITE(FF_CUDA_TEST_SUITE) {
  TEST_CASE("batch_norm_gpu_forward_kernel") {
    ManagedPerDeviceFFHandle managed_handle = initialize_single_gpu_handle(
        /*workSpaceSize=*/1024 * 1024,
        /*allowTensorOpMathConversion=*/true);
    ManagedFFStream managed_stream{};

    Allocator allocator = create_local_cuda_memory_allocator();

    BatchNormAttrs attrs = make_attrs();

    GenericTensorAccessorR input = make_input(allocator);
    GenericTensorAccessorR gamma = make_gamma(allocator);
    GenericTensorAccessorR beta = make_beta(allocator);

    // Intentionally randomize this tensor so we can be confident we never read
    // it
    GenericTensorAccessorW output =
        create_random_filled_accessor_w(input.shape, allocator);

    BatchNormPerDeviceState per_device_state =
        batch_norm_gpu_init_kernel(allocator, attrs, input.shape, output.shape);

    batch_norm_gpu_forward_kernel(
        /*stream=*/managed_stream.raw_stream(),
        /*handle=*/managed_handle.raw_handle(),
        /*per_device_state=*/per_device_state,
        /*attrs=*/attrs,
        /*input=*/input,
        /*gamma=*/gamma,
        /*beta=*/beta,
        /*output=*/output);

    GenericTensorAccessorR correct = make_output(allocator);

    CHECK_MESSAGE(accessors_are_equal(output, correct),
                  check_kv("output", format_accessor_w_contents(output)));

    batch_norm_gpu_cleanup_kernel(allocator, per_device_state);
  }

  TEST_CASE("batch_norm_gpu_backward_kernel") {
    ManagedPerDeviceFFHandle managed_handle = initialize_single_gpu_handle(
        /*workSpaceSize=*/1024 * 1024,
        /*allowTensorOpMathConversion=*/true);
    ManagedFFStream managed_stream{};

    Allocator allocator = create_local_cuda_memory_allocator();

    BatchNormAttrs attrs = make_attrs();

    GenericTensorAccessorR input = make_input(allocator);
    GenericTensorAccessorR gamma = make_gamma(allocator);
    GenericTensorAccessorR beta = make_beta(allocator);

    GenericTensorAccessorW forward_output =
        create_random_filled_accessor_w(input.shape, allocator);

    BatchNormPerDeviceState per_device_state = batch_norm_gpu_init_kernel(
        allocator, attrs, input.shape, forward_output.shape);

    // cudnnBatchNormalizationBackward reads the batch statistics saved by the
    // forward pass, so the forward kernel has to run first
    batch_norm_gpu_forward_kernel(
        /*stream=*/managed_stream.raw_stream(),
        /*handle=*/managed_handle.raw_handle(),
        /*per_device_state=*/per_device_state,
        /*attrs=*/attrs,
        /*input=*/input,
        /*gamma=*/gamma,
        /*beta=*/beta,
        /*output=*/forward_output);

    GenericTensorAccessorR output = make_output(allocator);

    GenericTensorAccessorR output_grad =
        create_4d_accessor_r_with_contents<float>(
            {
                {
                    {{-0.875, -0.75}, {-0.625, -0.5}},
                    {{-0.375, -0.25}, {-0.125, 0}},
                },
                {
                    {{0.125, 0.25}, {0.375, 0.5}},
                    {{0.625, 0.75}, {0.875, 1}},
                },
            },
            allocator);

    // The gradients are accumulated into, so they need to start from known
    // values
    GenericTensorAccessorW input_grad =
        create_4d_accessor_w_with_contents<float>(
            {
                {
                    {{-1.75, -1.5}, {-1.25, -1}},
                    {{-0.75, -0.5}, {-0.25, 0}},
                },
                {
                    {{0.25, 0.5}, {0.75, 1}},
                    {{1.25, 1.5}, {1.75, 2}},
                },
            },
            allocator);

    GenericTensorAccessorW gamma_grad =
        create_1d_accessor_w_with_contents<float>({7, -3}, allocator);

    GenericTensorAccessorW beta_grad =
        create_1d_accessor_w_with_contents<float>({0.5, 11}, allocator);

    batch_norm_gpu_backward_kernel(
        /*stream=*/managed_stream.raw_stream(),
        /*handle=*/managed_handle.raw_handle(),
        /*per_device_state=*/per_device_state,
        /*attrs=*/attrs,
        /*output=*/output,
        /*output_grad=*/output_grad,
        /*input=*/input,
        /*input_grad=*/input_grad,
        /*gamma=*/gamma,
        /*gamma_grad=*/gamma_grad,
        /*beta_grad=*/beta_grad);

    GenericTensorAccessorR correct_input_grad =
        create_4d_accessor_r_with_contents<float>(
            {
                {
                    {{-1.6772620677947998, -1.510392189025879},
                     {-1.3435224294662476, -1.1766525506973267}},
                    {{-0.9591808915138245, -0.6475732326507568},
                     {-0.4087578058242798, -0.11274851113557816}},
                },
                {
                    {{0.42665261030197144, 0.5935224294662476},
                     {0.7603921890258789, 0.927262008190155}},
                    {{1.3049046993255615, 1.634710431098938},
                     {1.9905133247375488, 2.198132038116455}},
                },
            },
            allocator);

    GenericTensorAccessorR correct_gamma_grad =
        create_1d_accessor_r_with_contents<float>(
            {11.037027359008789, -2.2195112705230713}, allocator);

    GenericTensorAccessorR correct_beta_grad =
        create_1d_accessor_r_with_contents<float>({-1.0, 13.5}, allocator);

    CHECK_MESSAGE(
        // cuDNN's batch norm data gradient is not bit-identical to PyTorch's
        accessors_within_epsilon(input_grad, correct_input_grad, 1e-6),
        check_kv("input_grad", format_accessor_w_contents(input_grad)));

    CHECK_MESSAGE(
        accessors_are_equal(gamma_grad, correct_gamma_grad),
        check_kv("gamma_grad", format_accessor_w_contents(gamma_grad)));

    CHECK_MESSAGE(accessors_are_equal(beta_grad, correct_beta_grad),
                  check_kv("beta_grad", format_accessor_w_contents(beta_grad)));

    batch_norm_gpu_cleanup_kernel(allocator, per_device_state);
  }
}
