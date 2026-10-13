#include "internal/realm_test_utils.h"
#include "realm-execution/distributed_ff_handle.h"
#include "op-attrs/ops/replicate_attrs.dtg.h"
#include "op-attrs/pcg_operator_attrs.dtg.h"
#include "realm-execution/realm_manager.h"
#include "realm-execution/tasks/impl/nccl_task.h"
#include "realm-execution/tensor_instance_backing.h"
#include "task-spec/dynamic_graph/training_operation_attrs.dtg.h"

#include <cuda_runtime.h>
#include <doctest/doctest.h>
#include <nccl.h>

namespace test {

using namespace ::FlexFlow;
namespace Realm = ::FlexFlow::Realm;

TEST_SUITE(FF_CUDA_TEST_SUITE) {

TEST_CASE("NCCL task spawns successfully") {
  std::vector<char *> fake_args =
      make_fake_realm_args(/*num_cpus=*/1_p, /*num_gpus=*/1_n);

  int fake_argc = fake_args.size();
  char **fake_argv = fake_args.data();

  RealmManager manager(&fake_argc, &fake_argv);

  ControllerTaskResult result =
      manager.start_controller([](RealmContext &ctx) {
          Realm::Machine::ProcessorQuery processor_query(
              Realm::Machine::get_machine());
          processor_query.only_kind(Realm::Processor::TOC_PROC);

        Realm::Processor gpu_proc = processor_query.first();

        DistributedFfHandle distributed_handle =
            create_distributed_ff_handle(
                ctx,
                /*workSpaceSize=*/1024 * 1024,
                /*allowTensorOpMathConversion=*/true,
                Realm::Event::NO_EVENT);
        DynamicNodeInvocation invocation{
            /*inputs=*/{},
            /*node_attrs=*/
            DynamicNodeAttrs{
                /*task_type=*/std::nullopt,
                /*device_coord=*/std::nullopt,
                /*mapping=*/std::nullopt,
                /*op_attrs=*/
                TrainingOperationAttrs{
                    PCGOperatorAttrs{
                        ReplicateAttrs{
                            /*replicate_degree=*/8_p,
                        },
                    },
                },
                /*layer_guid=*/
                dynamic_layer_guid_t{
                    parallel_layer_guid_t{
                        Node{0},
                    },
                },
                /*per_device_op_state=*/std::nullopt,
            },
            /*outputs=*/{},
        };

        TensorInstanceBacking tensor_backing =
            make_empty_tensor_instance_backing();

        Realm::Event event = spawn_nccl_task(
            ctx,
            gpu_proc,
            invocation,
            tensor_backing,
            distributed_handle.at(gpu_proc),
            Realm::Event::NO_EVENT);

        event.wait();
      });

  result.wait();
}

TEST_CASE("NCCL broadcast and reduce helpers") {
  constexpr size_t count = 8;
  size_t const buffer_size = count * sizeof(int);

  ncclUniqueId unique_id;
  REQUIRE(ncclGetUniqueId(&unique_id) == ncclSuccess);

  ncclComm_t communicator;
  REQUIRE(ncclCommInitRank(
              &communicator,
              /*num_ranks=*/1,
              unique_id,
              /*rank=*/0) == ncclSuccess);

  ffStream_t stream;
  REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);

  int *send_buffer = nullptr;
  int *receive_buffer = nullptr;

  REQUIRE(cudaMalloc(&send_buffer, buffer_size) == cudaSuccess);
  REQUIRE(cudaMalloc(&receive_buffer, buffer_size) == cudaSuccess);

  std::vector<int> input = {1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<int> output(count, 0);

  REQUIRE(cudaMemcpy(send_buffer,
                     input.data(),
                     buffer_size,
                     cudaMemcpyHostToDevice) == cudaSuccess);

  SUBCASE("broadcast") {
    REQUIRE(cudaMemset(receive_buffer, 0, buffer_size) == cudaSuccess);

    REQUIRE(run_nccl_broadcast(send_buffer,
                               receive_buffer,
                               count,
                               ncclInt32,
                               /*root_rank=*/0,
                               communicator,
                               stream) == ncclSuccess);

    REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);

    REQUIRE(cudaMemcpy(output.data(),
                       receive_buffer,
                       buffer_size,
                       cudaMemcpyDeviceToHost) == cudaSuccess);

    for (size_t i = 0; i < count; i++) {
      CHECK(output[i] == input[i]);
    }
  }

  SUBCASE("reduce") {
    REQUIRE(cudaMemset(receive_buffer, 0, buffer_size) == cudaSuccess);

    REQUIRE(run_nccl_reduce(send_buffer,
                            receive_buffer,
                            count,
                            ncclInt32,
                            ncclSum,
                            /*root_rank=*/0,
                            communicator,
                            stream) == ncclSuccess);

    REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);

    REQUIRE(cudaMemcpy(output.data(),
                       receive_buffer,
                       buffer_size,
                       cudaMemcpyDeviceToHost) == cudaSuccess);

    for (size_t i = 0; i < count; i++) {
      CHECK(output[i] == input[i]);
    }
  }

  REQUIRE(cudaFree(send_buffer) == cudaSuccess);
  REQUIRE(cudaFree(receive_buffer) == cudaSuccess);
  REQUIRE(cudaStreamDestroy(stream) == cudaSuccess);
  REQUIRE(ncclCommDestroy(communicator) == ncclSuccess);
}

} // TEST_SUITE

} // namespace test
