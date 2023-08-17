/* Copyright 2023 CMU, Facebook, LANL, MIT, NVIDIA, and Stanford (alphabetical)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef _FLEXFLOW_SIMULATOR_H_
#define _FLEXFLOW_SIMULATOR_H_

#include "config.h"
#include "ffconst.h"
#include "flexflow/operator_params.h"
#include "flexflow/utils/hash_utils.h"
#include "mpark/variant.hpp"
#include "parallel_tensor.h"
#include <fstream>
#include <memory>
#include <queue>
#include <unordered_map>
#include <unordered_set>

namespace mp = mpark;

namespace FlexFlow {

#define MOD(a, b) ((a) % (b)) < 0 ? ((a) % (b)) + (b) : ((a) % (b))

class Conv2DMeta;
class LinearMeta;
class Pool2DMeta;
class ElementUnaryMeta;
class ElementBinaryMeta;
class LayerNormMeta;
// class EmbeddingMeta;
// class SoftmaxMeta;
class BatchMatmulMeta;
// class BatchNormMeta;
class ConcatMeta;
// class DropoutMeta;
class TransposeMeta;
class Op;
class FFModel;

/**
 * @brief Costs of an operator.
 */
struct CostMetrics {
  /**
   * @brief Return the sum of inputs_memory, outputs_memory, and weights_memory
   * recorded in this CostMetrics.
   */
  size_t total_memory() const;

  /**
   * @brief Return the sum of memory recorded in this CostMetrics, but in MB,
   * instead of Bytes.
   */
  float total_memory_in_mb() const;

  /**
   * @brief Get the incremental difference between the total memory in
   * CostMetrics and sim->offset.
   * @details This is to easily compute the difference between sim->offset and
   * sum of all memory usage recorded in this CostMetrics.
   *
   * @param sim_offset Simulator->offset
   * @return size_t The incremental memory usage difference
   */
  size_t total_mem_diff_from(off_t sim_offset) const;

public:
  float forward_time = 0, backward_time = 0, sync_time = 0;
  ///< Bytes of memory usage of different parts
  // Assume:
  // 1. all memory allocations use Simulator::allocate
  // 2. we call Simulator::free_all before measuring an operator
  // Therefore, the current memory usage of an operator is (size_t)sim->offset
  size_t inputs_memory = 0, outputs_memory = 0, weights_memory = 0;
  ///< Memory usage of Op* considering parallelization over devices
  size_t op_total_mem = 0;
};

class Device {
public:
  enum DeviceType {
    DEVICE_COMP,
    DEVICE_MEM,
    DEVICE_COMM,
  };
  Device(std::string const &name,
         DeviceType type,
         int node_id,
         int socket_id,
         int device_id);
  std::string name;
  DeviceType type;
  int node_id;
  int socket_id;
  int device_id;
};

class CompDevice : public Device {
public:
  enum CompDevType {
    LOC_PROC, // CPU
    TOC_PROC, // GPU
  };
  CompDevType comp_type;
  size_t capacity;
  CompDevice(std::string const &name,
             CompDevType comp_type,
             int node_id,
             int socket_id,
             int device_id);
};

class MemDevice : public Device {
public:
  enum MemDevType {
    SYSTEM_MEM, // DRAM on a single node
    Z_COPY_MEM, // Zero-copy memory betweeen CPU DRAM and all GPUs on a single
                // node
    GPU_FB_MEM, // GPU framebuffer memory for a single GPU
  };
  MemDevType mem_type;
  size_t capacity;
  MemDevice(std::string const &name,
            MemDevType mem_type,
            int node_id,
            int socket_id,
            int device_id,
            size_t capacity);
};

class CommDevice : public Device {
public:
  enum CommDevType {
    MEMBUS_COMM,
    UPI_IN_COMM,
    UPI_OUT_COMM,
    NIC_IN_COMM,
    NIC_OUT_COMM,
    PCI_TO_HOST_COMM,
    PCI_TO_DEV_COMM,
    NVLINK_COMM,
    NW_COMM,
    NW_NOMINAL,
  };
  CommDevType comm_type;
  float latency;
  float bandwidth;
  CommDevice(std::string const &name,
             CommDevType comm_type,
             int node_id,
             int socket_id,
             int device_id,
             float latency,
             float bandwidth);
};

typedef std::vector<CommDevice *> Route;
/* first is an array of cumulative distribution */
typedef std::pair<std::vector<float>, std::vector<Route>> EcmpRoutes;
typedef std::vector<int> ConnectionMatrix;
class NetworkRoutingStrategy;
/**
 * Nomincal communication device.
 * This is an communication device that allows "path expansion"
 * With this device, its possible to store a taskgraph in the "logical"
 * view (p2p) while when doing the simulaion, expand to physical version
 */
class NominalCommDevice : public CommDevice {
public:
  NominalCommDevice(std::string const &name,
                    int device_id,
                    int nnode,
                    NetworkRoutingStrategy *routing);
  /* pick one of the weighted ECMP path */
  Route expand_to_physical() const;
  EcmpRoutes const &get_all_routes();
  void set_physical_paths(EcmpRoutes const &rs);
  void reset();

public:
  NetworkRoutingStrategy *routing_strategy;
  EcmpRoutes routes;
  bool dirty = true;
  int nnode;
};

/**
 * Base class that provides the network routing strategy
 */
class NetworkRoutingStrategy {
public:
  virtual ~NetworkRoutingStrategy() = default;
  /**
   * For weighted ecmp support: the return type is a vector of pair of
   * <possible route, chance>
   */
  virtual EcmpRoutes get_routes(int src_node, int dst_node) = 0;
  virtual std::vector<EcmpRoutes> get_routes_from_src(int src_node) = 0;
};

class MachineModel {
public:
  virtual ~MachineModel() = default;
  virtual int get_version() const = 0;
  virtual CompDevice *get_gpu(int device_id) const = 0;
  virtual MemDevice *get_gpu_fb_mem(int devicd_id) const = 0;
  virtual int get_num_gpus() const = 0;
  virtual float get_intra_node_gpu_bandwidth() const = 0;
  virtual float get_inter_node_gpu_bandwidth() const = 0;
  virtual float get_intra_node_gpu_latency() const = 0;
  virtual float get_inter_node_gpu_latency() const = 0;
  virtual std::vector<CommDevice *> get_comm_path(MemDevice *src_mem,
                                                  MemDevice *tar_mem) = 0;
  virtual std::string to_string() const = 0;
  int version;
};

class SimpleMachineModel : public MachineModel {
public:
  SimpleMachineModel(int num_nodes, int num_gpus_per_node, size_t capacity);
  ~SimpleMachineModel();
  int get_version() const;
  CompDevice *get_gpu(int device_id) const;
  MemDevice *get_gpu_fb_mem(int devicd_id) const;
  int get_num_gpus() const;
  float get_intra_node_gpu_bandwidth() const;
  float get_inter_node_gpu_bandwidth() const;
  float get_intra_node_gpu_latency() const {
    return 0;
  }
  float get_inter_node_gpu_latency() const {
    return 0;
  }
  std::vector<CommDevice *> get_comm_path(MemDevice *src_mem,
                                          MemDevice *tar_mem);
  std::string to_string() const;

private:
  int num_nodes;
  int num_gpus_per_node;
  int num_gpus;
  float inter_gpu_bandwidth;
  float inter_node_bandwidth;
  float gpu_dram_bandwidth;
  std::map<int, CompDevice *> id_to_gpu;
  std::map<int, MemDevice *> id_to_gpu_fb_mem;
  std::map<int, CommDevice *> id_to_gputodram_comm_device;
  std::map<int, CommDevice *> id_to_dramtogpu_comm_device;
  std::map<size_t, CommDevice *> ids_to_inter_gpu_comm_device;
  std::map<size_t, CommDevice *> ids_to_inter_node_comm_device;
};

/**
 * An enhanced machine model supports the following features:
 * 1. Customize the machine model with a configuration file.
 * 2. Support socket-level simulation.
 * 3. Simulate congestions on a communication device. In this machine model,
 * some communication devices, such as NIC_IN and NIC_OUT, represent the
 * communication ports instead of the links in the simple machine model. In this
 * way, for example, concurrent inter-node communications from node A to node B
 * and from node A to node C share the same NIC_OUT device on node A, which
 * simulates the slowdown of concurrent communications when transferring big
 * messages.
 * 4. When passing big messages, the messages usually are divided into segments
 * and transferred one-by-one to overlap the communications on different
 * devices. This machine model can simulate this kind of pipelining.
 */
class EnhancedMachineModel : public MachineModel {
public:
  EnhancedMachineModel(std::string file, size_t gpu_fb_mem_capacity);
  ~EnhancedMachineModel();
  int get_version() const;
  CompDevice *get_cpu(int device_id) const;
  CompDevice *get_cpu(int socket_id, int local_id) const;
  CompDevice *get_gpu(int device_id) const;
  CompDevice *get_gpu(int socket_id, int local_id) const;
  MemDevice *get_sys_mem(int socket_id) const;
  MemDevice *get_z_copy_mem(int socket_id) const;
  MemDevice *get_gpu_fb_mem(int device_id) const;
  MemDevice *get_gpu_fb_mem(int socket_id, int local_id) const;
  CommDevice *get_nvlink(MemDevice *src_mem, MemDevice *tar_mem) const;
  CommDevice *get_next_nic_in(int socket_id);
  CommDevice *get_next_nic_out(int socket_id) const;
  int get_num_gpus() const;
  float get_intra_node_gpu_bandwidth() const;
  float get_inter_node_gpu_bandwidth() const;
  float get_intra_node_gpu_latency() const {
    return membus_latency;
  }
  float get_inter_node_gpu_latency() const {
    return nic_latency;
  }
  std::vector<CommDevice *> get_comm_path(MemDevice *src_mem,
                                          MemDevice *tar_mem);
  std::string to_string() const;

private:
  int num_nodes;
  int num_sockets_per_node;
  int num_cpus_per_socket;
  int num_gpus_per_socket;
  int num_sockets;
  int num_cpus;
  int num_gpus;
  int num_nvlinks_per_node;
  float membus_latency;
  float membus_bandwidth;
  float upi_latency;
  float upi_bandwidth;
  float nic_latency;
  float nic_bandwidth;
  int nic_persocket;
  int cur_nic_local_id;
  float pci_latency;
  float pci_bandwidth;
  float nvlink_latency;
  float nvlink_bandwidth;
  size_t gpu_fb_mem_capacity;
  std::vector<CommDevice::CommDevType> intra_socket_sys_mem_to_sys_mem;
  std::vector<CommDevice::CommDevType> inter_socket_sys_mem_to_sys_mem;
  std::vector<CommDevice::CommDevType> inter_node_sys_mem_to_sys_mem;
  std::vector<CommDevice::CommDevType> intra_socket_sys_mem_to_gpu_fb_mem;
  std::vector<CommDevice::CommDevType> inter_socket_sys_mem_to_gpu_fb_mem;
  std::vector<CommDevice::CommDevType> inter_node_sys_mem_to_gpu_fb_mem;
  std::vector<CommDevice::CommDevType> intra_socket_gpu_fb_mem_to_sys_mem;
  std::vector<CommDevice::CommDevType> inter_socket_gpu_fb_mem_to_sys_mem;
  std::vector<CommDevice::CommDevType> inter_node_gpu_fb_mem_to_sys_mem;
  std::vector<CommDevice::CommDevType> intra_socket_gpu_fb_mem_to_gpu_fb_mem;
  std::vector<CommDevice::CommDevType> inter_socket_gpu_fb_mem_to_gpu_fb_mem;
  std::vector<CommDevice::CommDevType> inter_node_gpu_fb_mem_to_gpu_fb_mem;
  std::vector<std::vector<CompDevice *>> cpus;       // socket_id, local_id
  std::vector<std::vector<CompDevice *>> gpus;       // socket_id, local_id
  std::vector<MemDevice *> sys_mems;                 // socket_id
  std::vector<MemDevice *> z_copy_mems;              // socket_id
  std::vector<std::vector<MemDevice *>> gpu_fb_mems; // socket_id, local_id
  std::vector<CommDevice *> membuses;                // socket_id
  std::vector<CommDevice *> upi_ins;                 // socket_id
  std::vector<CommDevice *> upi_outs;                // socket_id
  std::vector<std::vector<CommDevice *>> nic_ins;    // socket_id, local_id
  std::vector<std::vector<CommDevice *>> nic_outs;   // socket_id, local_id
  std::vector<CommDevice *> pcis_to_host; // from gpu to main memory, socket_id
  std::vector<CommDevice *>
      pcis_to_device; // from main memory to gpu, socket_id
  std::vector<std::vector<CommDevice *>> nvlinks; // node_id, local_id
  std::unordered_map<size_t, CommDevice *> mem_to_nvlink;
  // set up communication paths from a config file
  void set_comm_path(std::vector<CommDevice::CommDevType> &comm_path,
                     std::string device_str);
  void add_cpus();
  void add_gpus();
  void add_membuses(float latency, float bandwidth);
  void add_upis(float latency, float bandwidth);
  void add_nics(float latency, float bandwidth, int nic_persocket);
  void add_pcis(float latency, float bandwidth);
  void add_nvlinks(float latency, float bandwidth);
  // attach a nvlink communication device to a pair of GPU framebuffer memories
  void attach_nvlink(MemDevice *src_mem, MemDevice *tar_mem, CommDevice *comm);
  // return a list of specific communication devices based on the descriptions
  // of a communication path
  void add_comm_path(
      std::vector<CommDevice::CommDevType> const &comm_device_list,
      MemDevice *src_mem,
      MemDevice *tar_mem,
      std::vector<CommDevice *> &ret);
};

/**
 * Single shortest path routing based on hop count
 */
class WeightedShortestPathRoutingStrategy : public NetworkRoutingStrategy {
public:
  WeightedShortestPathRoutingStrategy(
      ConnectionMatrix const &c,
      std::map<size_t, CommDevice *> const &devmap,
      int total_devs);
  virtual EcmpRoutes get_routes(int src_node, int dst_node);
  virtual std::vector<EcmpRoutes> get_routes_from_src(int src_node);
  void hop_count(int src_node, int dst_node, int &hop, int &narrowest);
  std::vector<std::pair<int, int>> hop_count(int src_node);
  void clear();

public:
  ConnectionMatrix const &conn;
  std::map<size_t, CommDevice *> const &devmap;
  int total_devs;
};

class ShortestPathNetworkRoutingStrategy : public NetworkRoutingStrategy {
public:
  ShortestPathNetworkRoutingStrategy(
      ConnectionMatrix const &c,
      std::map<size_t, CommDevice *> const &devmap,
      int total_devs);
  virtual EcmpRoutes get_routes(int src_node, int dst_node);
  virtual std::vector<EcmpRoutes> get_routes_from_src(int src_node);
  void hop_count(int src_node, int dst_node, int &hop, int &narrowest);
  std::vector<std::pair<int, int>> hop_count(int src_node);
  void clear();

public:
  ConnectionMatrix const &conn;
  std::map<size_t, CommDevice *> const &devmap;
  int total_devs;
};

/**
 * A (virtual base) class that generates network topology
 * Maybe this should be moved out of simulator
 */
class NetworkTopologyGenerator {
public:
  virtual ConnectionMatrix generate_topology() const = 0;
  static void
      print_conn_matrix(ConnectionMatrix const &conn, int nnode, int nswitch) {
    int nnwdevs = nnode + nswitch;
    for (int i = 0; i < nnwdevs; i++) {
      if (i == nnode) {
        std::cout << std::endl;
      }
      for (int j = 0; j < nnwdevs; j++) {
        if (j == nnode) {
          std::cout << "\t";
        }
        std::cout << conn[i * nnwdevs + j] << "\t";
      }
      std::cout << std::endl;
    }
  }
};

/**
 * Generate a flat network topology that's degree constraint and guaranteed
 * to be connected
 */
class FlatDegConstraintNetworkTopologyGenerator
    : public NetworkTopologyGenerator {
public:
  FlatDegConstraintNetworkTopologyGenerator(int num_nodes, int degree);
  virtual ConnectionMatrix generate_topology() const;

public:
  inline int get_id(int i, int j) const;
  inline int get_if_in_use(int node, ConnectionMatrix const &conn) const;
  int num_nodes;
  int degree;
};

/**
 * Generate an abstract-switch network topology
 * good for simple simulation of a fattree
 */
class BigSwitchNetworkTopologyGenerator : public NetworkTopologyGenerator {
public:
  BigSwitchNetworkTopologyGenerator(int num_nodes);
  virtual ConnectionMatrix generate_topology() const;

public:
  int num_nodes;
};

/**
 * Generate a zero matrix
 */
class FlatEmptyNetworkTopologyGenerator : public NetworkTopologyGenerator {
public:
  FlatEmptyNetworkTopologyGenerator(int num_nodes) : num_nodes(num_nodes) {}
  virtual ConnectionMatrix generate_topology() const {
    return ConnectionMatrix(num_nodes * num_nodes, 0);
  }

public:
  int num_nodes;
};

class FCTopologyGenerator : public NetworkTopologyGenerator {
public:
  FCTopologyGenerator(int num_nodes) : num_nodes(num_nodes) {}
  virtual ConnectionMatrix generate_topology() const {
    ConnectionMatrix result = ConnectionMatrix(num_nodes * num_nodes, 1);
    for (int i = 0; i < num_nodes; i++) {
      result[i + i * num_nodes] = 0;
    }
    return result;
  }

public:
  int num_nodes;
};
/**
 * A model that is network topology-aware.
 * The network topology is represented as follows:
 *      An adjacency matrix is used to represnt the network connection
 *      The matrix has dimension (n+s)*(n+s) where n is the number of servers
 *      in the cluster, and s is the number of switches in the cluster.
 *      This implies that for a flat topology the matrix is n*n,
 *      while for a FatTree topology the network will have the upper n*n
 *      block to be 0. Switches has node_id starting from n.
 *      Note that the "big switch" model has the convinent representation of
 *      {{0, 1},{1, 0}} in block form.
 * As a first implementation this class is based on the existing SimpleMachine
 * model. We could use the enhanced version but it could be too much for the
 * MCMC search to run for thousand of iterations...
 */
class NetworkedMachineModel : public MachineModel {
public:
  /**
   * Constructor. A network topology specified as above needs to be provided
   * in the form of a single vector.
   */
  NetworkedMachineModel(int num_nodes,
                        int num_gpus_per_node,
                        int num_switches,
                        float network_latency,
                        std::vector<int> const &topology,
                        size_t capacity,
                        float link_bandwidth);
  ~NetworkedMachineModel();
  int get_version() const;
  CompDevice *get_gpu(int device_id) const;
  MemDevice *get_gpu_fb_mem(int devicd_id) const;
  int get_num_gpus() const;
  int get_num_nodes() const {
    return num_nodes;
  }
  int get_total_devs() const {
    return num_nodes + num_switches;
  }
  int get_num_switches() const {
    return num_switches;
  }
  float get_intra_node_gpu_bandwidth() const;
  float get_inter_node_gpu_bandwidth() const;
  float get_link_bandwidth() const;
  float get_link_bandwidth(int src, int dst) const;
  float get_intra_node_gpu_latency() const {
    return 0;
  }
  float get_inter_node_gpu_latency() const {
    return network_latency;
  }
  void set_routing_strategy(NetworkRoutingStrategy *rs);
  std::vector<CommDevice *> get_comm_path(MemDevice *src_mem,
                                          MemDevice *tar_mem);
  std::string to_string() const;
  /* return only the nominal device. For recording tg. */
  CommDevice *get_nominal_path(MemDevice *src_mem, MemDevice *tar_mem) const;
  /* stores the network topology as a json */
  void save_topology_json(std::string const &fname) const;
  void update_route();

  void set_topology(std::vector<int> const &topology);
  ConnectionMatrix const &get_conn_matrix();
  std::map<size_t, NominalCommDevice *> const &get_nomm_comm_devs();

  void set_pcie(bool state);
  void set_pipeline(bool state);

  int num_nodes;
  int num_gpus_per_node;
  int num_gpus;
  int num_switches;
  int total_devs;
  float inter_gpu_bandwidth;
  float link_bandwidth;
  float network_latency;
  float gpu_dram_bandwidth;

  bool pipelined;
  bool pcie_on;

  // float gpu_dram_bandwidth;
  /* Note that every non-zero entry corrsepond to a device in
   * in_to_nw_comm_device */
  ConnectionMatrix conn_matrix;
  NetworkRoutingStrategy *routing_strategy;
  std::map<int, CompDevice *> id_to_gpu;
  std::map<int, MemDevice *> id_to_gpu_fb_mem;
  // don't model PCIE for speed
  std::map<int, CommDevice *> id_to_gputodram_comm_device;
  std::map<int, CommDevice *> id_to_dramtogpu_comm_device;
  std::map<size_t, CommDevice *> ids_to_inter_gpu_comm_device;

  /* this refers to the actual links in the system */
  std::map<size_t, CommDevice *> ids_to_nw_comm_device;
  /* on the other hand, this represents the "nomical" communication device
   * or the "logical connection" in side the system. Note that this is
   * keyed on GPUs only
   */
  std::map<size_t, NominalCommDevice *> ids_to_nw_nominal_device;

public:
  std::map<size_t, uint64_t> logical_traffic_demand;
  std::map<size_t, uint64_t> physical_traffic_matrix;
};

struct OpSyncTask {
  Op const *op;
  int unsatisfied_dependencies;
  float finish_time;
};

struct OpSyncTaskEarliestFirst {
  // earliest finish time is first
  bool operator()(OpSyncTask const *lhs, OpSyncTask const *rhs) const {
    return lhs->finish_time > rhs->finish_time;
  }
};

class SimTask {
public:
  enum SimTaskType {
    TASK_FORWARD,
    TASK_BACKWARD,
    TASK_COMM,
    TASK_UPDATE,
    TASK_BARRIER,
    TASK_NOMINAL_COMM,
    TASK_ALLREDUCE
  };
  SimTask();
  void add_next_task(SimTask *task);

public:
  float ready_time, run_time;
  SimTaskType type;
  Device *device;
  MemDevice *mem;
  int counter;
  size_t xfer_size;
  size_t xfer_left;
  std::vector<SimTask *> next_tasks;
  // const char *op_name;
  bool store;
  std::string name;
  std::string get_type_str() const;
};

class SimTaskCompare {
public:
  bool operator()(SimTask *lhs, SimTask *rhs) {
    return lhs->ready_time > rhs->ready_time;
  }
};

class TaskManager {
public:
  TaskManager(size_t max_num_tasks);
  void reset();
  SimTask *new_barrier_task();
  SimTask *new_update_task();
  SimTask *new_comm_task();
  SimTask *new_nominal_comm_task();
  SimTask *new_comm_task(std::string const &name,
                         CommDevice *comm_device,
                         size_t message_size);
  SimTask *new_nominal_comm_task(std::string const &name,
                                 CommDevice *comm_device,
                                 size_t message_size);
  SimTask *new_forward_task(Op const *op, int idx);
  SimTask *new_allreduce_task(Op const *op,
                              std::vector<int> const &node_ids,
                              size_t message_size);
  SimTask *new_backward_task(Op const *op, int idx);
  SimTask *get_forward_task(Op const *op, int idx);
  SimTask *get_backward_task(Op const *op, int idx);

  SimTask *new_task();

public:
  size_t global_task_id, max_num_tasks;
  SimTask **tasks;

  std::map<size_t, SimTask *> hash_to_forward_task, hash_to_backward_task;
};

using ProfilingRecordKey = std::tuple<OperatorParameters, MachineView>;

class Simulator {
public:
  static constexpr float MAXIMUM_TASK_RUN_TIME = 1e7;
  Simulator(FFModel const *model,
            FFHandler handler,
            Legion::Memory memory,
            MachineModel *machine);
  ~Simulator(void);
  void free_all();
  void *allocate(size_t num_elements, DataType type);
  void add_task_dependencies_with_xfer(SimTask *src_task,
                                       SimTask *dst_task,
                                       size_t message_size,
                                       bool force_zero_cost = false);
  CostMetrics measure_operator_cost(Op const *op, ParallelConfig const &config);
  CostMetrics measure_operator_cost(Op const *op, MachineView const &view);
  float estimate_xfer_cost(Op const *op,
                           int input_idx,
                           MachineView const &source_view,
                           MachineView const &sink_view);
  float
      default_estimate_sync_cost(const ParallelDim tensor_dims[MAX_TENSOR_DIM],
                                 int tensor_ndims,
                                 MachineView const &view);
  float default_estimate_sync_cost(ParallelTensorShape const &tensor_shape,
                                   MachineView const &view,
                                   int num_replicate_dims);
  float default_estimate_sync_cost(const ParallelTensor tensor,
                                   MachineView const &view,
                                   int num_replicate_dims);
  float simulate_runtime(FFModel const *model,
                         std::map<Op const *, ParallelConfig> const &global,
                         CompMode comp_mode);
  float simulate_runtime(FFModel const *model,
                         std::map<Op const *, ParallelConfig> const &global,
                         CompMode comp_mode,
                         std::string const &export_file_name);
  static void
      strategy_search_task(Legion::Task const *task,
                           std::vector<Legion::PhysicalRegion> const &regions,
                           Legion::Context ctx,
                           Legion::Runtime *runtime);

public:
  Realm::RegionInstance simulatorInst;
  MachineModel *machine;
  Legion::Memory memory;
  FFHandler handler;
  char *base_ptr;
  size_t capacity;
  off_t offset;
  int warmup_times, repeat_times;
  TaskManager *task_manager;
  CompMode computationMode;
#if defined(FF_USE_CUDA) || defined(FF_USE_HIP_CUDA)
  cudaEvent_t start_event, end_event;
#else
  hipEvent_t start_event, end_event;
#endif
  std::unordered_map<size_t, CostMetrics> hash_to_operator_cost;
  std::unordered_map<ProfilingRecordKey, CostMetrics>
      strict_hash_to_operator_cost;

public:
  Conv2DMeta *conv2d_meta;
  LinearMeta *linear_meta;
  Pool2DMeta *pool2d_meta;
  ElementUnaryMeta *ele_unary_meta;
  LayerNormMeta *layernorm_meta;
  // ElementBinaryMeta *ele_binary_meta;
  // EmbeddingMeta *embedding_meta;
  // SoftmaxMeta *softmax_meta;
  BatchMatmulMeta *batch_matmul_meta;
  // BatchNormMeta *batch_norm_meta;
  ConcatMeta *concat_meta;
  // DropoutMeta *dropout_meta;
  TransposeMeta *transpose_meta;
  int segment_size;
  int max_num_segments; // simulation could be slow if the number of segments
                        // are too large
private:
  float estimate_repartition_xfer_cost(
      int repartition_dim,
      int repartition_degree,
      ParallelTensorShape const &input_tensor_shape,
      ParallelTensorShape const &output_tensor_shape,
      MachineView const &source_view,
      MachineView const &target_view) const;
};

/**
 * An alternative implementation of the simulator which uses the "logical
 * task graph", defined as a taskgraph that only records computation
 * and communication on a logical level.
 */
class LogicalTaskgraphBasedSimulator : public Simulator {
public:
  LogicalTaskgraphBasedSimulator(FFModel const *model,
                                 FFHandler handler,
                                 Legion::Memory memory,
                                 MachineModel *machine);

  SimTask *new_comm_task_unrecorded();
  SimTask *new_update_task_unrecorded();
  virtual float
      simulate_runtime(FFModel const *model,
                       std::map<Op const *, ParallelConfig> const &global,
                       CompMode comp_mode);
  virtual float
      simulate_runtime(FFModel const *model,
                       std::map<Op const *, ParallelConfig> const &global,
                       CompMode comp_mode,
                       std::string const &export_file_name);
  virtual float route_transfer(SimTask *transfer_task,
                               float start_time,
                               std::map<Device *, float> &device_times);
  virtual float route_transfer_seg(SimTask *transfer_task,
                                   float start_time,
                                   std::map<Device *, float> &device_times,
                                   bool &finished);
  virtual void expand_allreduce(
      SimTask *allreduce_task,
      float start_time,
      std::priority_queue<SimTask *, std::vector<SimTask *>, SimTaskCompare>
          &ready_queue);
  void add_task_dependencies_with_xfer(SimTask *src_task,
                                       SimTask *dst_task,
                                       size_t message_size);
  static void
      simulation_task(Legion::Task const *task,
                      std::vector<Legion::PhysicalRegion> const &regions,
                      Legion::Context ctx,
                      Legion::Runtime *runtime);
  bool segment_transfer;
  size_t segment_size;

  // flatbuffers::FlatBufferBuilder builder;
};

}; // namespace FlexFlow
#endif
