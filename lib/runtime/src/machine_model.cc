#include "simulator.h"

#include <algorithm>
#include <functional>
#include <thread>
#include <vector>
namespace FlexFlow {

/// @param[in] nb_elements : size of your for loop
/// @param[in] functor(start, end) :
/// your function processing a sub chunk of the for loop.
/// "start" is the first index to process (included) until the index "end"
/// (excluded)
/// @code
///     for(int i = start; i < end; ++i)
///         computation(i);
/// @endcode
/// @param use_threads : enable / disable threads.
///
///
static void parallel_for(unsigned nb_elements,
                         std::function<void(int start, int end)> functor,
                         bool use_threads = true) {
  // -------
  unsigned nb_threads_hint = std::thread::hardware_concurrency();
  unsigned nb_threads = nb_threads_hint == 0 ? 8 : (nb_threads_hint);

  unsigned batch_size = nb_elements / nb_threads;
  unsigned batch_remainder = nb_elements % nb_threads;

  std::vector<std::thread> my_threads(nb_threads);

  if (use_threads) {
    // Multithread execution
    for (unsigned i = 0; i < nb_threads; ++i) {
      int start = i * batch_size;
      my_threads[i] = std::thread(functor, start, start + batch_size);
    }
  } else {
    // Single thread execution (for easy debugging)
    for (unsigned i = 0; i < nb_threads; ++i) {
      int start = i * batch_size;
      functor(start, start + batch_size);
    }
  }

  // Deform the elements left
  int start = nb_threads * batch_size;
  functor(start, start + batch_remainder);

  // Wait for the other thread to finish their task
  if (use_threads) {
    std::for_each(
        my_threads.begin(), my_threads.end(), std::mem_fn(&std::thread::join));
  }
}

SimpleMachineModel::SimpleMachineModel(int num_nodes,
                                       int num_gpus_per_node,
                                       size_t capacity) {
  version = 0;
  this->num_nodes = num_nodes;
  this->num_gpus_per_node = num_gpus_per_node;
  printf(
      "num_nodes = %d num_gpus_per_node = %d\n", num_nodes, num_gpus_per_node);
  num_gpus = num_nodes * num_gpus_per_node;
  inter_gpu_bandwidth = 20 * 1024 * 1024.0f;              /* B/ms*/
  inter_node_bandwidth = 12 * 1024 * 1024.0f / num_nodes; /* B/ms*/
  gpu_dram_bandwidth = 16 * 1024 * 1024.0f;               /* B/ms*/

  // Create GPU compute device
  for (int i = 0; i < num_nodes; i++) {
    for (int j = 0; j < num_gpus_per_node; j++) {
      int device_id = i * num_gpus_per_node + j;
      std::string gpu_name = "GPU " + std::to_string(device_id);
      id_to_gpu[device_id] =
          new CompDevice(gpu_name, CompDevice::TOC_PROC, i, i, device_id);
      std::string gpu_mem_name = "GPU_FB_MEM " + std::to_string(device_id);
      id_to_gpu_fb_mem[device_id] = new MemDevice(
          gpu_mem_name, MemDevice::GPU_FB_MEM, i, i, device_id, capacity);
    }
  }

  // Create inter GPU comm devices (NVLinks)
  for (int i = 0; i < num_gpus; i++) {
    for (int j = 0; j < num_gpus; j++) {
      Device *src = id_to_gpu[i];
      Device *dst = id_to_gpu[j];
      if (src->node_id == dst->node_id && src != dst) {
        int device_id = i * num_gpus + j;
        std::string nvlink_name = "NVLINK " + std::to_string(device_id);
        ids_to_inter_gpu_comm_device[device_id] =
            new CommDevice(nvlink_name,
                           CommDevice::NVLINK_COMM,
                           src->node_id,
                           src->node_id,
                           device_id,
                           0,
                           inter_gpu_bandwidth);
      }
    }
  }

  // Create gpu<->dram comm devices
  for (int i = 0; i < num_gpus; i++) {
    int node_id = num_gpus / num_gpus_per_node;
    std::string pci_to_host_name = "PCI_TO_HOST " + std::to_string(i);
    id_to_gputodram_comm_device[i] =
        new CommDevice(pci_to_host_name,
                       CommDevice::PCI_TO_HOST_COMM,
                       node_id,
                       node_id,
                       i,
                       0,
                       gpu_dram_bandwidth);
    std::string pci_to_dev_name = "PCI_TO_DEV " + std::to_string(i);
    id_to_dramtogpu_comm_device[i] = new CommDevice(pci_to_dev_name,
                                                    CommDevice::PCI_TO_DEV_COMM,
                                                    node_id,
                                                    node_id,
                                                    i,
                                                    0,
                                                    gpu_dram_bandwidth);
  }

  // Create inter node comm devices
  for (int i = 0; i < num_nodes; i++) {
    for (int j = 0; j < num_nodes; j++) {
      if (i != j) {
        int device_id = i * num_nodes + j;
        std::string nic_name = "NIC " + std::to_string(device_id);
        ids_to_inter_node_comm_device[device_id] =
            new CommDevice(nic_name,
                           CommDevice::NIC_OUT_COMM,
                           -1,
                           -1,
                           device_id,
                           0,
                           inter_node_bandwidth);
      }
    }
  }
}

SimpleMachineModel::~SimpleMachineModel() {}

int SimpleMachineModel::get_version() const {
  return version;
}

CompDevice *SimpleMachineModel::get_gpu(int device_id) const {
  assert(id_to_gpu.find(device_id) != id_to_gpu.end());
  return id_to_gpu.at(device_id);
}

MemDevice *SimpleMachineModel::get_gpu_fb_mem(int device_id) const {
  assert(id_to_gpu_fb_mem.find(device_id) != id_to_gpu_fb_mem.end());
  return id_to_gpu_fb_mem.at(device_id);
}

int SimpleMachineModel::get_num_gpus() const {
  return num_gpus;
}

float SimpleMachineModel::get_intra_node_gpu_bandwidth() const {
  return inter_gpu_bandwidth;
}

float SimpleMachineModel::get_inter_node_gpu_bandwidth() const {
  return inter_node_bandwidth;
}

std::vector<CommDevice *>
    SimpleMachineModel::get_comm_path(MemDevice *src_mem, MemDevice *tar_mem) {
  std::vector<CommDevice *> ret;
  // on the same memory
  if (src_mem->mem_type == tar_mem->mem_type and
      src_mem->device_id == tar_mem->device_id) {
    return ret;
  }
  if (src_mem->mem_type == MemDevice::SYSTEM_MEM and
      tar_mem->mem_type == MemDevice::SYSTEM_MEM) {
    if (src_mem->node_id == tar_mem->node_id) {
      return ret;
    } else {
      int device_id = src_mem->node_id * num_nodes + tar_mem->node_id;
      ret.emplace_back(ids_to_inter_node_comm_device.at(device_id));
    }
  } else if (src_mem->mem_type == MemDevice::GPU_FB_MEM and
             tar_mem->mem_type == MemDevice::GPU_FB_MEM) {
    if (src_mem->node_id == tar_mem->node_id) {
      int device_id = src_mem->device_id * num_gpus + tar_mem->device_id;
      ret.emplace_back(ids_to_inter_gpu_comm_device.at(device_id));
    } else {
      ret.emplace_back(id_to_gputodram_comm_device.at(src_mem->device_id));
      int device_id = src_mem->node_id * num_nodes + tar_mem->node_id;
      ret.emplace_back(ids_to_inter_node_comm_device.at(device_id));
      ret.emplace_back(id_to_dramtogpu_comm_device.at(tar_mem->device_id));
    }
  } else if (src_mem->mem_type == MemDevice::SYSTEM_MEM and
             tar_mem->mem_type == MemDevice::GPU_FB_MEM) {
    if (src_mem->node_id == tar_mem->node_id) {
      ret.emplace_back(id_to_dramtogpu_comm_device.at(tar_mem->device_id));
    } else {
      int device_id = src_mem->node_id * num_nodes + tar_mem->node_id;
      ret.emplace_back(ids_to_inter_node_comm_device.at(device_id));
      ret.emplace_back(id_to_dramtogpu_comm_device.at(tar_mem->device_id));
    }
  } else if (src_mem->mem_type == MemDevice::GPU_FB_MEM and
             tar_mem->mem_type == MemDevice::SYSTEM_MEM) {
    if (src_mem->node_id == tar_mem->node_id) {
      ret.emplace_back(id_to_gputodram_comm_device.at(src_mem->device_id));
    } else {
      ret.emplace_back(id_to_gputodram_comm_device.at(src_mem->device_id));
      int device_id = src_mem->node_id * num_nodes + tar_mem->node_id;
      ret.emplace_back(ids_to_inter_node_comm_device.at(device_id));
    }
  } else {
    printf("No path found between %s and %s\n",
           src_mem->name.c_str(),
           tar_mem->name.c_str());
    assert(false);
  }
  return ret;
}

std::string SimpleMachineModel::to_string() const {
  std::string s;
  for (int i = 0; i < num_nodes; i++) {
    int node_id = i;
    s += "==========================================\n";
    s += "Node " + std::to_string(node_id) + '\n';
    s += "COMP: \n";
    for (int j = 0; j < num_gpus_per_node; j++) {
      int device_id = i * num_gpus_per_node + j;
      s += id_to_gpu.at(device_id)->name + '\n';
    }
    s += '\n';
    s += "MEM: \n";
    for (int j = 0; j < num_gpus_per_node; j++) {
      int device_id = i * num_gpus_per_node + j;
      s += id_to_gpu_fb_mem.at(device_id)->name + '\n';
    }
  }
  return s;
}

EnhancedMachineModel::EnhancedMachineModel(std::string file,
                                           size_t gpu_fb_mem_capacity) {
  version = 1;
  this->gpu_fb_mem_capacity = gpu_fb_mem_capacity;
  std::ifstream machine_config(file);
  std::string line;
  while (std::getline(machine_config, line)) {
    if (line[0] != '#') {
      // split a line into words
      std::istringstream iss(line);
      std::vector<std::string> words{std::istream_iterator<std::string>{iss},
                                     std::istream_iterator<std::string>{}};
      if (words.size() >= 3) {
        if (words[0] == "num_nodes") {
          num_nodes = stoi(words[2]);
          printf("num_nodes = %d\n", num_nodes);
        } else if (words[0] == "num_sockets_per_node") {
          num_sockets_per_node = stoi(words[2]);
          printf("num_sockets_per_node = %d\n", num_sockets_per_node);
        } else if (words[0] == "num_cpus_per_socket") {
          num_cpus_per_socket = stoi(words[2]);
          printf("num_cpus_per_socket = %d\n", num_cpus_per_socket);
        } else if (words[0] == "num_gpus_per_socket") {
          num_gpus_per_socket = stoi(words[2]);
          printf("num_gpus_per_socket = %d\n", num_gpus_per_socket);
        } else if (words[0] == "membus_latency") {
          membus_latency = stof(words[2]);
          printf("membus_latency = %f\n", membus_latency);
        } else if (words[0] == "membus_bandwidth") {
          membus_bandwidth = stof(words[2]);
          printf("membus_bandwidth = %f\n", membus_bandwidth);
        } else if (words[0] == "upi_latency") {
          upi_latency = stof(words[2]);
          printf("upi_latency = %f\n", upi_latency);
        } else if (words[0] == "upi_bandwidth") {
          upi_bandwidth = stof(words[2]);
          printf("upi_bandwidth = %f\n", upi_bandwidth);
        } else if (words[0] == "nic_latency") {
          nic_latency = stof(words[2]);
          printf("nic_latency = %f\n", nic_latency);
        } else if (words[0] == "nic_bandwidth") {
          nic_bandwidth = stof(words[2]);
          printf("nic_bandwidth = %f\n", nic_bandwidth);
        } else if (words[0] == "nic_persocket") {
          nic_persocket = stoi(words[2]);
          printf("nic_persocket = %d\n", nic_persocket);
        } else if (words[0] == "pci_latency") {
          pci_latency = stof(words[2]);
          printf("pci_latency = %f\n", pci_latency);
        } else if (words[0] == "pci_bandwidth") {
          pci_bandwidth = stof(words[2]);
          printf("pci_bandwidth = %f\n", pci_bandwidth);
        } else if (words[0] == "nvlink_latency") {
          nvlink_latency = stof(words[2]);
          printf("nvlink_latency = %f\n", nvlink_latency);
        } else if (words[0] == "nvlink_bandwidth") {
          nvlink_bandwidth = stof(words[2]);
          printf("nvlink_bandwidth = %f\n", nvlink_bandwidth);
        } else if (words[0] == "intra_socket_sys_mem_to_sys_mem") {
          printf("intra_socket_sys_mem_to_sys_mem = ");
          for (size_t i = 2; i < words.size(); i++) {
            set_comm_path(intra_socket_sys_mem_to_sys_mem, words[i]);
            printf("%s ", words[i].c_str());
          }
          printf("\n");
        } else if (words[0] == "inter_socket_sys_mem_to_sys_mem") {
          printf("inter_socket_sys_mem_to_sys_mem = ");
          for (size_t i = 2; i < words.size(); i++) {
            set_comm_path(inter_socket_sys_mem_to_sys_mem, words[i]);
            printf("%s ", words[i].c_str());
          }
          printf("\n");
        } else if (words[0] == "inter_node_sys_mem_to_sys_mem") {
          printf("inter_node_sys_mem_to_sys_mem = ");
          for (size_t i = 2; i < words.size(); i++) {
            set_comm_path(inter_node_sys_mem_to_sys_mem, words[i]);
            printf("%s ", words[i].c_str());
          }
          printf("\n");
        } else if (words[0] == "intra_socket_gpu_fb_mem_to_gpu_fb_mem") {
          printf("intra_socket_gpu_fb_mem_to_gpu_fb_mem = ");
          for (size_t i = 2; i < words.size(); i++) {
            set_comm_path(intra_socket_gpu_fb_mem_to_gpu_fb_mem, words[i]);
            printf("%s ", words[i].c_str());
          }
          printf("\n");
        } else if (words[0] == "inter_socket_gpu_fb_mem_to_gpu_fb_mem") {
          printf("inter_socket_gpu_fb_mem_to_gpu_fb_mem = ");
          for (size_t i = 2; i < words.size(); i++) {
            set_comm_path(inter_socket_gpu_fb_mem_to_gpu_fb_mem, words[i]);
            printf("%s ", words[i].c_str());
          }
          printf("\n");
        } else if (words[0] == "inter_node_gpu_fb_mem_to_gpu_fb_mem") {
          printf("inter_node_gpu_fb_mem_to_gpu_fb_mem = ");
          for (size_t i = 2; i < words.size(); i++) {
            set_comm_path(inter_node_gpu_fb_mem_to_gpu_fb_mem, words[i]);
            printf("%s ", words[i].c_str());
          }
          printf("\n");
        } else if (words[0] == "intra_socket_sys_mem_to_gpu_fb_mem") {
          printf("intra_socket_sys_mem_to_gpu_fb_mem = ");
          for (size_t i = 2; i < words.size(); i++) {
            set_comm_path(intra_socket_sys_mem_to_gpu_fb_mem, words[i]);
            printf("%s ", words[i].c_str());
          }
          printf("\n");
        } else if (words[0] == "inter_socket_sys_mem_to_gpu_fb_mem") {
          printf("inter_socket_sys_mem_to_gpu_fb_mem = ");
          for (size_t i = 2; i < words.size(); i++) {
            set_comm_path(inter_socket_sys_mem_to_gpu_fb_mem, words[i]);
            printf("%s ", words[i].c_str());
          }
          printf("\n");
        } else if (words[0] == "inter_node_sys_mem_to_gpu_fb_mem") {
          printf("inter_node_sys_mem_to_gpu_fb_mem = ");
          for (size_t i = 2; i < words.size(); i++) {
            set_comm_path(inter_node_sys_mem_to_gpu_fb_mem, words[i]);
            printf("%s ", words[i].c_str());
          }
          printf("\n");
        } else if (words[0] == "intra_socket_gpu_fb_mem_to_sys_mem") {
          printf("intra_socket_gpu_fb_mem_to_sys_mem = ");
          for (size_t i = 2; i < words.size(); i++) {
            set_comm_path(intra_socket_gpu_fb_mem_to_sys_mem, words[i]);
            printf("%s ", words[i].c_str());
          }
          printf("\n");
        } else if (words[0] == "inter_socket_gpu_fb_mem_to_sys_mem") {
          printf("inter_socket_gpu_fb_mem_to_sys_mem = ");
          for (size_t i = 2; i < words.size(); i++) {
            set_comm_path(inter_socket_gpu_fb_mem_to_sys_mem, words[i]);
            printf("%s ", words[i].c_str());
          }
          printf("\n");
        } else if (words[0] == "inter_node_gpu_fb_mem_to_sys_mem") {
          printf("inter_node_gpu_fb_mem_to_sys_mem = ");
          for (size_t i = 2; i < words.size(); i++) {
            set_comm_path(inter_node_gpu_fb_mem_to_sys_mem, words[i]);
            printf("%s ", words[i].c_str());
          }
          printf("\n");
        }
      }
    }
  }

  num_sockets = num_nodes * num_sockets_per_node;
  num_cpus = num_sockets * num_cpus_per_socket;
  num_gpus = num_sockets * num_gpus_per_socket;
  cur_nic_local_id = 0;
  num_nvlinks_per_node = 0;
  mem_to_nvlink.clear();
  this->add_cpus();
  this->add_gpus();
  this->add_membuses(membus_latency, membus_bandwidth * 1024 * 1024);
  this->add_upis(upi_latency / 2, upi_bandwidth * 2 * 1024 * 1024);
  this->add_nics(
      nic_latency / 2, nic_bandwidth * 2 * 1024 * 1024, nic_persocket);
  this->add_pcis(pci_latency, pci_bandwidth * 1024 * 1024);
  this->add_nvlinks(nvlink_latency, nvlink_bandwidth * 1024 * 1024);
  //   printf("%s", this->to_string().c_str());
}

EnhancedMachineModel::~EnhancedMachineModel() {}

int EnhancedMachineModel::get_version() const {
  return version;
}

void EnhancedMachineModel::set_comm_path(
    std::vector<CommDevice::CommDevType> &comm_path, std::string device_str) {
  if (device_str == "membus") {
    comm_path.emplace_back(CommDevice::MEMBUS_COMM);
  } else if (device_str == "upi") {
    comm_path.emplace_back(CommDevice::UPI_OUT_COMM);
    comm_path.emplace_back(CommDevice::UPI_IN_COMM);
  } else if (device_str == "nic") {
    comm_path.emplace_back(CommDevice::NIC_OUT_COMM);
    comm_path.emplace_back(CommDevice::NIC_IN_COMM);
  } else if (device_str == "pci_to_host") {
    comm_path.emplace_back(CommDevice::PCI_TO_HOST_COMM);
  } else if (device_str == "pci_to_dev") {
    comm_path.emplace_back(CommDevice::PCI_TO_DEV_COMM);
  } else if (device_str == "nvlink") {
    comm_path.emplace_back(CommDevice::NVLINK_COMM);
  }
}

void EnhancedMachineModel::add_cpus() {
  for (int i = 0; i < num_nodes; i++) {
    int node_id = i;
    for (int j = 0; j < num_sockets_per_node; j++) {
      int socket_id = i * num_sockets_per_node + j;
      int device_id = socket_id;
      // add system memory
      std::string sys_mem_name = "SYSTEM_MEM " + std::to_string(device_id);
      MemDevice *sys_mem = new MemDevice(sys_mem_name,
                                         MemDevice::SYSTEM_MEM,
                                         node_id,
                                         socket_id,
                                         device_id,
                                         -1);
      sys_mems.emplace_back(sys_mem);
      // add cpus
      cpus.push_back({});
      for (int k = 0; k < num_cpus_per_socket; k++) {
        device_id = socket_id * num_cpus_per_socket + k;
        std::string cpu_name = "CPU " + std::to_string(device_id);
        cpus[socket_id].emplace_back(new CompDevice(
            cpu_name, CompDevice::LOC_PROC, node_id, socket_id, device_id));
      }
    }
  }
}

void EnhancedMachineModel::add_gpus() {
  for (int i = 0; i < num_nodes; i++) {
    int node_id = i;
    for (int j = 0; j < num_sockets_per_node; j++) {
      int socket_id = i * num_sockets_per_node + j;
      int device_id = socket_id;
      // add zero copy memory
      std::string z_copy_mem_name = "Z_COPY_MEM " + std::to_string(device_id);
      MemDevice *z_copy_mem = new MemDevice(z_copy_mem_name,
                                            MemDevice::Z_COPY_MEM,
                                            node_id,
                                            socket_id,
                                            device_id,
                                            -1);
      z_copy_mems.push_back(z_copy_mem);
      // add gpus and gpu framebuffer memories
      gpus.push_back({});
      gpu_fb_mems.push_back({});
      for (int k = 0; k < num_gpus_per_socket; k++) {
        device_id = socket_id * num_gpus_per_socket + k;
        std::string gpu_name = "GPU " + std::to_string(device_id);
        gpus[socket_id].push_back(new CompDevice(
            gpu_name, CompDevice::TOC_PROC, node_id, socket_id, device_id));
        std::string gpu_mem_name = "GPU_FB_MEM " + std::to_string(device_id);
        MemDevice *gpu_mem = new MemDevice(gpu_mem_name,
                                           MemDevice::GPU_FB_MEM,
                                           node_id,
                                           socket_id,
                                           device_id,
                                           gpu_fb_mem_capacity);
        gpu_fb_mems[socket_id].push_back({gpu_mem});
      }
    }
  }
}

void EnhancedMachineModel::add_membuses(float latency, float bandwidth) {
  for (int i = 0; i < num_nodes; i++) {
    int node_id = i;
    for (int j = 0; j < num_sockets_per_node; j++) {
      int socket_id = i * num_sockets_per_node + j;
      int device_id = socket_id;
      std::string membus_name = "MEMBUS " + std::to_string(device_id);
      CommDevice *membus = new CommDevice(membus_name,
                                          CommDevice::MEMBUS_COMM,
                                          node_id,
                                          socket_id,
                                          device_id,
                                          latency,
                                          bandwidth);
      membuses.push_back(membus);
    }
  }
}

void EnhancedMachineModel::add_upis(float latency, float bandwidth) {
  for (int i = 0; i < num_nodes; i++) {
    int node_id = i;
    for (int j = 0; j < num_sockets_per_node; j++) {
      int socket_id = i * num_sockets_per_node + j;
      int device_id = socket_id;
      std::string upi_in_name = "UPI_IN " + std::to_string(device_id);
      CommDevice *upi_in = new CommDevice(upi_in_name,
                                          CommDevice::UPI_IN_COMM,
                                          node_id,
                                          socket_id,
                                          device_id,
                                          latency,
                                          bandwidth);
      upi_ins.push_back(upi_in);
      std::string upi_out_name = "UPI_OUT " + std::to_string(device_id);
      CommDevice *upi_out = new CommDevice(upi_out_name,
                                           CommDevice::UPI_OUT_COMM,
                                           node_id,
                                           socket_id,
                                           device_id,
                                           latency,
                                           bandwidth);
      upi_outs.push_back(upi_out);
    }
  }
}

void EnhancedMachineModel::add_nics(float latency,
                                    float bandwidth,
                                    int nic_persocket) {
  if (nic_persocket == 0) {
    for (int i = 0; i < num_nodes; i++) {
      int node_id = i;
      for (int j = 0; j < num_sockets_per_node; j++) {
        int socket_id = i * num_sockets_per_node + j;
        int device_id = socket_id;
        CommDevice *nic_in;
        CommDevice *nic_out;
        if (j == 0) {
          std::string nic_in_name = "NIC_IN " + std::to_string(device_id);
          nic_in = new CommDevice(nic_in_name,
                                  CommDevice::NIC_IN_COMM,
                                  node_id,
                                  socket_id,
                                  device_id,
                                  latency,
                                  bandwidth);
          nic_ins.push_back({});
          nic_ins[socket_id].push_back(nic_in);
          std::string nic_out_name = "NIC_OUT " + std::to_string(device_id);
          nic_out = new CommDevice(nic_out_name,
                                   CommDevice::NIC_OUT_COMM,
                                   node_id,
                                   socket_id,
                                   device_id,
                                   latency,
                                   bandwidth);
          nic_outs.push_back({});
          nic_outs[socket_id].push_back(nic_out);
        } else {
          nic_ins.push_back({});
          nic_ins[socket_id].push_back(nic_in);
          nic_outs.push_back({});
          nic_outs[socket_id].push_back(nic_out);
        }
      }
    }
  } else {
    for (int i = 0; i < num_nodes; i++) {
      int node_id = i;
      for (int j = 0; j < num_sockets_per_node; j++) {
        int socket_id = i * num_sockets_per_node + j;
        nic_ins.push_back({});
        nic_outs.push_back({});
        for (int k = 0; k < nic_persocket; k++) {
          int device_id = socket_id * nic_persocket + k;
          std::string nic_in_name = "NIC_IN " + std::to_string(device_id);
          CommDevice *nic_in = new CommDevice(nic_in_name,
                                              CommDevice::NIC_IN_COMM,
                                              node_id,
                                              socket_id,
                                              device_id,
                                              latency,
                                              bandwidth);
          nic_ins[socket_id].push_back(nic_in);
          std::string nic_out_name = "NIC_OUT " + std::to_string(device_id);
          CommDevice *nic_out = new CommDevice(nic_out_name,
                                               CommDevice::NIC_OUT_COMM,
                                               node_id,
                                               socket_id,
                                               device_id,
                                               latency,
                                               bandwidth);
          nic_outs[socket_id].push_back(nic_out);
        }
      }
    }
  }
}

void EnhancedMachineModel::add_pcis(float latency, float bandwidth) {
  for (int i = 0; i < num_nodes; i++) {
    int node_id = i;
    for (int j = 0; j < num_sockets_per_node; j++) {
      int socket_id = i * num_sockets_per_node + j;
      int device_id = socket_id;
      std::string pci_to_host_name =
          "PCI_TO_HOST " + std::to_string(device_id); // pcie to memory
      CommDevice *pci_to_host = new CommDevice(pci_to_host_name,
                                               CommDevice::PCI_TO_HOST_COMM,
                                               node_id,
                                               socket_id,
                                               socket_id,
                                               latency,
                                               bandwidth);
      pcis_to_host.push_back(pci_to_host);
      std::string pci_to_dev_name =
          "PCI_TO_DEV " + std::to_string(device_id); // memory to pcie
      CommDevice *pci_to_dev = new CommDevice(pci_to_dev_name,
                                              CommDevice::PCI_TO_DEV_COMM,
                                              node_id,
                                              socket_id,
                                              socket_id,
                                              latency,
                                              bandwidth);
      pcis_to_device.push_back(pci_to_dev);
    }
  }
}

// assume each GPU has nvlinks to the other GPUs on the same node and the
// nvlinks have the same latency and bandwidth
void EnhancedMachineModel::add_nvlinks(float latency, float bandwidth) {
  int num_gpus_per_node = num_gpus_per_socket * num_sockets_per_node;
  num_nvlinks_per_node = num_gpus_per_node * (num_gpus_per_node - 1) / 2;
  for (int i = 0; i < num_nodes; i++) {
    int node_id = i;
    int socket_id = i * num_sockets_per_node;
    nvlinks.push_back({});
    for (int j = 0; j < num_nvlinks_per_node * 2; j++) {
      int nvlink_id = node_id * num_nvlinks_per_node * 2 + j;
      std::string nvlink_name = "NVLINK " + std::to_string(nvlink_id);
      nvlinks[i].push_back(new CommDevice(nvlink_name,
                                          CommDevice::NVLINK_COMM,
                                          node_id,
                                          socket_id,
                                          nvlink_id,
                                          latency,
                                          bandwidth));
    }

    for (int j = 0; j < num_sockets_per_node; j++) {
      int src_socket_id = i * num_sockets_per_node + j;
      for (int k = 0; k < num_gpus_per_socket; k++) {
        MemDevice *src_gpu_fb_mem = gpu_fb_mems[src_socket_id][k];
        int src_local_id = j * num_gpus_per_socket + k;
        for (int l = 0; l < num_sockets_per_node; l++) {
          int tar_socket_id = i * num_sockets_per_node + l;
          for (int m = 0; m < num_gpus_per_socket; m++) {
            MemDevice *tar_gpu_fb_mem = gpu_fb_mems[tar_socket_id][m];
            int tar_local_id = l * num_gpus_per_socket + m;
            if (src_local_id != tar_local_id) {
              int local_nvlink_id =
                  src_local_id * (num_gpus_per_node - 1) + tar_local_id;
              if (tar_local_id > src_local_id) {
                local_nvlink_id--;
              }
              attach_nvlink(
                  src_gpu_fb_mem, tar_gpu_fb_mem, nvlinks[i][local_nvlink_id]);
              printf(
                  "add nvlink: gpu_fb_mem %d , gpu_fb_mem %d, nvlink %d %d\n",
                  src_gpu_fb_mem->device_id,
                  tar_gpu_fb_mem->device_id,
                  node_id,
                  local_nvlink_id);
            }
          }
        }
      }
    }
  }
}

void EnhancedMachineModel::attach_nvlink(MemDevice *src_mem,
                                         MemDevice *tar_mem,
                                         CommDevice *comm) {
  assert(comm->comm_type == CommDevice::NVLINK_COMM);
  int hash = src_mem->device_id * num_gpus + tar_mem->device_id;
  if (mem_to_nvlink.find(hash) == mem_to_nvlink.end()) {
    mem_to_nvlink[hash] = comm;
  }
}

CompDevice *EnhancedMachineModel::get_cpu(int device_id) const {
  return get_cpu(device_id / num_cpus_per_socket,
                 device_id % num_cpus_per_socket);
}

CompDevice *EnhancedMachineModel::get_cpu(int socket_id, int local_id) const {
  if (socket_id < num_sockets and local_id < num_cpus_per_socket) {
    return cpus[socket_id][local_id];
  } else {
    printf("MachineModel: get_cpu - cannot find cpu (%d %d)\n",
           socket_id,
           local_id);
    assert(false);
  }
}

CompDevice *EnhancedMachineModel::get_gpu(int device_id) const {
  return get_gpu(device_id / num_gpus_per_socket,
                 device_id % num_gpus_per_socket);
}

CompDevice *EnhancedMachineModel::get_gpu(int socket_id, int local_id) const {
  if (socket_id < num_sockets and local_id < num_cpus_per_socket) {
    return gpus[socket_id][local_id];
  } else {
    printf("MachineModel: get_gpu - cannot find gpu (%d %d)\n",
           socket_id,
           local_id);
    assert(false);
  }
}

MemDevice *EnhancedMachineModel::get_sys_mem(int socket_id) const {
  return sys_mems[socket_id];
}

MemDevice *EnhancedMachineModel::get_z_copy_mem(int socket_id) const {
  return z_copy_mems[socket_id];
}

MemDevice *EnhancedMachineModel::get_gpu_fb_mem(int device_id) const {
  return get_gpu_fb_mem(device_id / num_gpus_per_socket,
                        device_id % num_gpus_per_socket);
}

MemDevice *EnhancedMachineModel::get_gpu_fb_mem(int socket_id,
                                                int local_id) const {
  if (socket_id < num_sockets and local_id < num_cpus_per_socket) {
    return gpu_fb_mems[socket_id][local_id];
  } else {
    printf("MachineModel: get_gpu_fb_mem - cannot find gpu_fb_mem (%d %d)\n",
           socket_id,
           local_id);
    assert(false);
  }
}

CommDevice *EnhancedMachineModel::get_nvlink(MemDevice *src_mem,
                                             MemDevice *tar_mem) const {
  int hash = src_mem->device_id * num_gpus + tar_mem->device_id;
  if (mem_to_nvlink.find(hash) != mem_to_nvlink.end()) {
    return mem_to_nvlink.at(hash);
  } else {
    printf("MachineModel: get_nvlink - cannot get nvlink between %s and %s\n",
           src_mem->name.c_str(),
           tar_mem->name.c_str());
    assert(false);
  }
}

CommDevice *EnhancedMachineModel::get_next_nic_in(int socket_id) {
  if (nic_persocket == 0) {
    return nic_ins[socket_id][0];
  }
  if (socket_id < num_sockets) {
    CommDevice *ret = nic_ins[socket_id][cur_nic_local_id];
    cur_nic_local_id = (cur_nic_local_id + 1) % nic_persocket;
    return ret;
  } else {
    printf("MachineModel: get_next_nic_in - cannot find next nic_in socket_id "
           "%d cur_nic_local_id %d\n",
           socket_id,
           cur_nic_local_id);
    assert(false);
  }
}

CommDevice *EnhancedMachineModel::get_next_nic_out(int socket_id) const {
  if (nic_persocket == 0) {
    return nic_outs[socket_id][0];
  }
  if (socket_id < num_sockets) {
    return nic_outs[socket_id][cur_nic_local_id];
  } else {
    printf("MachineModel: get_next_nic_out - cannot find next nic_out "
           "socket_id %d cur_nic_local_id %d\n",
           socket_id,
           cur_nic_local_id);
    assert(false);
  }
}
int EnhancedMachineModel::get_num_gpus() const {
  return num_gpus;
}

void EnhancedMachineModel::add_comm_path(
    std::vector<CommDevice::CommDevType> const &comm_device_list,
    MemDevice *src_mem,
    MemDevice *tar_mem,
    std::vector<CommDevice *> &ret) {
  MemDevice *cur_mem = src_mem;
  for (size_t i = 0; i < comm_device_list.size(); i++) {
    switch (comm_device_list[i]) {
      case CommDevice::MEMBUS_COMM:
        ret.emplace_back(membuses[cur_mem->socket_id]);
        break;
      case CommDevice::UPI_IN_COMM:
        cur_mem = tar_mem;
        ret.emplace_back(upi_ins[cur_mem->socket_id]);
        break;
      case CommDevice::UPI_OUT_COMM:
        ret.emplace_back(upi_outs[cur_mem->socket_id]);
        break;
      case CommDevice::NIC_IN_COMM:
        cur_mem = tar_mem;
        ret.emplace_back(get_next_nic_in(cur_mem->socket_id));
        break;
      case CommDevice::NIC_OUT_COMM:
        ret.emplace_back(get_next_nic_out(cur_mem->socket_id));
        break;
      case CommDevice::PCI_TO_HOST_COMM:
        ret.emplace_back(pcis_to_host[cur_mem->socket_id]);
        break;
      case CommDevice::PCI_TO_DEV_COMM:
        ret.emplace_back(pcis_to_device[cur_mem->socket_id]);
        break;
      case CommDevice::NVLINK_COMM:
        ret.emplace_back(get_nvlink(src_mem, tar_mem));
        break;
      default:
        break;
    }
  }
}

std::vector<CommDevice *>
    EnhancedMachineModel::get_comm_path(MemDevice *src_mem,
                                        MemDevice *tar_mem) {
  std::vector<CommDevice *> ret;
  if (src_mem->device_id == tar_mem->device_id) {
    return ret;
  }
  if (src_mem->mem_type == MemDevice::SYSTEM_MEM and
      tar_mem->mem_type == MemDevice::SYSTEM_MEM) {
    if (src_mem->socket_id == tar_mem->socket_id) {
      add_comm_path(intra_socket_sys_mem_to_sys_mem, src_mem, tar_mem, ret);
    } else if (src_mem->node_id == tar_mem->node_id) {
      add_comm_path(inter_socket_sys_mem_to_sys_mem, src_mem, tar_mem, ret);
    } else {
      add_comm_path(inter_node_sys_mem_to_sys_mem, src_mem, tar_mem, ret);
    }
  } else if (src_mem->mem_type == MemDevice::SYSTEM_MEM and
             tar_mem->mem_type == MemDevice::GPU_FB_MEM) {
    if (src_mem->socket_id == tar_mem->socket_id) {
      add_comm_path(intra_socket_sys_mem_to_gpu_fb_mem, src_mem, tar_mem, ret);
    } else if (src_mem->node_id == tar_mem->node_id) {
      add_comm_path(inter_socket_sys_mem_to_gpu_fb_mem, src_mem, tar_mem, ret);
    } else {
      add_comm_path(inter_node_sys_mem_to_gpu_fb_mem, src_mem, tar_mem, ret);
    }
  } else if (src_mem->mem_type == MemDevice::GPU_FB_MEM and
             tar_mem->mem_type == MemDevice::SYSTEM_MEM) {
    if (src_mem->socket_id == tar_mem->socket_id) {
      add_comm_path(intra_socket_gpu_fb_mem_to_sys_mem, src_mem, tar_mem, ret);
    } else if (src_mem->node_id == tar_mem->node_id) {
      add_comm_path(inter_socket_gpu_fb_mem_to_sys_mem, src_mem, tar_mem, ret);
    } else {
      add_comm_path(inter_node_gpu_fb_mem_to_sys_mem, src_mem, tar_mem, ret);
    }
  } else if (src_mem->mem_type == MemDevice::GPU_FB_MEM and
             tar_mem->mem_type == MemDevice::GPU_FB_MEM) {
    if (src_mem->socket_id == tar_mem->socket_id) {
      add_comm_path(
          intra_socket_gpu_fb_mem_to_gpu_fb_mem, src_mem, tar_mem, ret);
    } else if (src_mem->node_id == tar_mem->node_id) {
      add_comm_path(
          inter_socket_gpu_fb_mem_to_gpu_fb_mem, src_mem, tar_mem, ret);
    } else {
      add_comm_path(inter_node_gpu_fb_mem_to_gpu_fb_mem, src_mem, tar_mem, ret);
    }
  } else {
    printf("MachineModel: get_comm_path - no path found between %s and %s\n",
           src_mem->name.c_str(),
           tar_mem->name.c_str());
    assert(false);
  }
  return ret;
}

float EnhancedMachineModel::get_intra_node_gpu_bandwidth() const {
  return nvlink_bandwidth;
}

// Use inter-node cpu bandwidth for now
float EnhancedMachineModel::get_inter_node_gpu_bandwidth() const {
  return nic_bandwidth;
}

std::string EnhancedMachineModel::to_string() const {
  std::string s;
  for (int i = 0; i < num_nodes; i++) {
    int node_id = i;
    s += "==========================================\n";
    s += "Node " + std::to_string(node_id) + '\n';
    for (int j = 0; j < num_sockets_per_node; j++) {
      s += "------------------------------------------\n";
      int socket_id = i * num_sockets_per_node + j;
      s += "Socket " + std::to_string(socket_id) + '\n';
      s += "COMP: \n";
      for (int k = 0; k < num_cpus_per_socket; k++) {
        s += cpus[socket_id][k]->name + '\n';
      }
      for (int k = 0; k < num_gpus_per_socket; k++) {
        s += gpus[socket_id][k]->name + '\n';
      }
      s += '\n';
      s += "MEM: \n";
      s += sys_mems[socket_id]->name + '\n';
      s += z_copy_mems[socket_id]->name + '\n';
      for (int k = 0; k < num_gpus_per_socket; k++) {
        s += gpu_fb_mems[socket_id][k]->name + '\n';
      }
      s += '\n';
      s += "COMM: \n";
      s += membuses[socket_id]->name + '\n';
      s += upi_ins[socket_id]->name + '\n';
      s += upi_outs[socket_id]->name + '\n';
      s += pcis_to_host[socket_id]->name + '\n';
      s += pcis_to_device[socket_id]->name + '\n';
      for (int k = 0; k < nic_persocket; k++) {
        s += nic_ins[socket_id][k]->name + '\n';
        s += nic_outs[socket_id][k]->name + '\n';
      }
    }
    s += "------------------------------------------\n";
    for (int j = 0; j < num_nvlinks_per_node * 2; j++) {
      s += nvlinks[node_id][j]->name + '\n';
    }
  }
  return s;
}

/* Networked machine model */
NetworkedMachineModel::NetworkedMachineModel(int num_nodes,
                                             int num_gpus_per_node,
                                             int num_switches,
                                             float network_latency,
                                             std::vector<int> const &topology,
                                             size_t capacity,
                                             float link_bandwidth)
    : num_nodes(num_nodes), num_gpus_per_node(num_gpus_per_node),
      num_switches(num_switches), link_bandwidth(link_bandwidth),
      network_latency(network_latency), conn_matrix(topology) {
  version = 0;

  num_gpus = num_nodes * num_gpus_per_node;
  inter_gpu_bandwidth = 20 * 1024 * 1024.0f; /* B/ms*/
  gpu_dram_bandwidth = 32 * 1024 * 1024.0f;

  // Create GPU compute device
  for (int i = 0; i < num_nodes; i++) {
    for (int j = 0; j < num_gpus_per_node; j++) {
      int device_id = i * num_gpus_per_node + j;
      std::string gpu_name = "GPU " + std::to_string(device_id);
      id_to_gpu[device_id] =
          new CompDevice(gpu_name, CompDevice::TOC_PROC, i, i, device_id);
      std::string gpu_mem_name = "GPU_FB_MEM " + std::to_string(device_id);
      id_to_gpu_fb_mem[device_id] = new MemDevice(
          gpu_mem_name, MemDevice::GPU_FB_MEM, i, i, device_id, capacity);
    }
  }

  // Create inter GPU comm devices (NVLinks)
  for (int i = 0; i < num_gpus; i++) {
    for (int j = 0; j < num_gpus; j++) {
      Device *src = id_to_gpu[i];
      Device *dst = id_to_gpu[j];
      if (src->node_id == dst->node_id && src != dst) {
        int device_id = i * num_gpus + j;
        std::string nvlink_name = "NVLINK " + std::to_string(device_id);
        ids_to_inter_gpu_comm_device[device_id] =
            new CommDevice(nvlink_name,
                           CommDevice::NVLINK_COMM,
                           src->node_id,
                           src->node_id,
                           device_id,
                           0,
                           inter_gpu_bandwidth);
      }
    }
  }

  // Create gpu<->dram comm devices
  // if (pcie_on) {
  for (int i = 0; i < num_gpus; i++) {
    int node_id = num_gpus / num_gpus_per_node;
    std::string pci_to_host_name = "PCI_TO_HOST " + std::to_string(i);
    id_to_gputodram_comm_device[i] =
        new CommDevice(pci_to_host_name,
                       CommDevice::PCI_TO_HOST_COMM,
                       node_id,
                       node_id,
                       i,
                       0,
                       gpu_dram_bandwidth);
    std::string pci_to_dev_name = "PCI_TO_DEV " + std::to_string(i);
    id_to_dramtogpu_comm_device[i] = new CommDevice(pci_to_dev_name,
                                                    CommDevice::PCI_TO_DEV_COMM,
                                                    node_id,
                                                    node_id,
                                                    i,
                                                    0,
                                                    gpu_dram_bandwidth);
  }
  // }

  // network links
  total_devs = num_nodes + num_switches;
  for (int i = 0; i < total_devs; i++) {
    for (int j = 0; j < total_devs; j++) {
      // if (conn_matrix[i * total_devs + j] > 0) {
      int device_id = i * total_devs + j;
      std::string link_name =
          "LINK " + std::to_string(i) + "-" + std::to_string(j);
      ids_to_nw_comm_device[device_id] =
          new CommDevice(link_name,
                         CommDevice::NW_COMM,
                         -1,
                         -1,
                         device_id,
                         0,
                         conn_matrix[i * total_devs + j] * link_bandwidth);
    }
    // }
  }

  routing_strategy = new ShortestPathNetworkRoutingStrategy(
      conn_matrix, ids_to_nw_comm_device, total_devs);
  update_route();
}

NetworkedMachineModel::~NetworkedMachineModel() {
  delete routing_strategy;
}

int NetworkedMachineModel::get_version() const {
  return version;
}

void NetworkedMachineModel::update_route() {
  // nominal network links
  int total_devs = num_nodes + num_switches;
  for (int i = 0; i < num_nodes; i++) {
    for (int j = 0; j < num_nodes; j++) {
      int device_id = i * total_devs + j;
      std::string link_name =
          "NOMINAL " + std::to_string(i) + "-" + std::to_string(j);
      if (ids_to_nw_nominal_device.find(device_id) ==
          ids_to_nw_nominal_device.end()) {
        ids_to_nw_nominal_device[device_id] = new NominalCommDevice(
            link_name, device_id, total_devs, routing_strategy);
      }
      ids_to_nw_nominal_device[device_id]->reset();
      // ids_to_nw_nominal_device[device_id]->set_physical_paths(routing_strategy->get_routes(i,
      // j));
    }
  }

  // std::vector<int> indicies(num_nodes);
  // std::iota(ivec.begin(), ivec.end(), 0);

  // for (int i = 0; i < num_nodes; i++) {
  parallel_for(num_nodes, [&](int start, int end) {
    for (int i = start; i < end; i++) {
      // std::cerr << "node i " << i << std::endl;
      auto all_routes = routing_strategy->get_routes_from_src(i);
      for (int j = 0; j < num_nodes; j++) {
        ids_to_nw_nominal_device[i * total_devs + j]->set_physical_paths(
            all_routes[j]);
      }
    }
  });
}

CompDevice *NetworkedMachineModel::get_gpu(int device_id) const {
  assert(id_to_gpu.find(device_id) != id_to_gpu.end());
  return id_to_gpu.at(device_id);
}

MemDevice *NetworkedMachineModel::get_gpu_fb_mem(int device_id) const {
  assert(id_to_gpu_fb_mem.find(device_id) != id_to_gpu_fb_mem.end());
  return id_to_gpu_fb_mem.at(device_id);
}

int NetworkedMachineModel::get_num_gpus() const {
  return num_gpus;
}

float NetworkedMachineModel::get_intra_node_gpu_bandwidth() const {
  return inter_gpu_bandwidth;
}

float NetworkedMachineModel::get_link_bandwidth() const {
  return link_bandwidth;
}

float NetworkedMachineModel::get_link_bandwidth(int src, int dst) const {
  return link_bandwidth * conn_matrix[src * total_devs + dst];
}

float NetworkedMachineModel::get_inter_node_gpu_bandwidth() const {
  return link_bandwidth;
}

void NetworkedMachineModel::set_routing_strategy(NetworkRoutingStrategy *rs) {
  delete routing_strategy;
  routing_strategy = rs;
}

std::vector<CommDevice *>
    NetworkedMachineModel::get_comm_path(MemDevice *src_mem,
                                         MemDevice *tar_mem) {
  /* This implementation very much is an extension of the simple_machine
   * model's version. no details about socket and memories
   */
  std::vector<CommDevice *> ret;
  // on the same memory
  if (src_mem->mem_type == tar_mem->mem_type and
      src_mem->device_id == tar_mem->device_id) {
    return ret;
  }
  if (src_mem->mem_type == MemDevice::SYSTEM_MEM and
      tar_mem->mem_type == MemDevice::SYSTEM_MEM) {
    if (src_mem->node_id == tar_mem->node_id) {
      return ret;
    } else {
      int device_id = src_mem->node_id * total_devs + tar_mem->node_id;
      if (pipelined) {
        ret.emplace_back(ids_to_nw_nominal_device.at(device_id));
      } else {
        std::vector<CommDevice *> physical_path =
            ids_to_nw_nominal_device.at(device_id)->expand_to_physical();
        ret.insert(ret.end(), physical_path.cbegin(), physical_path.cend());
      }
    }
  } else if (src_mem->mem_type == MemDevice::GPU_FB_MEM and
             tar_mem->mem_type == MemDevice::GPU_FB_MEM) {
    if (src_mem->node_id == tar_mem->node_id) {
      int device_id = src_mem->device_id * num_gpus + tar_mem->device_id;
      ret.emplace_back(ids_to_inter_gpu_comm_device.at(device_id));
    } else {
      if (pcie_on) {
        ret.emplace_back(id_to_gputodram_comm_device.at(src_mem->device_id));
      }
      int device_id = src_mem->node_id * total_devs + tar_mem->node_id;
      if (pipelined) {
        ret.emplace_back(ids_to_nw_nominal_device.at(device_id));
      } else {
        std::vector<CommDevice *> physical_path =
            ids_to_nw_nominal_device.at(device_id)->expand_to_physical();
        ret.insert(ret.end(), physical_path.cbegin(), physical_path.cend());
      }
      if (pcie_on) {
        ret.emplace_back(id_to_dramtogpu_comm_device.at(tar_mem->device_id));
      }
    }
  } else if (src_mem->mem_type == MemDevice::SYSTEM_MEM and
             tar_mem->mem_type == MemDevice::GPU_FB_MEM) {
    if (src_mem->node_id == tar_mem->node_id) {
      if (pcie_on) {
        ret.emplace_back(id_to_dramtogpu_comm_device.at(tar_mem->device_id));
      }
    } else {
      int device_id = src_mem->node_id * total_devs + tar_mem->node_id;
      if (pipelined) {
        ret.emplace_back(ids_to_nw_nominal_device.at(device_id));
      } else {
        std::vector<CommDevice *> physical_path =
            ids_to_nw_nominal_device.at(device_id)->expand_to_physical();
        ret.insert(ret.end(), physical_path.cbegin(), physical_path.cend());
      }
      if (pcie_on) {
        ret.emplace_back(id_to_dramtogpu_comm_device.at(tar_mem->device_id));
      }
    }
  } else if (src_mem->mem_type == MemDevice::GPU_FB_MEM and
             tar_mem->mem_type == MemDevice::SYSTEM_MEM) {
    if (src_mem->node_id == tar_mem->node_id) {
      if (pcie_on) {
        ret.emplace_back(id_to_gputodram_comm_device.at(src_mem->device_id));
      }
    } else {
      if (pcie_on) {
        ret.emplace_back(id_to_gputodram_comm_device.at(src_mem->device_id));
      }
      int device_id = src_mem->node_id * total_devs + tar_mem->node_id;
      if (pipelined) {
        ret.emplace_back(ids_to_nw_nominal_device.at(device_id));
      } else {
        std::vector<CommDevice *> physical_path =
            ids_to_nw_nominal_device.at(device_id)->expand_to_physical();
        ret.insert(ret.end(), physical_path.cbegin(), physical_path.cend());
      }
    }
  } else {
    printf("No path found between %s and %s\n",
           src_mem->name.c_str(),
           tar_mem->name.c_str());
    assert(false);
  }
  return ret;
}

std::string NetworkedMachineModel::to_string() const {
  // TODO
  return "";
}

CommDevice *NetworkedMachineModel::get_nominal_path(MemDevice *src_mem,
                                                    MemDevice *tar_mem) const {
  if (src_mem->node_id == tar_mem->node_id) {
    return nullptr;
  }
  int device_id = src_mem->node_id * num_nodes + tar_mem->node_id;
  return ids_to_nw_nominal_device.at(device_id);
}

// TODO
void NetworkedMachineModel::save_topology_json(std::string const &fname) const {
  return;
}

void NetworkedMachineModel::set_pcie(bool state) {
  pcie_on = state;
}

void NetworkedMachineModel::set_pipeline(bool state) {
  pipelined = state;
}

void NetworkedMachineModel::set_topology(ConnectionMatrix const &conn) {
  conn_matrix = conn;
  int total_devs = num_nodes + num_switches;
  for (int i = 0; i < total_devs; i++) {
    for (int j = 0; j < total_devs; j++) {
      // if (conn_matrix[i * total_devs + j] > 0) {
      int device_id = i * total_devs + j;
      ids_to_nw_comm_device[device_id]->bandwidth =
          conn[i * total_devs + j] * link_bandwidth;
    }
    // }
  }
  update_route();
}

ConnectionMatrix const &NetworkedMachineModel::get_conn_matrix() {
  return conn_matrix;
}

std::map<size_t, NominalCommDevice *> const &
    NetworkedMachineModel::get_nomm_comm_devs() {
  return ids_to_nw_nominal_device;
}

}; // namespace FlexFlow
