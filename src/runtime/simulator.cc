/* Copyright 2020 Stanford
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

#include "simulator.h"
#include "model.h"
#include "queue"

int ParallelConfig::num_parts() const
{
  int nparts = 1;
  for (int i = 0; i < nDims; i++)
    nparts *= dim[i];
  return nparts;
}

bool ParallelConfig::is_data_parallel() const
{
  int nparts = 1;
  for (int i = 0; i < nDims; i++) {
    nparts *= dim[i];
    if ((i < nDims-1) && (dim[i] > 1))
      return false;
  }
  for (int i = 0; i < nparts; i++)
    if (device_ids[i] != i)
      return false;
  return true;
}

// class Device
Device::Device(std::string name, DeviceType type, int node_id, int socket_id, int device_id)
: name(name), type(type), node_id(node_id), socket_id(socket_id), device_id(device_id)
{}

// class Comp_device
Comp_device::Comp_device(std::string name, CompDevType comp_type, int node_id, int socket_id, int device_id)
: Device(name, Device::DEVICE_COMP, node_id, socket_id, device_id), comp_type(comp_type)
{}

// class Mem_device
Mem_device::Mem_device(std::string name, MemDevType mem_type, int node_id, int socket_id, int device_id, size_t capacity)
: Device(name, Device::DEVICE_MEM, node_id, socket_id, device_id), mem_type(mem_type), capacity(capacity)
{}

// class Comm_device
Comm_device::Comm_device(std::string name, CommDevType comm_type, int node_id, int socket_id, int device_id, float latency, float bandwidth)
: Device(name, Device::DEVICE_COMM, node_id, socket_id, device_id), comm_type(comm_type), latency(latency), bandwidth(bandwidth)
{}

// class MachineModel_old
MachineModel_old::MachineModel_old(int num_nodes, int num_gpus_per_node, size_t capacity)
{
  version = 0;
  this->num_nodes = num_nodes;
  this->num_gpus_per_node = num_gpus_per_node;
  printf("num_nodes = %d num_gpus_per_node = %d\n", num_nodes, num_gpus_per_node);
  num_gpus = num_nodes * num_gpus_per_node;
  inter_gpu_bandwidth = 20 * 1024 * 1024.0f; /* B/ms*/
  inter_node_bandwidth = 12 * 1024 * 1024.0f / num_nodes; /* B/ms*/
  gpu_dram_bandwidth = 16 * 1024 * 1024.0f; /* B/ms*/

  // Create GPU compute device
  for (int i = 0; i < num_nodes; i++) {
    for (int j = 0; j < num_gpus_per_node; j++) {
      int device_id = i * num_gpus_per_node + j;
      std::string gpu_name = "GPU " + std::to_string(device_id);
      id_to_gpu[device_id] = new Comp_device(gpu_name, Comp_device::TOC_PROC, i, i, device_id);
      std::string gpu_mem_name = "GPU_FB_MEM " + std::to_string(device_id);
      id_to_gpu_fb_mem[device_id] = new Mem_device(gpu_mem_name, Mem_device::GPU_FB_MEM, i, i, device_id, capacity);
    }
  }

  // Create inter GPU comm devices (NVLinks)
  for (int i = 0; i < num_gpus; i++) {
    for (int j = 0; j < num_gpus; j++) {
      Device* src = id_to_gpu[i];
      Device* dst = id_to_gpu[j];
      if (src->node_id == dst->node_id && src != dst) {
        int device_id = i * num_gpus + j;
        std::string nvlink_name = "NVLINK " + std::to_string(device_id);
        ids_to_inter_gpu_comm_device[device_id] = new Comm_device(nvlink_name, Comm_device::NVLINK_COMM, src->node_id, src->node_id, device_id, 0, inter_gpu_bandwidth);
      }
    }
  }

  // Create gpu<->dram comm devices
  for (int i = 0; i < num_gpus; i++) {
    int node_id = num_gpus / num_gpus_per_node;
    std::string pci_to_host_name = "PCI_TO_HOST " + std::to_string(i);
    id_to_gputodram_comm_device[i] = new Comm_device(pci_to_host_name, Comm_device::PCI_TO_HOST_COMM, node_id, node_id, i, 0, gpu_dram_bandwidth);
    std::string pci_to_dev_name = "PCI_TO_DEV " + std::to_string(i);
    id_to_dramtogpu_comm_device[i] = new Comm_device(pci_to_dev_name, Comm_device::PCI_TO_DEV_COMM, node_id, node_id, i, 0, gpu_dram_bandwidth);
  }

  // Create inter node comm devices
  for (int i = 0; i < num_nodes; i++) {
    for (int j = 0; j < num_nodes; j++) {
      if (i != j) {
        int device_id = i * num_nodes + j;
        std::string nic_name = "NIC " + std::to_string(device_id);
        ids_to_inter_node_comm_device[device_id] = new Comm_device(nic_name, Comm_device::NIC_OUT_COMM, -1, -1, device_id, 0, inter_node_bandwidth);
      }
    }
  }
}

int MachineModel_old::get_version()
{
  return version;
}

Comp_device *MachineModel_old::get_gpu(int device_id) 
{
  assert(id_to_gpu.find(device_id) != id_to_gpu.end());
  return id_to_gpu[device_id];
}

Mem_device *MachineModel_old::get_gpu_fb_mem(int device_id) 
{
  assert(id_to_gpu_fb_mem.find(device_id) != id_to_gpu_fb_mem.end());
  return id_to_gpu_fb_mem[device_id];
}

int MachineModel_old::get_num_gpus()
{
  return num_gpus;
}

float MachineModel_old::get_intra_node_gpu_bandwidth()
{
  return inter_gpu_bandwidth;
}

float MachineModel_old::get_inter_node_gpu_bandwidth()
{
  return inter_node_bandwidth;
}


std::vector<Comm_device *> MachineModel_old::get_comm_path(Mem_device *src_mem, Mem_device *tar_mem)
{
    std::vector<Comm_device *> ret;
    // on the same memory
    if (src_mem->mem_type == tar_mem->mem_type and src_mem->device_id == tar_mem->device_id) {
        return ret;
    }
    if (src_mem->mem_type == Mem_device::SYSTEM_MEM and tar_mem->mem_type == Mem_device::SYSTEM_MEM) {
        if (src_mem->node_id == tar_mem->node_id) {
            return ret;
        }
        else {
            int device_id = src_mem->node_id * num_nodes + tar_mem->node_id;
            ret.emplace_back(ids_to_inter_node_comm_device[device_id]);
        }
    }
    else if (src_mem->mem_type == Mem_device::GPU_FB_MEM and tar_mem->mem_type == Mem_device::GPU_FB_MEM) {
        if (src_mem->node_id == tar_mem->node_id) {
            int device_id = src_mem->device_id * num_gpus + tar_mem->device_id;
            ret.emplace_back(ids_to_inter_gpu_comm_device[device_id]);
        }
        else {
            ret.emplace_back(id_to_gputodram_comm_device[src_mem->device_id]);
            int device_id = src_mem->node_id * num_nodes + tar_mem->node_id;
            ret.emplace_back(ids_to_inter_node_comm_device[device_id]);
            ret.emplace_back(id_to_dramtogpu_comm_device[tar_mem->device_id]);
        }
    }
    else if (src_mem->mem_type == Mem_device::SYSTEM_MEM and tar_mem->mem_type == Mem_device::GPU_FB_MEM) {
        if (src_mem->node_id == tar_mem->node_id) {
            ret.emplace_back(id_to_dramtogpu_comm_device[tar_mem->device_id]);
        }
        else {
            int device_id = src_mem->node_id * num_nodes + tar_mem->node_id;
            ret.emplace_back(ids_to_inter_node_comm_device[device_id]);
            ret.emplace_back(id_to_dramtogpu_comm_device[tar_mem->device_id]);
        }
    }
    else if (src_mem->mem_type == Mem_device::GPU_FB_MEM and tar_mem->mem_type == Mem_device::SYSTEM_MEM) {
        if (src_mem->node_id == tar_mem->node_id) {
            ret.emplace_back(id_to_gputodram_comm_device[src_mem->device_id]);
        }
        else {
            ret.emplace_back(id_to_gputodram_comm_device[src_mem->device_id]);
            int device_id = src_mem->node_id * num_nodes + tar_mem->node_id;
            ret.emplace_back(ids_to_inter_node_comm_device[device_id]);
        }
    }
    else {
        printf("No path found between %s and %s\n", src_mem->name.c_str(), tar_mem->name.c_str());
    }

    return ret;
}

std::string MachineModel_old::to_string()
{
  std::string s;
  for (int i = 0; i < num_nodes; i++) {
    int node_id = i;
    s += "==========================================\n";
    s += "Node " + std::to_string(node_id) + '\n';
    s += "COMP: \n";
    for (int j = 0; j < num_gpus_per_node; j++) {
      int device_id = i * num_gpus_per_node + j;
      s += id_to_gpu[device_id]->name + '\n';
    }
    s += '\n';
    s += "MEM: \n";
    for (int j = 0; j < num_gpus_per_node; j++) {
      int device_id = i * num_gpus_per_node + j;
      s += id_to_gpu_fb_mem[device_id]->name + '\n';
    }
  }
  return s;
}

MachineModel_new::MachineModel_new(std::string file, size_t gpu_fb_mem_capacity)
{
  version = 1;
  this->gpu_fb_mem_capacity = gpu_fb_mem_capacity;
  std::ifstream machine_config(file);
  std::string line;
  while (std::getline(machine_config, line))
  {
    if (line[0] != '#') {
      // split a line into words
      std::istringstream iss(line);
      std::vector<std::string> words{std::istream_iterator<std::string>{iss}, std::istream_iterator<std::string>{}};
      if (words.size() >= 3) {
        if (words[0] == "num_nodes") {
          num_nodes = stoi(words[2]);
          printf("num_nodes = %d\n", num_nodes);
        }
        else if (words[0] == "num_sockets_per_node") {
          num_sockets_per_node = stoi(words[2]);
          printf("num_sockets_per_node = %d\n", num_sockets_per_node);
        }
        else if (words[0] == "num_cpus_per_socket") {
          num_cpus_per_socket = stoi(words[2]);
          printf("num_cpus_per_socket = %d\n", num_cpus_per_socket);
        }
        else if (words[0] == "num_gpus_per_socket") {
          num_gpus_per_socket = stoi(words[2]);
          printf("num_gpus_per_socket = %d\n", num_gpus_per_socket);
        }
        else if (words[0] == "membus_latency") {
          membus_latency = stof(words[2]);
          printf("membus_latency = %f\n", membus_latency);
        }
        else if (words[0] == "membus_bandwidth") {
          membus_bandwidth = stof(words[2]);
          printf("membus_bandwidth = %f\n", membus_bandwidth);
        }
        else if (words[0] == "upi_latency") {
          upi_latency = stof(words[2]);
          printf("upi_latency = %f\n", upi_latency);
        }
        else if (words[0] == "upi_bandwidth") {
          upi_bandwidth = stof(words[2]);
          printf("upi_bandwidth = %f\n", upi_bandwidth);
        }
        else if (words[0] == "nic_latency") {
          nic_latency = stof(words[2]);
          printf("nic_latency = %f\n", nic_latency);
        }
        else if (words[0] == "nic_bandwidth") {
          nic_bandwidth = stof(words[2]);
          printf("nic_bandwidth = %f\n", nic_bandwidth);
        }
        else if (words[0] == "nic_distribution") {
          nic_distribution = static_cast<NicDistribution>(stoi(words[2]));
          printf("nic_distribution = %d\n", nic_distribution);
        }
        else if (words[0] == "pci_latency") {
          pci_latency = stof(words[2]);
          printf("pci_latency = %f\n", pci_latency);
        }
        else if (words[0] == "pci_bandwidth") {
          pci_bandwidth = stof(words[2]);
          printf("pci_bandwidth = %f\n", pci_bandwidth);
        }
        else if (words[0] == "nvlink_latency") {
          nvlink_latency = stof(words[2]);
          printf("nvlink_latency = %f\n", nvlink_latency);
        }
        else if (words[0] == "nvlink_bandwidth") {
          nvlink_bandwidth = stof(words[2]);
          printf("nvlink_bandwidth = %f\n", nvlink_bandwidth);
        }
        else if (words[0] == "intra_socket_sys_mem_to_sys_mem") {
          printf("intra_socket_sys_mem_to_sys_mem = ");
          for (size_t i = 2; i < words.size(); i++) {
            set_comm_path(intra_socket_sys_mem_to_sys_mem, words[i]);
            printf("%s ", words[i].c_str());
          }
          printf("\n");
        }
        else if (words[0] == "inter_socket_sys_mem_to_sys_mem") {
          printf("inter_socket_sys_mem_to_sys_mem = ");
          for (size_t i = 2; i < words.size(); i++) {
            set_comm_path(inter_socket_sys_mem_to_sys_mem, words[i]);
            printf("%s ", words[i].c_str());
          }
          printf("\n");
        }
        else if (words[0] == "inter_node_sys_mem_to_sys_mem") {
          printf("inter_node_sys_mem_to_sys_mem = ");
          for (size_t i = 2; i < words.size(); i++) {
            set_comm_path(inter_node_sys_mem_to_sys_mem, words[i]);
            printf("%s ", words[i].c_str());
          }
          printf("\n");
        }
        else if (words[0] == "intra_socket_gpu_fb_mem_to_gpu_fb_mem") {
          printf("intra_socket_gpu_fb_mem_to_gpu_fb_mem = ");
          for (size_t i = 2; i < words.size(); i++) {
            set_comm_path(intra_socket_gpu_fb_mem_to_gpu_fb_mem, words[i]);
            printf("%s ", words[i].c_str());
          }
          printf("\n");
        }
        else if (words[0] == "inter_socket_gpu_fb_mem_to_gpu_fb_mem") {
          printf("inter_socket_gpu_fb_mem_to_gpu_fb_mem = ");
          for (size_t i = 2; i < words.size(); i++) {
            set_comm_path(inter_socket_gpu_fb_mem_to_gpu_fb_mem, words[i]);
            printf("%s ", words[i].c_str());
          }
          printf("\n");
        }
        else if (words[0] == "inter_node_gpu_fb_mem_to_gpu_fb_mem") {
          printf("inter_node_gpu_fb_mem_to_gpu_fb_mem = ");
          for (size_t i = 2; i < words.size(); i++) {
            set_comm_path(inter_node_gpu_fb_mem_to_gpu_fb_mem, words[i]);
            printf("%s ", words[i].c_str());
          }
          printf("\n");
        }
        else if (words[0] == "intra_socket_sys_mem_to_gpu_fb_mem") {
          printf("intra_socket_sys_mem_to_gpu_fb_mem = ");
          for (size_t i = 2; i < words.size(); i++) {
            set_comm_path(intra_socket_sys_mem_to_gpu_fb_mem, words[i]);
            printf("%s ", words[i].c_str());
          }
          printf("\n");
        }
        else if (words[0] == "inter_socket_sys_mem_to_gpu_fb_mem") {
          printf("inter_socket_sys_mem_to_gpu_fb_mem = ");
          for (size_t i = 2; i < words.size(); i++) {
            set_comm_path(inter_socket_sys_mem_to_gpu_fb_mem, words[i]);
            printf("%s ", words[i].c_str());
          }
          printf("\n");
        }
        else if (words[0] == "inter_node_sys_mem_to_gpu_fb_mem") {
          printf("inter_node_sys_mem_to_gpu_fb_mem = ");
          for (size_t i = 2; i < words.size(); i++) {
            set_comm_path(inter_node_sys_mem_to_gpu_fb_mem, words[i]);
            printf("%s ", words[i].c_str());
          }
          printf("\n");
        }
        else if (words[0] == "intra_socket_gpu_fb_mem_to_sys_mem") {
          printf("intra_socket_gpu_fb_mem_to_sys_mem = ");
          for (size_t i = 2; i < words.size(); i++) {
            set_comm_path(intra_socket_gpu_fb_mem_to_sys_mem, words[i]);
            printf("%s ", words[i].c_str());
          }
          printf("\n");
        }
        else if (words[0] == "inter_socket_gpu_fb_mem_to_sys_mem") {
          printf("inter_socket_gpu_fb_mem_to_sys_mem = ");
          for (size_t i = 2; i < words.size(); i++) {
            set_comm_path(inter_socket_gpu_fb_mem_to_sys_mem, words[i]);
            printf("%s ", words[i].c_str());
          }
          printf("\n");
        }
        else if (words[0] == "inter_node_gpu_fb_mem_to_sys_mem") {
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
  num_nvlinks_per_node = 0;
  mem_to_nvlink.clear();
  add_cpus();
  add_gpus();
  add_membuses(membus_latency, membus_bandwidth * 1024 * 1024);
  add_upis(upi_latency / 2, upi_bandwidth * 2 * 1024 * 1024);
  add_nics(nic_latency / 2, nic_bandwidth * 2 * 1024 * 1024, nic_distribution);
  add_pcis(pci_latency, pci_bandwidth * 1024 * 1024);
  add_nvlinks(nvlink_latency, nvlink_bandwidth * 1024 * 1024);
}

int MachineModel_new::get_version()
{
  return version;
}

void MachineModel_new::set_comm_path(std::vector<Comm_device::CommDevType> &comm_path, std::string device_str)
{
  if (device_str == "membus") {
    comm_path.emplace_back(Comm_device::MEMBUS_COMM);
  }
  else if (device_str == "upi") {
    comm_path.emplace_back(Comm_device::UPI_OUT_COMM);
    comm_path.emplace_back(Comm_device::UPI_IN_COMM);
  }
  else if (device_str == "nic") {
    comm_path.emplace_back(Comm_device::NIC_OUT_COMM);
    comm_path.emplace_back(Comm_device::NIC_IN_COMM);
  }
  else if (device_str == "pci_to_host") {
    comm_path.emplace_back(Comm_device::PCI_TO_HOST_COMM);
  }
  else if (device_str == "pci_to_dev") {
    comm_path.emplace_back(Comm_device::PCI_TO_DEV_COMM);
  }
  else if (device_str == "nvlink") {
    comm_path.emplace_back(Comm_device::NVLINK_COMM);
  }
}

void MachineModel_new::add_cpus()
{
  for (int i = 0; i < num_nodes; i++) {
    int node_id = i;
    for (int j = 0; j < num_sockets_per_node; j++) {
      int socket_id = i * num_sockets_per_node + j;
      int device_id = socket_id;
      // add system memory
      std::string sys_mem_name = "SYSTEM_MEM " + std::to_string(device_id);
      Mem_device *sys_mem = new Mem_device(sys_mem_name, Mem_device::SYSTEM_MEM, node_id, socket_id, device_id, -1);
      sys_mems.emplace_back(sys_mem);
      // add cpus
      cpus.push_back({});
      for (int k = 0; k < num_cpus_per_socket; k++) {
        device_id = socket_id * num_cpus_per_socket + k;
        std::string cpu_name = "CPU " + std::to_string(device_id);
        cpus[socket_id].emplace_back(new Comp_device(cpu_name, Comp_device::LOC_PROC, node_id, socket_id, device_id));
      }
    }
  }
}

void MachineModel_new::add_gpus()
{
  for (int i = 0; i < num_nodes; i++) {
    int node_id = i;
    for (int j = 0; j < num_sockets_per_node; j++) {
      int socket_id = i * num_sockets_per_node + j;
      int device_id = socket_id;
      // add zero copy memory
      std::string z_copy_mem_name = "Z_COPY_MEM " + std::to_string(device_id);
      Mem_device *z_copy_mem = new Mem_device(z_copy_mem_name, Mem_device::Z_COPY_MEM, node_id, socket_id, device_id, -1);
      z_copy_mems.push_back(z_copy_mem);
      // add gpus and gpu framebuffer memories
      gpus.push_back({});
      gpu_fb_mems.push_back({});
      for (int k = 0; k < num_gpus_per_socket; k++) {
          device_id = socket_id * num_gpus_per_socket + k;
          std::string gpu_name = "GPU " + std::to_string(device_id);
          gpus[socket_id].push_back(new Comp_device(gpu_name, Comp_device::TOC_PROC, node_id, socket_id, device_id));
          std::string gpu_mem_name = "GPU_FB_MEM " + std::to_string(device_id);
          Mem_device *gpu_mem = new Mem_device(gpu_mem_name, Mem_device::GPU_FB_MEM, node_id, socket_id, device_id, gpu_fb_mem_capacity);
          gpu_fb_mems[socket_id].push_back({gpu_mem});
      }
    }
  }
}

void MachineModel_new::add_membuses(float latency, float bandwidth)
{
  for (int i = 0; i < num_nodes; i++) {
    int node_id = i;
    for (int j = 0; j < num_sockets_per_node; j++) {
      int socket_id = i * num_sockets_per_node + j;
      int device_id = socket_id;
      std::string membus_name = "MEMBUS " + std::to_string(device_id);
      Comm_device *membus = new Comm_device(membus_name, Comm_device::MEMBUS_COMM, node_id, socket_id, device_id, latency, bandwidth);
      membuses.push_back(membus);
    }
  }
}

void MachineModel_new::add_upis(float latency, float bandwidth)
{
  for (int i = 0; i < num_nodes; i++) {
    int node_id = i;
    for (int j = 0; j < num_sockets_per_node; j++) {
      int socket_id = i * num_sockets_per_node + j;
      int device_id = socket_id;
      std::string upi_in_name = "UPI_IN " + std::to_string(device_id);
      Comm_device *upi_in = new Comm_device(upi_in_name, Comm_device::UPI_IN_COMM, node_id, socket_id, device_id, latency, bandwidth);
      upi_ins.push_back(upi_in);
      std::string upi_out_name = "UPI_OUT " + std::to_string(device_id);
      Comm_device *upi_out = new Comm_device(upi_out_name, Comm_device::UPI_OUT_COMM, node_id, socket_id, device_id, latency, bandwidth);
      upi_outs.push_back(upi_out);
    }
  }
}

void MachineModel_new::add_nics(float latency, float bandwidth, NicDistribution nic_distribution)
{
  if (nic_distribution == PER_NODE) {
    for (int i = 0; i < num_nodes; i++) {
      int node_id = i;
      for (int j = 0; j < num_sockets_per_node; j++) {
        int socket_id = i * num_sockets_per_node + j;
        int device_id = socket_id;
        Comm_device *nic_in;
        Comm_device *nic_out;
        if (j == 0) {
          std::string nic_in_name = "NIC_IN " + std::to_string(device_id);
          nic_in = new Comm_device(nic_in_name, Comm_device::NIC_IN_COMM, node_id, socket_id, device_id, latency, bandwidth);
          nic_ins.push_back(nic_in);
          std::string nic_out_name = "NIC_OUT " + std::to_string(device_id);
          nic_out = new Comm_device(nic_out_name, Comm_device::NIC_OUT_COMM, node_id, socket_id, device_id, latency, bandwidth);
          nic_outs.push_back(nic_out);
        }
        else {
          nic_ins.push_back(nic_in);
          nic_outs.push_back(nic_out);
        }
      }
    }
  }
  else if (nic_distribution == PER_SOCKET) {
    for (int i = 0; i < num_nodes; i++) {
      int node_id = i;
      for (int j = 0; j < num_sockets_per_node; j++) {
        int socket_id = i * num_sockets_per_node + j;
        int device_id = socket_id;
        std::string nic_in_name = "NIC_IN " + std::to_string(device_id);
        Comm_device *nic_in = new Comm_device(nic_in_name, Comm_device::NIC_IN_COMM, node_id, socket_id, device_id, latency, bandwidth);
        nic_ins.push_back(nic_in);
        std::string nic_out_name = "NIC_OUT " + std::to_string(device_id);
        Comm_device *nic_out = new Comm_device(nic_out_name, Comm_device::NIC_OUT_COMM, node_id, socket_id, device_id, latency, bandwidth);
        nic_outs.push_back(nic_out);
      }
    }
  }
  else {
    assert(false && "Unknown nic distribution type");
  }
}

void MachineModel_new::add_pcis(float latency, float bandwidth)
{
  for (int i = 0; i < num_nodes; i++) {
    int node_id = i;
    for (int j = 0; j < num_sockets_per_node; j++) {
      int socket_id = i * num_sockets_per_node + j;
      int device_id = socket_id;
      std::string pci_to_host_name = "PCI_TO_HOST " + std::to_string(device_id);    // pcie to memory
      Comm_device *pci_to_host = new Comm_device(pci_to_host_name, Comm_device::PCI_TO_HOST_COMM, node_id, socket_id, socket_id, latency, bandwidth);
      pcis_to_host.push_back(pci_to_host);
      std::string pci_to_dev_name = "PCI_TO_DEV " + std::to_string(device_id);  // memory to pcie
      Comm_device *pci_to_dev = new Comm_device(pci_to_dev_name, Comm_device::PCI_TO_DEV_COMM, node_id, socket_id, socket_id, latency, bandwidth);
      pcis_to_device.push_back(pci_to_dev);
    }
  }    
}

// assume each GPU has nvlinks to the other GPUs on the same node and the nvlinks have the same latency and bandwidth
void MachineModel_new::add_nvlinks(float latency, float bandwidth)
{
  int num_gpus_per_node = num_gpus_per_socket * num_sockets_per_node;
  num_nvlinks_per_node = num_gpus_per_node * (num_gpus_per_node - 1) / 2;
  for (int i = 0; i < num_nodes; i++) {
    int node_id = i;
    int socket_id = i * num_sockets_per_node;
    nvlinks.push_back({});
    for (int j = 0; j < num_nvlinks_per_node * 2; j++) {
      int nvlink_id = node_id * num_nvlinks_per_node * 2 + j;
      std::string nvlink_name = "NVLINK " + std::to_string(nvlink_id);
      nvlinks[i].push_back(new Comm_device(nvlink_name, Comm_device::NVLINK_COMM, node_id, socket_id, nvlink_id, latency, bandwidth));
    }

    for (int j = 0; j < num_sockets_per_node; j++) {
      int src_socket_id = i * num_sockets_per_node + j;
      for (int k = 0; k < num_gpus_per_socket; k++) {
        Mem_device *src_gpu_fb_mem = gpu_fb_mems[src_socket_id][k];
        int src_local_id = j * num_gpus_per_socket + k;
        for (int l = 0; l < num_sockets_per_node; l++) {
          int tar_socket_id = i * num_sockets_per_node + l;
          for (int m = 0; m < num_gpus_per_socket; m++) {
            Mem_device *tar_gpu_fb_mem = gpu_fb_mems[tar_socket_id][m];
            int tar_local_id = l * num_gpus_per_socket + m;
            if (src_local_id != tar_local_id) {
              int local_nvlink_id = src_local_id * (num_gpus_per_node - 1) + tar_local_id;
              if (tar_local_id > src_local_id) {
                local_nvlink_id--;
              }
              attach_nvlink(src_gpu_fb_mem, tar_gpu_fb_mem, nvlinks[i][local_nvlink_id]);
              printf("add nvlink: gdb_fb_mem %d , gou_fb_mem %d, nvlink %d %d\n", src_gpu_fb_mem->device_id, tar_gpu_fb_mem->device_id, node_id, local_nvlink_id);
            }
          }
        }
      }
    }
  }
}

void MachineModel_new::attach_nvlink(Mem_device *src_mem, Mem_device *tar_mem, Comm_device *comm) 
{
  assert(comm->comm_type == Comm_device::NVLINK_COMM);
  int hash = src_mem->device_id * num_gpus + tar_mem->device_id;
  if (mem_to_nvlink.find(hash) == mem_to_nvlink.end()) {
    mem_to_nvlink[hash] = comm;
  }
}

Comp_device *MachineModel_new::get_cpu(int device_id)
{
  return get_cpu(device_id / num_cpus_per_socket, device_id % num_cpus_per_socket);
}

Comp_device *MachineModel_new::get_cpu(int socket_id, int local_id)
{
  if (socket_id < num_sockets and local_id < num_cpus_per_socket) {
    return cpus[socket_id][local_id];
  }
  else {
    printf("MachineModel: get_cpu - cannot find cpu (%d %d)\n", socket_id, local_id);
    assert(false);
  }
} 

Comp_device *MachineModel_new::get_gpu(int device_id)
{
  return get_gpu(device_id / num_gpus_per_socket, device_id % num_gpus_per_socket);
}

Comp_device *MachineModel_new::get_gpu(int socket_id, int local_id)
{
  if (socket_id < num_sockets and local_id < num_cpus_per_socket) {
    return gpus[socket_id][local_id];
  }
  else {
    printf("MachineModel: get_gpu - cannot find gpu (%d %d)\n", socket_id, local_id);
    assert(false);
  }
}

Mem_device *MachineModel_new::get_sys_mem(int socket_id)
{
  return sys_mems[socket_id];
}

Mem_device *MachineModel_new::get_z_copy_mem(int socket_id)
{
  return z_copy_mems[socket_id];
}

Mem_device *MachineModel_new::get_gpu_fb_mem(int device_id)
{
  return get_gpu_fb_mem(device_id / num_gpus_per_socket, device_id % num_gpus_per_socket);
}

Mem_device *MachineModel_new::get_gpu_fb_mem(int socket_id, int local_id)
{
  if (socket_id < num_sockets and local_id < num_cpus_per_socket) {
    return gpu_fb_mems[socket_id][local_id];
  }
  else {
    printf("MachineModel: get_gpu_fb_mem - cannot find gpu_fb_mem (%d %d)\n", socket_id, local_id);
    assert(false);
  }
}

Comm_device *MachineModel_new::get_nvlink(Mem_device *src_mem, Mem_device *tar_mem)
{
  int hash = src_mem->device_id * num_gpus + tar_mem->device_id;
  if (mem_to_nvlink.find(hash) != mem_to_nvlink.end()) {
    return mem_to_nvlink[hash];
  }
  else {
    printf("MachineModel: get_nvlink - cannot get nvlink between %s and %s\n", src_mem->name.c_str(), tar_mem->name.c_str());
    assert(false);
  }
}

int MachineModel_new::get_num_gpus()
{
  return num_gpus;
}

void MachineModel_new::add_comm_path(std::vector<Comm_device::CommDevType> comm_device_list, Mem_device *src_mem, 
                   Mem_device *tar_mem, std::vector<Comm_device *> &ret) 
{
  Mem_device *cur_mem = src_mem;
  for (size_t i = 0; i < comm_device_list.size(); i++) {
    switch (comm_device_list[i])
    {
    case Comm_device::MEMBUS_COMM:
      ret.emplace_back(membuses[cur_mem->socket_id]);
      break;
    case Comm_device::UPI_IN_COMM:
      cur_mem = tar_mem;
      ret.emplace_back(upi_ins[cur_mem->socket_id]);
      break;
    case Comm_device::UPI_OUT_COMM:
      ret.emplace_back(upi_outs[cur_mem->socket_id]);
      break;
    case Comm_device::NIC_IN_COMM:
      cur_mem = tar_mem;
      ret.emplace_back(nic_ins[cur_mem->socket_id]);
      break;
    case Comm_device::NIC_OUT_COMM:
      ret.emplace_back(nic_outs[cur_mem->socket_id]);
      break;
    case Comm_device::PCI_TO_HOST_COMM:
      ret.emplace_back(pcis_to_host[cur_mem->socket_id]);
      break;
    case Comm_device::PCI_TO_DEV_COMM:
      ret.emplace_back(pcis_to_device[cur_mem->socket_id]);
      break;
    case Comm_device::NVLINK_COMM:
      ret.emplace_back(get_nvlink(src_mem, tar_mem));
      break;
    default:
      break;
    }
  }
}

std::vector<Comm_device *> MachineModel_new::get_comm_path(Mem_device *src_mem, Mem_device *tar_mem)
{
  std::vector<Comm_device *> ret;
  if (src_mem->device_id == tar_mem->device_id) {
      return ret;
  }
  if (src_mem->mem_type == Mem_device::SYSTEM_MEM and tar_mem->mem_type == Mem_device::SYSTEM_MEM) {
    if (src_mem->socket_id == tar_mem->socket_id) {
      add_comm_path(intra_socket_sys_mem_to_sys_mem, src_mem, tar_mem, ret);
    }
    else if (src_mem->node_id == tar_mem->node_id) {
      add_comm_path(inter_socket_sys_mem_to_sys_mem, src_mem, tar_mem, ret);
    }
    else {
      add_comm_path(inter_node_sys_mem_to_sys_mem, src_mem, tar_mem, ret);
    }
  }
  else if (src_mem->mem_type == Mem_device::SYSTEM_MEM and tar_mem->mem_type == Mem_device::GPU_FB_MEM) {
    if (src_mem->socket_id == tar_mem->socket_id) {
      add_comm_path(intra_socket_sys_mem_to_gpu_fb_mem, src_mem, tar_mem, ret);
    }
    else if (src_mem->node_id == tar_mem->node_id) {
      add_comm_path(inter_socket_sys_mem_to_gpu_fb_mem, src_mem, tar_mem, ret);
    }
    else {
      add_comm_path(inter_node_sys_mem_to_gpu_fb_mem, src_mem, tar_mem, ret);
    }
  }
  else if (src_mem->mem_type == Mem_device::GPU_FB_MEM and tar_mem->mem_type == Mem_device::SYSTEM_MEM) {
    if (src_mem->socket_id == tar_mem->socket_id) {
      add_comm_path(intra_socket_gpu_fb_mem_to_sys_mem, src_mem, tar_mem, ret);
    }
    else if (src_mem->node_id == tar_mem->node_id) {
      add_comm_path(inter_socket_gpu_fb_mem_to_sys_mem, src_mem, tar_mem, ret);
    }
    else {
      add_comm_path(inter_node_gpu_fb_mem_to_sys_mem, src_mem, tar_mem, ret);
    }
  }
  else if (src_mem->mem_type == Mem_device::GPU_FB_MEM and tar_mem->mem_type == Mem_device::GPU_FB_MEM) {
    if (src_mem->socket_id == tar_mem->socket_id) {
      add_comm_path(intra_socket_gpu_fb_mem_to_gpu_fb_mem, src_mem, tar_mem, ret);
    }
    else if (src_mem->node_id == tar_mem->node_id) {
      add_comm_path(inter_socket_gpu_fb_mem_to_gpu_fb_mem, src_mem, tar_mem, ret);
    }
    else {
      add_comm_path(inter_node_gpu_fb_mem_to_gpu_fb_mem, src_mem, tar_mem, ret);
    }
  }
  else {
    printf("MachineModel: get_comm_path - no path found between %s and %s\n", src_mem->name.c_str(), tar_mem->name.c_str());
    assert(false);
  }
  return ret;
}

float MachineModel_new::get_intra_node_gpu_bandwidth()
{
  return nvlink_bandwidth;
}

// Use inter-node cpu bandwidth for now 
float MachineModel_new::get_inter_node_gpu_bandwidth()
{
  return nic_bandwidth;
    // SimTask *src_task = new SimTask();
    // src_task->ready_time = 0.0f;
    // src_task->run_time = 0.0f;
    // src_task->next_tasks.clear();
    // src_task->counter = 0;
    // src_task->device = get_gpu(0, 0);   // the first gpu on the first socket of the first node
    // src_task->mem = get_gpu_fb_mem(0, 0);
    // src_task->name = "test_inter_node_gpu_bw_src";

    // SimTask *dst_task = new SimTask();
    // dst_task->ready_time = 0.0f;
    // dst_task->run_time = 0.0f;
    // dst_task->next_tasks.clear();
    // dst_task->counter = 0;
    // dst_task->device = get_gpu(num_sockets_per_node, 0);  // the first gpu on the first socket of the second node
    // dst_task->mem = get_gpu_fb_mem(num_socket_per_node, 0);
    // dst_task->name = "test_inter_node_gpu_bw_dst";

    // add_task_dependencies_with_xfer(src_task, dst_task, 64 << 20);
}

std::string MachineModel_new::to_string()
{
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
      s += nic_ins[socket_id]->name + '\n';
      s += nic_outs[socket_id]->name + '\n';
      s += pcis_to_host[socket_id]->name + '\n';
      s += pcis_to_device[socket_id]->name + '\n';
    }
    s += "------------------------------------------\n";
    for (int j = 0; j < num_nvlinks_per_node * 2; j++) {
      s += nvlinks[node_id][j]->name + '\n';
    }
  }
  return s;
}

SimTask::SimTask()
{}

void SimTask::add_next_task(SimTask* task)
{
  next_tasks.push_back(task);
  task->counter ++;
}

std::string SimTask::get_type_str() const {
  switch (type) {
    case TASK_FORWARD:
      return "Forward";
    case TASK_BACKWARD:
      return "Backward";
    case TASK_COMM:
      return "Comm";
    case TASK_UPDATE:
      return "Update";
    case TASK_BARRIER:
      return "Barrier";
    default:
      assert(false && "Unknown task type");
  }
}

TaskManager::TaskManager(size_t _max_num_tasks)
: max_num_tasks(_max_num_tasks)
{
  tasks = (SimTask**) malloc(sizeof(SimTask*) * max_num_tasks);
  for (size_t i = 0; i < max_num_tasks; i++) {
    tasks[i] = new SimTask();
  }
}

void TaskManager::reset()
{
  global_task_id = 0;
  hash_to_forward_task.clear();
  hash_to_backward_task.clear();
}

SimTask* TaskManager::new_task()
{
  assert(global_task_id + 1 < max_num_tasks);
  SimTask* task = tasks[global_task_id++];
  task->ready_time = 0.0f;
  task->run_time = 0.0f;
  task->next_tasks.clear();
  task->counter = 0;
  task->device = NULL;
  task->mem = NULL;
  (task->name).clear();
  return task;
}

SimTask* TaskManager::new_update_task()
{
  SimTask* task = new_task();
  task->type = SimTask::TASK_UPDATE;
  return task;
}

SimTask* TaskManager::new_barrier_task()
{
  SimTask* task = new_task();
  task->type = SimTask::TASK_BARRIER;
  return task;
}

SimTask* TaskManager::new_comm_task()
{
  SimTask* task = new_task();
  task->type = SimTask::TASK_COMM;
  return task;
}

SimTask* TaskManager::new_comm_task(std::string name, Comm_device *comm_device, size_t message_size)
{
  SimTask* task = new_task();
  task->type = SimTask::TASK_COMM;
  task->name = name;
  task->device = comm_device;
  task->run_time = comm_device->latency + message_size / comm_device->bandwidth;
  return task;
}

SimTask* TaskManager::new_forward_task(Op* op, int idx)
{
  SimTask* task = new_task();
  task->type = SimTask::TASK_FORWARD;
  size_t hash = 17 * 31 + (size_t)(op);
  hash = hash * 31 + std::hash<int>()(idx);
  hash_to_forward_task[hash] = task;
  task->name = op->name;
  return task;
}

SimTask* TaskManager::new_backward_task(Op* op, int idx)
{
  SimTask* task = new_task();
  task->type = SimTask::TASK_BACKWARD;
  size_t hash = 17 * 31 + (size_t)(op);
  hash = hash * 31 + std::hash<int>()(idx);
  hash_to_backward_task[hash] = task;
  task->name = op->name;
  return task;
}

SimTask* TaskManager::get_forward_task(Op* op, int idx)
{
  size_t hash = 17 * 31 + (size_t)(op);
  hash = hash * 31 + std::hash<int>()(idx);
  assert(hash_to_forward_task.find(hash) != hash_to_forward_task.end());
  return hash_to_forward_task[hash];
}

SimTask* TaskManager::get_backward_task(Op* op, int idx)
{
  size_t hash = 17 * 31 + (size_t)(op);
  hash = hash * 31 + std::hash<int>()(idx);
  assert(hash_to_backward_task.find(hash) != hash_to_backward_task.end());
  return hash_to_backward_task[hash];
}

void Simulator::free_all()
{
  offset = 0;
}

void* Simulator::allocate(size_t num_elements, DataType type)
{
  size_t element_size = 0;
  switch (type) {
    case DT_FLOAT:
      element_size = sizeof(float);
      break;
    case DT_DOUBLE:
      element_size = sizeof(double);
      break;
    case DT_INT32:
      element_size = sizeof(int32_t);
      break;
    case DT_INT64:
      element_size = sizeof(int64_t);
      break;
    case DT_BOOLEAN:
      element_size = sizeof(bool);
      break;
    default:
      assert(false);
  }
  void* ret_ptr = base_ptr + offset;
  offset += element_size * num_elements;
  if ((size_t)offset > capacity) {
    fprintf(stderr, "Simulator cannot measure some operators' performance."
        " Increate --simulator-workspace-size to at least %zd\n", offset);
    exit(0);
  }
  return ret_ptr;
}

void Simulator::add_task_dependencies_with_xfer(SimTask* src_task,
                                                SimTask* dst_task,
                                                size_t message_size)
{
  std::vector<Comm_device *> path = machine->get_comm_path(src_task->mem, dst_task->mem);
  // print the communication path
  // printf("Path from %s to %s is: ", src_task->mem->name.c_str(), dst_task->mem->name.c_str());
  // for (size_t i = 0; i < path.size(); i++) {
  //   printf("%s ", path[i]->name.c_str());
  // }
  // printf("\n");

  if (path.empty()) {
    src_task->add_next_task(dst_task);
    return;
  }
  assert(message_size > 0);
  std::vector<std::vector<SimTask *>> all_tasks;
  // Limit the max number of segments per message
  int seg_size = segment_size;
  int num_segment = message_size / seg_size;
  if (message_size % seg_size != 0) {
    num_segment += 1;
  }
  if (num_segment > max_num_segments) {
    num_segment = max_num_segments;
    seg_size = message_size / num_segment;
  }
  // Create all the comm tasks
  // Divide messages into segments
  for (size_t i = 0; i < path.size(); i++) {
    all_tasks.push_back({});
    for (int j = 0; j < num_segment; j++) {
      int cur_seg_size = seg_size;
      if (j == num_segment - 1) {
        cur_seg_size = message_size - (num_segment - 1) * seg_size;
      }
      std::string name = "seg " + std::to_string(j) + " from " + src_task->name + " to " + dst_task->name;
      SimTask *cur_task = task_manager->new_comm_task(name, path[i], cur_seg_size);
      all_tasks[i].push_back(cur_task);
    }
  }

  // Add dependencies among the comm tasks
  for (size_t i = 0; i < path.size(); i++) {
    for (int j = 0; j < num_segment; j++) {
      if (i == 0) {
        src_task->add_next_task(all_tasks[i][j]);
      }
      if (i == path.size() - 1) {
        all_tasks[i][j]->add_next_task(dst_task);
      }
      if (i > 0) {
        all_tasks[i-1][j]->add_next_task(all_tasks[i][j]);
      }
    }
  }

  // Add special dependencies for upi_ins, upi_outs, nic_ins, and nic_outs to prevent communication
  // overlap between upi_ins and upi_outs, and between nic_ins and nic_outs.
  if (num_segment > 1 and path.size() >= 2) {
    for (size_t i = 0; i < path.size(); i++) {
      for (int j = 1; j < num_segment; j++) {
        if (((Comm_device *)all_tasks[i][j]->device)->comm_type == Comm_device::NIC_OUT_COMM or
            ((Comm_device *)all_tasks[i][j]->device)->comm_type == Comm_device::UPI_OUT_COMM) {
          all_tasks[i+1][j-1]->add_next_task(all_tasks[i][j]);
        }
      }
    }
  }
}

[[noreturn]] void handle_measure_operator_cost_unimplemented(Op const *op) {
    std::cerr << "measure_operator_cost not implemented for op "
              << op->name
              << " (type " << op->op_type << ")"
              << ". Please report this issue to the FlexFlow developers."
              << std::endl;
    std::abort();
}

CostMetrics Simulator::measure_operator_cost(Op* op, const ParallelConfig& config)
{
  size_t hash = 17 * 31 + (size_t)(op);
  hash = hash * 31 + std::hash<int>()(config.device_type);
  hash = hash * 31 + std::hash<int>()(config.nDims);
  for (int i = 0; i < config.nDims; i++)
    hash = hash * 31 + std::hash<int>()(config.dim[i]);
  std::map<size_t, CostMetrics>::const_iterator iter =
    hash_to_operator_cost.find(hash);
  if (iter == hash_to_operator_cost.end()) {
    CostMetrics cost_metrics;
    bool is_implemented = op->measure_operator_cost(this, config, cost_metrics);
    if (! is_implemented) {
      handle_measure_operator_cost_unimplemented(op);
    }
    hash_to_operator_cost[hash] = cost_metrics;
    return cost_metrics;
  } else {
    return iter->second;
  }
}

float Simulator::simulate_runtime(const FFModel* model,
                                  const std::map<Op*, ParallelConfig>& global,
                                  CompMode comp_mode)
{
  return this->simulate_runtime(model, global, comp_mode, "");
}

float Simulator::simulate_runtime(const FFModel* model,
                                  const std::map<Op*, ParallelConfig>& global,
                                  CompMode comp_mode,
                                  std::string const &export_file_name)
{
  // printf("%s\n", machine->to_string().c_str());
  task_manager->reset();
  // Step 1: register forward and backward tasks
  for (size_t l = 0; l < model->layers.size(); l++) {
    Op* op = model->layers[l];
    ParallelConfig config = global.find(op)->second;
    CostMetrics cost_metrics = measure_operator_cost(op, config);
    float forward_time = cost_metrics.forward_time;
    float backward_time = cost_metrics.backward_time;
    for (int j = 0; j < config.num_parts(); j++) {
      SimTask* task1 = task_manager->new_forward_task(op, j);
      task1->device = machine->get_gpu(config.device_ids[j]);
      task1->mem = machine->get_gpu_fb_mem(config.device_ids[j]);
      task1->run_time = forward_time;
      if (comp_mode == COMP_MODE_TRAINING) {
        SimTask* task2 = task_manager->new_backward_task(op, j);
        task2->device = machine->get_gpu(config.device_ids[j]);
        task2->mem = machine->get_gpu_fb_mem(config.device_ids[j]);
        task2->run_time = backward_time;
        task1->add_next_task(task2);
      }
    }
  }
  // Step 2: insert dependencies and comm. tasks before compute tasks
  for (size_t l = 0; l < model->layers.size(); l++) {
    Op* op = model->layers[l];
    ParallelConfig config = global.find(op)->second;
    for (int j = 0; j < op->numInputs; j++) {
      Tensor t = op->inputs[j];
      Op* pre_op = t.owner_op;
      if (pre_op == NULL)
        continue;
      ParallelConfig pre_config = global.find(pre_op)->second;
      for (int dstId = 0; dstId < config.num_parts(); dstId ++) {
        Domain dstR = op->get_input_tensor_shape(config, j, dstId);
        for (int srcId = 0; srcId < pre_config.num_parts(); srcId ++) {
          Domain srcR = pre_op->get_output_tensor_shape(pre_config, t.owner_idx, srcId);
          if (dstR.intersection(srcR).get_volume() > 0) {
            // Forward dependency
            {
              SimTask* dstT = task_manager->get_forward_task(op, dstId);
              SimTask* srcT = task_manager->get_forward_task(pre_op, srcId);
              add_task_dependencies_with_xfer(srcT, dstT, dstR.intersection(srcR).get_volume());
            }
            // Backward dependency
            if (comp_mode == COMP_MODE_TRAINING) {
              SimTask* dstT = task_manager->get_backward_task(op, dstId);
              SimTask* srcT = task_manager->get_backward_task(pre_op, srcId);
              add_task_dependencies_with_xfer(dstT, srcT, dstR.intersection(srcR).get_volume());
            }
          }
        }
      }
    }
  }
#ifdef FF_USE_NCCL
  // Do nothing since we will calculate NCCL cost at the end
#else
  // Step 2.5: add finals tasks for each compute device to capture the returning comm tasks
  // from parameter servers
  std::vector<SimTask*> finals;
  for (int d = 0; d < machine->get_num_gpus(); d++) {
    SimTask* t = task_manager->new_barrier_task();
    t->device = machine->get_gpu(d);
    t->mem = machine->get_gpu_fb_mem(d);
    t->run_time = 0;
    finals.push_back(t);
  }

  if (model->config.search_overlap_backward_update && comp_mode == COMP_MODE_TRAINING) {
    // Step 3a: consider backpropagation and weight update are overlapped
    for (int l = model->layers.size()-1; l >= 0; l--) {
      Op* op = model->layers[l];
      ParallelConfig pc = global.find(op)->second;
      for (int j = 0; j < op->numWeights; j++) {
        std::set<int> synched;
        for (int firstId = 0; firstId < pc.num_parts(); firstId++)
          if (synched.find(firstId) == synched.end()) {
            synched.insert(firstId);
            Domain firstR = op->get_weight_tensor_shape(pc, j, firstId);
            // Add a compute task for parameter update
            SimTask* updateT = task_manager->new_update_task();
            updateT->device = machine->get_gpu(pc.device_ids[firstId]);
            updateT->mem = machine->get_gpu_fb_mem(pc.device_ids[firstId]);
            // TODO add parameter synchronization time
            updateT->run_time = 0.0f; // Assume update task takes no time
            for (int nextId = firstId+1; nextId < pc.num_parts(); nextId++) {
              Domain nextR = op->get_weight_tensor_shape(pc, j, nextId);
              if (firstR.intersection(nextR).get_volume() > 0) {
                // Assert all or nothing:
                // The two weights must be fully overlapped or not at all
                assert(firstR == nextR);
                assert(synched.find(nextId) == synched.end());
                synched.insert(nextId);
                // Add comm. tasks from backT to updateT
                SimTask* backT = task_manager->get_backward_task(op, nextId);
                add_task_dependencies_with_xfer(backT, updateT, firstR.get_volume());
                // Add comm. tasks from updateT to finalT
                SimTask* finalT = finals[backT->device->device_id];
                add_task_dependencies_with_xfer(updateT, finalT, firstR.get_volume());
              }
            }
          }
      }
    }
  } else if (comp_mode == COMP_MODE_TRAINING) {
    // Step 3b: Bulk Synchronous Model
    // Add a per-device barrier before weight update
    std::vector<SimTask*> barriers;
    for (int d = 0; d < machine->get_num_gpus(); d++) {
      SimTask* t = task_manager->new_barrier_task();
      t->device = machine->get_gpu(d);
      t->mem = machine->get_gpu_fb_mem(d);
      t->run_time = 0;
      barriers.push_back(t);
    }
    for (size_t l = 0; l < model->layers.size(); l++) {
      Op* op = model->layers[l];
      ParallelConfig pc = global.find(op)->second;
      for (int j = 0; j < pc.num_parts(); j++) {
        SimTask* backT = task_manager->get_backward_task(op, j);
        backT->add_next_task(barriers[backT->device->device_id]);
      }
    }
    for (size_t l = 0; l < model->layers.size(); l++) {
      Op* op = model->layers[l];
      ParallelConfig pc = global.find(op)->second;
      for (int j = 0; j < op->numWeights; j++) {
        std::set<int> synched;
        for (int firstId = 0; firstId < pc.num_parts(); firstId++)
          if (synched.find(firstId) == synched.end()) {
            synched.insert(firstId);
            Domain firstR = op->get_weight_tensor_shape(pc, j, firstId);
            // Add a compute task for parameter update
            SimTask* updateT = task_manager->new_update_task();
            updateT->device = machine->get_gpu(pc.device_ids[firstId]);
            updateT->mem = machine->get_gpu_fb_mem(pc.device_ids[firstId]);
            updateT->run_time = 0.0f; // Assume update task takes no time
            barriers[updateT->device->device_id]->add_next_task(updateT);
            for (int nextId = firstId+1; nextId < pc.num_parts(); nextId++) {
              Domain nextR = op->get_weight_tensor_shape(pc, j, nextId);
              if (firstR.intersection(nextR).get_volume() > 0) {
                // Assert all or nothing:
                // The two weights must be fully overlapped or not at all
                assert(firstR == nextR);
                assert(synched.find(nextId) == synched.end());
                synched.insert(nextId);
                SimTask* backT = task_manager->get_backward_task(op, nextId);
                assert(backT->device->device_id == pc.device_ids[nextId]);
                SimTask* barrierT = barriers[backT->device->device_id];
                // Add comm. tasks from barrierT to updateT
                add_task_dependencies_with_xfer(barrierT, updateT, firstR.get_volume());
                // Add comm. tasks from updateT to finalT
                SimTask* finalT = finals[backT->device->device_id];
                add_task_dependencies_with_xfer(updateT, finalT, firstR.get_volume());
              }
            }
          }
      }
    }
  } else {
    assert(comp_mode == COMP_MODE_INFERENCE);
  }
#endif
  // Step 4: add ready tasks into ready_queue
  std::priority_queue<SimTask*, std::vector<SimTask*>, SimTaskCompare> ready_queue;
  for (size_t i = 0; i < task_manager->global_task_id; i++)
    if (task_manager->tasks[i]->counter == 0)
      ready_queue.push(task_manager->tasks[i]);
  // Step 5: perform simulation
  float sim_time = 0.0f;
  std::map<Device*, float> device_times;
  size_t idx = 0;
  DotFile<SimTask *> taskGraph;
  bool export_taskgraph = (export_file_name != "");
  if (export_taskgraph) {
    taskGraph.set_filename(export_file_name);
  }
  while (!ready_queue.empty()) {
    // Find the task with the earliest start time
    SimTask* cur_task = ready_queue.top();
    ready_queue.pop();
    float ready_time = 0;
    if (device_times.find(cur_task->device) != device_times.end()) {
      ready_time = device_times[cur_task->device];
    }
    float start_time = std::max(ready_time, cur_task->ready_time);
    float end_time = start_time + cur_task->run_time;
    device_times[cur_task->device] = end_time;
    if (export_taskgraph) {
      std::map<std::string, std::string> nodeAttrs;
      std::ostringstream label;
      label << "\"{ ";
      if (!(cur_task->name).empty()) {
        label << cur_task->name << " | ";
      }
      label << cur_task->get_type_str() << " | ";
      label << "{ " << start_time << " | " << end_time << " }";
      label << " }\"";
      nodeAttrs["label"] = label.str();
      nodeAttrs["shape"] = "record";
      taskGraph.add_node(cur_task, nodeAttrs);
    }
    // printf("task[%lu] type(%d) run_time(%.4lf) ready_time(%.4lf) start_time(%.4lf) device(%s)\n",
    //       idx, cur_task->type, cur_task->run_time, ready_time, start_time, (cur_task->device->name).c_str());
    if (end_time > sim_time)
      sim_time = end_time;
    for (size_t i = 0; i < cur_task->next_tasks.size(); i++) {
      SimTask* next = cur_task->next_tasks[i];
      if (export_taskgraph) {
        taskGraph.add_edge(cur_task, next);
      }
      next->ready_time = std::max(next->ready_time, end_time);
      next->counter --;
      if (next->counter == 0) {
        ready_queue.push(next);
      }
    }
    idx++;
  }
  if (export_taskgraph) {
    taskGraph.close();
  }
  // Assert all tasks were processed
  assert(idx == task_manager->global_task_id);
#ifdef FF_USE_NCCL
  if (comp_mode == COMP_MODE_TRAINING) {
    for (size_t l = 0; l < model->layers.size(); l++) {
      Op* op = model->layers[l];
      ParallelConfig pc = global.find(op)->second;
      // Since all NCCL calls are blocking, we can add the NCCL cost
      // sequentially 
      for (int j = 0; j < op->numWeights; j++) {
        std::set<int> synched;
        for (int firstId = 0; firstId < pc.num_parts(); firstId++)
          if (synched.find(firstId) == synched.end()) {
            synched.insert(firstId);
            Domain firstR = op->get_weight_tensor_shape(pc, j, firstId);
            Device* firstDevice = machine->get_gpu(pc.device_ids[firstId]);
            float nccl_time = 0.0f;
            for (int nextId = firstId+1; nextId < pc.num_parts(); nextId++) {
              Domain nextR = op->get_weight_tensor_shape(pc, j, nextId);
              if (firstR.intersection(nextR).get_volume() > 0) {
                // Assert all or nothing:
                // The two weights must be fully overlapped or not at all
                assert(firstR == nextR);
                assert(synched.find(nextId) == synched.end());
                synched.insert(nextId);
                Device* nextDevice = machine->get_gpu(pc.device_ids[nextId]);
                // Compute the bandwidth between firstDevice/nextDevice
                float bandwidth = 0.0f;
                if (firstDevice->node_id == nextDevice->node_id) {
                  bandwidth = machine->get_intra_node_gpu_bandwidth();
                } else {
                  bandwidth = machine->get_inter_node_gpu_bandwidth();
                }
                nccl_time = std::max(nccl_time, (float)firstR.get_volume() * sizeof(float) / bandwidth);
              }
            }
            // Add ncclTime to sim_time given nccl calls are blocking
            sim_time += nccl_time;
          }
      }
    }
  } else {
    assert(comp_mode == COMP_MODE_INFERENCE);
  }
#endif
  // Step 6: add penalty to strategies that exceed the memory limits on devices
  std::vector<size_t> gpu_mem_usage(machine->get_num_gpus(), 0);
  float memory_penalty = 0.0f;
  for (size_t l = 0; l < model->layers.size(); l++) {
    Op* op = model->layers[l];
    ParallelConfig config = global.find(op)->second;
    CostMetrics cost_metrics = measure_operator_cost(op, config);
    size_t memory_requirement = cost_metrics.memory_requirement;
    for (int j = 0; j < config.num_parts(); j++) {
      gpu_mem_usage[config.device_ids[j]] += memory_requirement;
    }
  }
  if (export_file_name != "") {  
    for (int i = 0; i < total_num_gpus; i++) {
        printf("Before penalty, dev id %d, usage %zu \n", i, gpu_mem_usage[i]); 
    }
  }
  // Penalize the total runtiem by 1ms if we exceed the memory budget by 1MB
  for (int i = 0; i < machine->get_num_gpus(); i++) {
    Mem_device* gpu_fb_mem = machine->get_gpu_fb_mem(i);
    if (gpu_mem_usage[i] > gpu_fb_mem->capacity and gpu_fb_mem->capacity >= 0)
      memory_penalty += (gpu_mem_usage[i] - gpu_fb_mem->capacity) * 1e-6;
  }
  //if (memory_penalty > 0.0f)
  //  printf("Memory penalty = %.4lf ms\n", memory_penalty);
  return sim_time + memory_penalty;
}
