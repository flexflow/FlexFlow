/* Copyright 2018 Stanford
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
#include "model.h"
#include "cuda_helper.h"
//#include "realm/runtime_impl.h"
//#include "realm/cuda/cuda_module.h"

FFHandler UtilityTasks::init_cuda_task(
              const Task *task,
              const std::vector<PhysicalRegion> &regions,
              Context ctx, HighLevelRuntime *runtime)
{
  assert(regions.size() == 0);
  assert(task->arglen == sizeof(size_t));
  size_t workSpaceSize = *(const size_t*) task->args;
  printf("workSpaceSize (%d MB)\n", workSpaceSize / 1024 / 1024);
  FFHandler handle;
  handle.workSpaceSize = workSpaceSize;
  checkCUDA(cublasCreate(&handle.blas));
  checkCUDNN(cudnnCreate(&handle.dnn));
  //std::set<Memory> memFB;
  //assert(memFB.size() == 1);
  //assert(memFB.begin()->kind() == Memory::GPU_FB_MEM);
  //Realm::MemoryImpl* memImpl =
  //    Realm::get_runtime()->get_memory_impl(*memFB.begin());
  //Realm::Cuda::GPUFBMemory* memFBImpl = (Realm::Cuda::GPUFBMemory*) memImpl;
  //off_t offset = memFBImpl->alloc_bytes(workSpaceSize);
  //handle.workSpace = memFBImpl->get_direct_ptr(offset, 0);
  checkCUDA(cudaMalloc(&handle.workSpace, workSpaceSize));
  return handle;
}

void UtilityTasks::dummy_task(const Task *task,
                             const std::vector<PhysicalRegion> &regions,
                             Context ctx, Runtime *runtime)
{}

__inline__
int calc_offset(int c, int y, int x, int yscale, int xscale)
{
  return (c * yscale * xscale + y * xscale + x);
}

void nearest_neighbor(unsigned char* image,
                      unsigned char* buffer,
                      int height, int width,
                      int orig_height, int orig_width,
                      float height_scale, float width_scale)
{
  // Note buffer is in HWC layout while image is in CHW layout
  for (int y = 0; y < height; y++) {
    int y0 = std::min(static_cast<int>(roundf(y * height_scale)), orig_height - 1);
    for (int x = 0; x < width; x++) {
      int x0 = std::min(static_cast<int>(roundf(x * width_scale)), orig_width - 1);
      for (int c = 0; c < 3; c++) {
        int origOffset = calc_offset(y0, x0, c, orig_width, 3);
        int offset = calc_offset(c, y, x, height, width);
        image[offset] = buffer[origOffset];
      }
    }
  }
}

/*
  regions[0]: image (unsigned char)
  regions[1]: label (int)
*/
void UtilityTasks::load_images_task(
         const Task *task,
         const std::vector<PhysicalRegion> &regions,
         Context ctx, HighLevelRuntime *runtime)
{
#ifdef USE_DATA_LOADER
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  const AccessorWO<unsigned char, 3> acc_image(regions[0], FID_DATA);
  const AccessorWO<int, 1> acc_label(regions[1], FID_DATA);
  Rect<3> rect_image;
  Rect<1> rect_label;
  unsigned char *buffer = (unsigned char*) malloc(3000 * 3000 * 3);
  rect_image = runtime->get_index_space_domain(ctx,
                    task->regions[0].region.get_index_space());
  rect_label = runtime->get_index_space_domain(ctx,
                    task->regions[1].region.get_index_space());
  assert(acc_image.accessor.is_dense_arbitrary(rect_image));
  assert(acc_label.accessor.is_dense_arbitrary(rect_label));
  unsigned char *image_ptr = acc_image.ptr(rect_image.lo);
  int *label_ptr = acc_label.ptr(rect_label.lo);
  const DataLoadMeta* meta = (DataLoadMeta*) task->local_args;
  int height = rect_image.hi[0] - rect_image.lo[0] + 1;
  int width = rect_image.hi[1] - rect_image.lo[1] + 1;
  int numImages = (rect_image.hi[2] - rect_image.lo[2] + 1) / 3;
  assert((rect_image.hi[2] - rect_image.lo[2] + 1) % 3 == 0);
  assert(meta->numImages == numImages);
  for (int idx = 0; idx < numImages; idx ++) {
    label_ptr[idx] = meta->labels[idx];
    FILE *file;
    if ((file = fopen(meta->files[idx], "rb")) == NULL) {
      fprintf(stderr, "cannot open %s\n", meta->files[idx]);
      continue;
    }
    struct jpeg_decompress_struct cinfo;
    struct jpeg_error_mgr jerr;
    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_decompress(&cinfo);
    jpeg_stdio_src(&cinfo, file);
    jpeg_read_header(&cinfo, TRUE);
    jpeg_start_decompress(&cinfo);
    if (cinfo.output_components != 3) {
      printf(stderr, "skip non-RGB file %s\n", meta->files[idx]);
      jpeg_finish_decompress(&cinfo);
      jpeg_destroy_decompress(&cinfo);
      fclose(file);
      continue;
    }
    int origHeight = cinfo.output_height;
    int origWidth = cinfo.output_width;
    int rowStride = width * cinfo.output_components;
    JSAMPARRAY array;
    array = (*cinfo.mem->alloc_sarray) ((j_common_ptr) &cinfo, JPOOL_IMAGE, rowStride, 1);
    while (cinfo.output_scanline < cinfo.output_height) {
      jpeg_read_scanlines(&cinfo, buffer, 1);
      memcpy(buffer, array[0], rowStride * sizeof(JSAMPLE));
      buffer += rowStride;
    }
    jpeg_finish_decompress(&cinfo);
    jpeg_destroy_decompress(&cinfo);
    fclose(file);
    float heightScale = static_cast<float>(origHeight) / height;
    float widthScale = static_cast<float>(origWidth) / width;
    nearest_neighbor(image_ptr, buffer, height, width,
                     origHeight, origWidth, heightScale, widthScale);
    image_ptr += 3 * height * width;
  }
  free(buffer);
#endif
}

__global__
void apply_normalize(float *tensor_ptr, const unsigned char *rgb_ptr,
                     size_t size, size_t hxw)
{
  const float mean[3] = {0.485, 0.456, 0.406};
  const float var[3] = {0.229, 0.224, 0.225};

  CUDA_KERNEL_LOOP(i, size)
  {
    // decide the color of the current position by assuming NCHW layout
    int c = (i / hxw) % 3;
    tensor_ptr[i] = (static_cast<float>(rgb_ptr[i]) / 256 - mean[c]) / var[c];
  }
}

/*
  regions[0](O): input_images
  regions[1](I): input_rgb
*/
__host__
void UtilityTasks::normalize_images_task(
         const Task* task,
         const std::vector<PhysicalRegion> &regions,
         Context ctx, Runtime *runtime)
{
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  const AccessorWO<float, 3> acc_tensor(regions[0], FID_DATA);
  const AccessorRO<unsigned char, 3> acc_rgb(regions[1], FID_DATA);
  Rect<3> rect_tensor, rect_rgb;
  rect_tensor = runtime->get_index_space_domain(
                    ctx, task->regions[0].region.get_index_space());
  rect_rgb = runtime->get_index_space_domain(
                 ctx, task->regions[1].region.get_index_space());
  assert(acc_tensor.accessor.is_dense_arbitrary(rect_tensor));
  assert(acc_rgb.accessor.is_dense_arbitrary(rect_rgb));
  assert(rect_tensor == rect_rgb);
  size_t w = rect_tensor.hi[0] - rect_tensor.lo[0] + 1;
  size_t h = rect_tensor.hi[1] - rect_tensor.lo[1] + 1;
  float *tensor_ptr = acc_tensor.ptr(rect_tensor.lo);
  const unsigned char *rgb_ptr = acc_rgb.ptr(rect_rgb.lo);
  apply_normalize<<<GET_BLOCKS(rect_tensor.volume()), CUDA_NUM_THREADS>>>(
      tensor_ptr, rgb_ptr, rect_tensor.volume(), h * w);
}

__global__
void init_image_kernel(float* ptr, coord_t size)
{
  const coord_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < size) {
    ptr[tid] = 1.0f;
  }
}

__global__
void init_label_kernel(int* ptr, coord_t size)
{
  const coord_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < size) {
    ptr[tid] = 1;
  }
}

void UtilityTasks::init_images_task(const Task *task,
                                    const std::vector<PhysicalRegion> &regions,
                                    Context ctx, Runtime *runtime)
{
  const int BLKSIZE = 512;
  const AccessorWO<float, 3> acc_image(regions[0], FID_DATA);
  Rect<3> rect_image;
  rect_image = runtime->get_index_space_domain(ctx, task->regions[0].region.get_index_space());
  assert(acc_image.accessor.is_dense_arbitrary(rect_image));
  float *image_ptr = acc_image.ptr(rect_image.lo);
  int num_blocks = (rect_image.volume() + BLKSIZE - 1) / BLKSIZE;
  init_image_kernel<<<num_blocks, BLKSIZE>>>(image_ptr, rect_image.volume());
}

void UtilityTasks::init_labels_task(const Task *task,
                                    const std::vector<PhysicalRegion> &regions,
                                    Context ctx, Runtime *runtime)
{
  const int BLKSIZE = 512;
  const AccessorWO<int, 1> acc_label(regions[0], FID_DATA);
  Rect<1> rect_label;
  rect_label = runtime->get_index_space_domain(ctx, task->regions[0].region.get_index_space());
  assert(acc_label.accessor.is_dense_arbitrary(rect_label));
  int *label_ptr = acc_label.ptr(rect_label.lo);
  int num_blocks = (rect_label.volume() + BLKSIZE - 1) / BLKSIZE;
  init_label_kernel<<<num_blocks, BLKSIZE>>>(label_ptr, rect_label.volume());
}

//void FFModel::load_images(int batch_id)
//{
//  assert(false);
//}

void FFModel::prefetch()
{
  for (size_t i = 0; i < layers.size(); i++)
    layers[i]->prefetch(*this);
}

//void FFModel::update()
//{
//  for (int i = layers.size() - 1; i >= 0; i--)
//    layers[i]->update(*this);
//}


template <typename T>
bool Parameter::set_weights(const FFModel& ff,
                            const std::vector<int>& dims,
                            const T* data)
{
  //TODO: check data type matches
  size_t volume = 1;
  // Check dimensions
  if (numDim != (int)dims.size())
    return false;
  for (int i = 0; i < numDim; i++) {
    if (adim[numDim-1-i] != dims[i])
      return false;
    volume = volume * dims[i];
  }
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  RegionRequirement req(region, READ_WRITE, EXCLUSIVE, region);
  req.add_field(FID_DATA);
  InlineLauncher launcher(req);
  PhysicalRegion region = runtime->map_region(ctx, launcher);
  region.wait_until_valid();
  switch (numDim) {
    case 1:
    {
      TensorAccessorW<T, 1> acc(region, req, FID_DATA, ctx, runtime, true);
      assert(acc.rect.volume() == volume);
      memcpy(acc.ptr, data, volume * sizeof(T));
      break;
    }
    case 2:
    {
      TensorAccessorW<T, 2> acc(region, req, FID_DATA, ctx, runtime, true);
      assert(acc.rect.volume() == volume);
      memcpy(acc.ptr, data, volume * sizeof(T));
      break;
    }
    case 3:
    {
      TensorAccessorW<T, 3> acc(region, req, FID_DATA, ctx, runtime, true);
      assert(acc.rect.volume() == volume);
      memcpy(acc.ptr, data, volume * sizeof(T));
      break;
    }
    case 4:
    {
      TensorAccessorW<T, 4> acc(region, req, FID_DATA, ctx, runtime, true);
      assert(acc.rect.volume() == volume);
      memcpy(acc.ptr, data, volume * sizeof(T));
      break;
    }
    default:
      // Unsupported dim
      assert(false);
  }
  runtime->unmap_region(ctx, region);
  return true;
}

template <typename T>
bool Parameter::get_weights(const FFModel& ff,
                            T* data)
{
  //TODO: check data type matches
  size_t volume = 1;
  for (int i = 0; i < numDim; i++) {
    volume = volume * adim[i];
  }
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  RegionRequirement req(region, READ_ONLY, EXCLUSIVE, region);
  req.add_field(FID_DATA);
  InlineLauncher launcher(req);
  PhysicalRegion region = runtime->map_region(ctx, launcher);
  region.wait_until_valid();
  switch (numDim) {
    case 1:
    {
      TensorAccessorR<T, 1> acc(region, req, FID_DATA, ctx, runtime);
      assert(acc.rect.volume() == volume);
      memcpy(data, acc.ptr, volume * sizeof(T));
      break;
    }
    case 2:
    {
      TensorAccessorR<T, 2> acc(region, req, FID_DATA, ctx, runtime);
      assert(acc.rect.volume() == volume);
      memcpy(data, acc.ptr, volume * sizeof(T));
      break;
    }
    case 3:
    {
      TensorAccessorR<T, 3> acc(region, req, FID_DATA, ctx, runtime);
      assert(acc.rect.volume() == volume);
      memcpy(data, acc.ptr, volume * sizeof(T));
      break;
    }
    case 4:
    {
      TensorAccessorR<T, 4> acc(region, req, FID_DATA, ctx, runtime);
      assert(acc.rect.volume() == volume);
      memcpy(data, acc.ptr, volume * sizeof(T));
      break;
    }
    default:
      // Unsupported dim
      assert(false);
  }
  runtime->unmap_region(ctx, region);
  return true;
}

template bool Parameter::set_weights<float>(const FFModel& ff, const std::vector<int>& dims, const float* data);
template bool Parameter::get_weights<float>(const FFModel& ff, float* data);