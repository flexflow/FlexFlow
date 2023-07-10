// /* Copyright 2023 CMU, Facebook, LANL, MIT, NVIDIA, and Stanford (alphabetical)
//  *
//  * Licensed under the Apache License, Version 2.0 (the "License");
//  * you may not use this file except in compliance with the License.
//  * You may obtain a copy of the License at
//  *
//  *     http://www.apache.org/licenses/LICENSE-2.0
//  *
//  * Unless required by applicable law or agreed to in writing, software
//  * distributed under the License is distributed on an "AS IS" BASIS,
//  * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  * See the License for the specific language governing permissions and
//  * limitations under the License.
//  */

// #include "flexflow/ops/sampling.h"
// #include "flexflow/utils/cuda_helper.h"
// #include <thrust/scan.h>
// #include <thrust/sort.h>
// #include "cub/cub.cuh"

// namespace FlexFlow {

// template <typename DT>
// __global__ void mask_value_above_top_p(DT *input_ptr,
//                                        DT *cumsum_ptr,
//                                        float top_p,
//                                        int total_eles) {
//   CUDA_KERNEL_LOOP(i, total_eles) {
//     if ((cumsum_ptr[i] - input_ptr[i]) > static_cast<DT>(top_p)) {
//       input_ptr[i] = 0.0;
//     }
//   }
// }

// template <typename DT>
// __global__ void re_normalized(DT *input_ptr, DT div, int length) {
//   CUDA_KERNEL_LOOP(i, length) {
//     input_ptr[i] /= div;
//   }
// }

// template <typename DT>
// __global__ void sampleMultinomialOnce(long long N, DT *input_ptr) {
//   extern __shared__ unsigned char my_smem[];
//   __shared__ bool found;
//   __shared__ unsigned foundPos;

//   float *smem = reinterpret_cast<float *>(my_smem);

//   float accZero = static_cast<float>(0);
//   DT zero = static_cast<DT>(0);

//   for (int64_t curDist = blockIdx.x; curDist < distributions;
//        curDist += gridDim.x) {

//     float sum = accZero;
//     DT val;

//     for (int cat = threadIdx.x; cat < N; cat += blockDim.x) {
//       val = dist[curDist * stride_dist + cat * stride_categories];
//       CUDA_KERNEL_ASSERT(!at::_isnan(val));
//       CUDA_KERNEL_ASSERT(!_isinf(val));
//       CUDA_KERNEL_ASSERT(!(val < zero));
//       sum = sum + static_cast<float>(val);
//     }


//     //sum
//     sum = BlockReduceSum(sum, smem);

//     if (threadIdx.x == 0) {
//       foundPos = 0;
//       smem[0] = sum;
//       smem[1] = sampled[curDist];
//     }

//     __syncthreads();
//     sum = smem[0];

//     DT sample = static_cast<DT>(smem[1]);
//     __syncthreads();

//     if (sum == accZero) {
//       // Choose the first element
//       if (threadIdx.x == 0) {
//         dest[curDist] = 0;
//       }

//       continue;
//     }

//     //ELSE
//     int chunks = (categories + (int)blockDim.x - 1) / blockDim.x;
//     float prevHighProb = accZero;

//     found = false;
//     for (int chunk = 0; chunk < chunks && !found; ++chunk) {

//        int cat = chunk * blockDim.x + threadIdx.x;
//        float dist_val = cat < categories ?
//                              static_cast<float>(dist[curDist * stride_dist + cat * stride_categories]) / sum :
//                              accZero;

//         smem[threadIdx.x] = dist_val;
//       __syncthreads();

//       // Perform an inclusive prefix sum of the shared memory contents
//       for (int offset = 1; offset < blockDim.x; offset *= 2) {
//         float val = accZero;

//         if (threadIdx.x >= offset) {
//           val = smem[threadIdx.x - offset] + smem[threadIdx.x];
//         }

//         __syncthreads();
//         if (threadIdx.x >= offset) {
//           smem[threadIdx.x] = val;
//         }
//         __syncthreads();
//       }

//       // Each thread will check to see if the sample falls in its
//       // bucket
//       DT curBucket =
//           static_cast<DT>(smem[threadIdx.x] + prevHighProb);
//       DT prevBucket = static_cast<DT>(
//           threadIdx.x == 0 ? prevHighProb
//                           : smem[threadIdx.x - 1] + prevHighProb);
//       bool inBucket =
//           (cat < categories) &&
//           (!(sample >= curBucket) &&
//           (sample >= prevBucket) &&
//           (dist_val > zero));

//       if (inBucket) {
//         // We're done; we have the sample
//         // Torch indices are 1-based
//         atomicMax(&foundPos, cat);
//         found = true;
//       }

//       // Store the previous scan's high value for future use
//       prevHighProb = prevHighProb + smem[blockDim.x - 1];
//       __syncthreads();                     
//     }

//     if (threadIdx.x == 0) {
//       if (found) {
//           dest[curDist] = foundPos;
//       } else {
//         // This should address a rare bug where we don't select a valid index. This likely occurs when
//         // due to floating point arithmetic rounding errors, our cumulative sum does not add up to 1, but
//         // and our uniform sample is greater than this value. In this case we likely have unitialized memory
//         // in dest[curDist]. So basically we will loop through the distribution and pick the largest index
//         // where the distribution is non-zero. This is obviously terribly inefficient, but due to the
//         // rarity in which this occurs, this should not be an issue.
//         for (int cat = categories - 1; cat >= 0; --cat) {
//           if (dist[curDist * stride_dist + cat * stride_categories] > zero) {
//             dest[curDist] = cat;
//             break;
//           }
//         }
//       }
//     }


//   }
// }


// /*static*/
// template <typename DT>
// void Sampling::forward_kernel(SamplingMeta const *m,
//                               DT *input_ptr,
//                               int *indices_ptr,
//                               float top_p,
//                               int length,
//                               int batch_size,
//                               cudaStream_t stream) {
//   // 1. sort
//   // 2. cumsum
//   // how to do it in parallel?

//   checkCUDA(cudaMemcpy(static_cast<DT *>(m->origin_ptr),
//                        input_ptr,
//                        sizeof(DT) * 15 * length,
//                        cudaMemcpyDeviceToDevice));

//   std::cout << "asdqs: " << length << "\n";

//   for (int i = 0; i < 15; i++) {
//     thrust::sort(thrust::device,
//                  input_ptr + i * length,
//                  input_ptr + (i + 1) * length,
//                  thrust::greater<DT>());
//     thrust::sort(thrust::device,
//                  static_cast<DT *>(m->origin_ptr) + i * length,
//                  static_cast<DT *>(m->origin_ptr) + (i + 1) * length,
//                  thrust::greater<DT>());
//     thrust::inclusive_scan(thrust::device,
//                            input_ptr + i * length,
//                            input_ptr + (i + 1) * length,
//                            static_cast<DT *>(m->cumsum_ptr) + i * length);
//   }
//   std::cout << "sdsd"
//             << "\n";

//   // 3. mask
//   int parallelism = 15 * length;
//   mask_value_above_top_p<DT><<<GET_BLOCKS(parallelism),
//                                min(CUDA_NUM_THREADS, parallelism),
//                                0,
//                                stream>>>(
//       input_ptr, static_cast<DT *>(m->cumsum_ptr), top_p, parallelism);

//   // 4. sum/div
//   std::cout << "sadsd2www"
//             << "\n";
//   for (int i = 0; i < 15; i++) {
//     DT sum = thrust::reduce(
//         thrust::device, input_ptr + i * length, input_ptr + (i + 1) * length);
//     parallelism = length;

//     re_normalized<DT><<<GET_BLOCKS(parallelism),
//                         min(CUDA_NUM_THREADS, parallelism),
//                         0,
//                         stream>>>(input_ptr + i * length, sum, length);
//   }
//   std::cout << "sdds332"
//             << "\n";

//   // 5.multinominal
//   for (int i = 0; i < 15; i++) {
//     parallelism = length;
//     DT random = static_cast<DT>(((float)std::rand()) / RAND_MAX);
//     thrust::inclusive_scan(thrust::device,
//                            input_ptr + i * length,
//                            input_ptr + (i + 1) * length,
//                            static_cast<DT *>(m->cumsum_ptr) + i * length);

//     // find_idx<DT><<<GET_BLOCKS(parallelism),
//     //                min(CUDA_NUM_THREADS, parallelism),
//     //                0,
//     //                stream>>>(static_cast<DT *>(m->cumsum_ptr) + i * length,
//     //                          static_cast<DT *>(m->origin_ptr) + i * length,
//     //                          random,
//     //                          length,
//     //                          indices_ptr,
//     //                          i);
//     for (int j = 0; j < length; j++) {
//       if ((static_cast<DT *>(m->cumsum_ptr) + i * length)[j] >= random) {
//         indices_ptr[i] = (static_cast<DT *>(m->origin_ptr) + i * length)[i];
//         printf("k value is:%d. %f\n", i, indices_ptr[i]);
//         break;
//       }
//     }
//   }
//   // print_tensor<int>((int *)indices_ptr, 15, "sdsdasd");
//   assert(false);
// }

// /*static*/
// void Sampling::forward_kernel_wrapper(SamplingMeta const *m,
//                                       GenericTensorAccessorW const &input,
//                                       GenericTensorAccessorW const &indices) {
//   cudaStream_t stream;
//   checkCUDA(get_legion_stream(&stream));

//   cudaEvent_t t_start, t_end;
//   if (m->profiling) {
//     cudaEventCreate(&t_start);
//     cudaEventCreate(&t_end);
//     cudaEventRecord(t_start, stream);
//   }
//   int length = input.domain.hi()[0] - input.domain.lo()[0] + 1;
//   int batch_size = input.domain.get_volume() / length;

//   if (input.data_type == DT_HALF) {
//     Sampling::forward_kernel<half>(m,
//                                    input.get_half_ptr(),
//                                    indices.get_int32_ptr(),
//                                    m->top_p,
//                                    length,
//                                    batch_size,
//                                    stream);
//   } else if (input.data_type == DT_FLOAT) {
//     Sampling::forward_kernel<float>(m,
//                                     input.get_float_ptr(),
//                                     indices.get_int32_ptr(),
//                                     m->top_p,
//                                     length,
//                                     batch_size,
//                                     stream);
//   } else {
//     assert(false && "Unsupported data type");
//   }

//   if (m->profiling) {
//     cudaEventRecord(t_end, stream);
//     checkCUDA(cudaEventSynchronize(t_end));
//     float elapsed = 0;
//     checkCUDA(cudaEventElapsedTime(&elapsed, t_start, t_end));
//     cudaEventDestroy(t_start);
//     cudaEventDestroy(t_end);
//     printf("[Sampling] forward time = %.2lfms\n", elapsed);
//   }
// }

// SamplingMeta::SamplingMeta(FFHandler handler, Op const *op)
//     : OpMeta(handler, op) {
//   checkCUDA(cudaMalloc(&cumsum_ptr, 15 * 32000 * sizeof(float)));
//   checkCUDA(cudaMalloc(&sampled, 15 * 32000 * sizeof(float)));
// }

// }; // namespace FlexFlow