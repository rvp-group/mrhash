#include "marching_cubes.cuh"

namespace cupanutils {
  namespace cugeoutils {

    template <typename T>
    __device__ bool MarchingCubesExtractor<T>::checkVertexVoxels(const VoxelContainer<T>* container,
                                                                 const float3& pf,
                                                                 float3& scaled_P,
                                                                 float3& scaled_M) {
      const float virtual_voxel_size = container->getVoxelSize(pf);
      bool modified                  = false;

      {
        float3 p = pf + make_float3(scaled_P.x, 0.0f, 0.0f);
        float vs = container->getVoxelSize(p);
        if (vs > 0 && vs < 1 && vs != virtual_voxel_size) {
          scaled_P.x *= 0.499f;
          modified = true;
        }
      }

      {
        float3 p = pf + make_float3(scaled_M.x, 0.0f, 0.0f);
        float vs = container->getVoxelSize(p);
        if (vs > 0 && vs < 1 && vs != virtual_voxel_size) {
          scaled_M.x *= 0.499f;
          modified = true;
        }
      }

      {
        float3 p = pf + make_float3(0.0f, scaled_P.y, 0.0f);
        float vs = container->getVoxelSize(p);
        if (vs > 0 && vs < 1 && vs != virtual_voxel_size) {
          scaled_P.y *= 0.499f;
          modified = true;
        }
      }

      {
        float3 p = pf + make_float3(0.0f, scaled_M.y, 0.0f);
        float vs = container->getVoxelSize(p);
        if (vs > 0 && vs < 1 && vs != virtual_voxel_size) {
          scaled_M.y *= 0.499f;
          modified = true;
        }
      }

      {
        float3 p = pf + make_float3(0.0f, 0.0f, scaled_P.z);
        float vs = container->getVoxelSize(p);
        if (vs > 0 && vs < 1 && vs != virtual_voxel_size) {
          scaled_P.z *= 0.499f;
          modified = true;
        }
      }

      {
        float3 p = pf + make_float3(0.0f, 0.0f, scaled_M.z);
        float vs = container->getVoxelSize(p);
        if (vs > 0 && vs < 1 && vs != virtual_voxel_size) {
          scaled_M.z *= 0.499f;
          modified = true;
        }
      }

      return modified;
    }

    template <typename T>
    __device__ void MarchingCubesExtractor<T>::extractIsoSurfaceAtPosition(VoxelContainer<T>* container,
                                                                           const float3& pf,
                                                                           const float3& positive_scaling,
                                                                           const float3& negative_scaling) {
      const float isolevel           = 0.f;
      const float virtual_voxel_size = container->getVoxelSize(pf);
      const float P                  = virtual_voxel_size * 0.5f;
      const float M                  = -P;
      float3 scaled_P                = P * positive_scaling;
      float3 scaled_M                = M * negative_scaling;

      bool modified = checkVertexVoxels(container, pf, scaled_P, scaled_M);

      float3 p000 = pf + make_float3(scaled_M.x, scaled_M.y, scaled_M.z);
      float dist000;
      bool valid000 = container->trilinearInterpolation(p000, dist000);
      const T& v000 = container->getVoxel(p000);
      if (!valid000) {
        if (v000.weight < container->min_weight_threshold_)
          return;
        dist000 = v000.sdf;
      }

      float3 p001 = pf + make_float3(scaled_P.x, scaled_M.y, scaled_M.z);
      float dist001;
      bool valid001 = container->trilinearInterpolation(p001, dist001);
      const T& v001 = container->getVoxel(p001);
      if (!valid001) {
        if (v001.weight < container->min_weight_threshold_)
          return;
        dist001 = v001.sdf;
      }
      float3 p010 = pf + make_float3(scaled_M.x, scaled_P.y, scaled_M.z);
      float dist010;
      bool valid010 = container->trilinearInterpolation(p010, dist010);
      const T& v010 = container->getVoxel(p010);
      if (!valid010) {
        if (v010.weight < container->min_weight_threshold_)
          return;
        dist010 = v010.sdf;
      }
      float3 p011 = pf + make_float3(scaled_P.x, scaled_P.y, scaled_M.z);
      float dist011;
      bool valid011 = container->trilinearInterpolation(p011, dist011);
      const T& v011 = container->getVoxel(p011);
      if (!valid011) {
        if (v011.weight < container->min_weight_threshold_)
          return;
        dist011 = v011.sdf;
      }
      float3 p100 = pf + make_float3(scaled_M.x, scaled_M.y, scaled_P.z);
      float dist100;
      bool valid100 = container->trilinearInterpolation(p100, dist100);
      const T& v100 = container->getVoxel(p100);
      if (!valid100) {
        if (v100.weight < container->min_weight_threshold_)
          return;
        dist100 = v100.sdf;
      }
      float3 p101 = pf + make_float3(scaled_P.x, scaled_M.y, scaled_P.z);
      float dist101;
      bool valid101 = container->trilinearInterpolation(p101, dist101);
      const T& v101 = container->getVoxel(p101);
      if (!valid101) {
        if (v101.weight < container->min_weight_threshold_)
          return;
        dist101 = v101.sdf;
      }
      float3 p110 = pf + make_float3(scaled_M.x, scaled_P.y, scaled_P.z);
      float dist110;
      bool valid110 = container->trilinearInterpolation(p110, dist110);
      const T& v110 = container->getVoxel(p110);
      if (!valid110) {
        if (v110.weight < container->min_weight_threshold_)
          return;
        dist110 = v110.sdf;
      }
      float3 p111 = pf + make_float3(scaled_P.x, scaled_P.y, scaled_P.z);
      float dist111;
      bool valid111 = container->trilinearInterpolation(p111, dist111);
      const T& v111 = container->getVoxel(p111);
      if (!valid111) {
        if (v111.weight < container->min_weight_threshold_)
          return;
        dist111 = v111.sdf;
      }

      // if (!valid000 || !valid100 || !valid010 || !valid110 || !valid001 || !valid101 || !valid011 || !valid111) {
      //   return;
      // }

      uint cube_index = 0;
      if (dist000 < isolevel)
        cube_index += 1;
      if (dist001 < isolevel)
        cube_index += 2;
      if (dist010 < isolevel)
        cube_index += 4;
      if (dist011 < isolevel)
        cube_index += 8;
      if (dist100 < isolevel)
        cube_index += 16;
      if (dist101 < isolevel)
        cube_index += 32;
      if (dist110 < isolevel)
        cube_index += 64;
      if (dist111 < isolevel)
        cube_index += 128;

      const float marching_cubes_threshold = marching_cubes_threshold_;
      float dist_array[]                   = {dist000, dist100, dist010, dist110, dist001, dist101, dist011, dist111};
      for (uint k = 0; k < 8; k++) {
        for (uint l = 0; l < 8; l++) {
          if (dist_array[k] * dist_array[l] < 0.f) {
            if (abs(dist_array[k]) + abs(dist_array[l]) > marching_cubes_threshold) {
              return;
            }
          } else {
            if (abs(dist_array[k] - dist_array[l]) > marching_cubes_threshold) {
              return;
            }
          }
        }
      }
      if (abs(dist000) > marching_cubes_threshold || abs(dist100) > marching_cubes_threshold ||
          abs(dist010) > marching_cubes_threshold || abs(dist110) > marching_cubes_threshold ||
          abs(dist001) > marching_cubes_threshold || abs(dist101) > marching_cubes_threshold ||
          abs(dist011) > marching_cubes_threshold || abs(dist111) > marching_cubes_threshold) {
        return;
      }

      const RegularCellData& triangulation = regularCellData[regularCellClass[cube_index]];
      const int num_triangles              = triangulation.getTriangleCount();
      const int num_vertex                 = triangulation.getVertexCount();

      if (num_triangles == 0)
        return;

      Vertex vertex_list[12];

      const unsigned short* edge_flags = regularVertexData[cube_index];

      for (int i = 0; i < num_vertex; i++) {
        const unsigned short vertex_idx = edge_flags[i] & 0xFF;
        if (vertex_idx == 0x01) {
          vertex_list[i] = this->vertexInterp(isolevel, p000, p001, dist000, dist001, v000.rgb, v001.rgb);
        }
        if (vertex_idx == 0x02) {
          vertex_list[i] = this->vertexInterp(isolevel, p000, p010, dist000, dist010, v000.rgb, v010.rgb);
        }
        if (vertex_idx == 0x04) {
          vertex_list[i] = this->vertexInterp(isolevel, p000, p100, dist000, dist100, v000.rgb, v100.rgb);
        }
        if (vertex_idx == 0x13) {
          vertex_list[i] = this->vertexInterp(isolevel, p001, p011, dist001, dist011, v001.rgb, v011.rgb);
        }
        if (vertex_idx == 0x15) {
          vertex_list[i] = this->vertexInterp(isolevel, p001, p101, dist001, dist101, v001.rgb, v101.rgb);
        }
        if (vertex_idx == 0x23) {
          vertex_list[i] = this->vertexInterp(isolevel, p010, p011, dist010, dist011, v010.rgb, v011.rgb);
        }
        if (vertex_idx == 0x26) {
          vertex_list[i] = this->vertexInterp(isolevel, p010, p110, dist010, dist110, v010.rgb, v110.rgb);
        }
        if (vertex_idx == 0x37) {
          vertex_list[i] = this->vertexInterp(isolevel, p011, p111, dist011, dist111, v011.rgb, v111.rgb);
        }
        if (vertex_idx == 0x45) {
          vertex_list[i] = this->vertexInterp(isolevel, p100, p101, dist100, dist101, v100.rgb, v101.rgb);
        }
        if (vertex_idx == 0x46) {
          vertex_list[i] = this->vertexInterp(isolevel, p100, p110, dist100, dist110, v100.rgb, v110.rgb);
        }
        if (vertex_idx == 0x57) {
          vertex_list[i] = this->vertexInterp(isolevel, p101, p111, dist101, dist111, v101.rgb, v111.rgb);
        }
        if (vertex_idx == 0x67) {
          vertex_list[i] = this->vertexInterp(isolevel, p110, p111, dist110, dist111, v110.rgb, v111.rgb);
        }
      }

      for (int i = 0; i < num_triangles; i++) {
        Triangle t;
        t.v0 = vertex_list[triangulation.vertexIndex[3 * i]];
        t.v1 = vertex_list[triangulation.vertexIndex[3 * i + 1]];
        t.v2 = vertex_list[triangulation.vertexIndex[3 * i + 2]];
        this->appendTriangle(t);
      }
    }

    template <typename T>
    __global__ void extractIsoSurfaceKernel(MarchingCubesExtractor<T>* mesh_extractor, VoxelContainer<T>* container) {
      const uint entry_idx   = blockIdx.x * blockDim.x + threadIdx.x;
      const HashEntry& entry = container->d_compactHashTable_[entry_idx];
      if (entry.ptr == FREE_ENTRY)
        return;
      const int num_voxels   = container->getNumVoxels(entry);
      const uint voxel_index = blockIdx.y * blockDim.y + threadIdx.y;
      if (voxel_index >= num_voxels)
        return;

      const int scaling_factor = 1 << entry.resolution;

      const int3 pi_base      = SDFBlockToVirtualVoxelPos(entry.pos);
      const int3 voxel_coords = scaling_factor * make_int3(delinearizeVoxelPos(voxel_index, sdf_block_size / scaling_factor));
      const int3 pi           = pi_base + voxel_coords;
      const float3 pf         = virtualVoxelPosToWorld(container->virtual_voxel_size_, pi);

      float3 positive_scaling = make_float3(1, 1, 1);
      float3 negative_scaling = make_float3(1, 1, 1);

      mesh_extractor->extractIsoSurfaceAtPosition(container, pf, positive_scaling, negative_scaling);
    }

    template <typename T>
    void MarchingCubesExtractor<T>::extractIsoSurface(VoxelContainer<T>& container) {
      if (container.current_occupied_blocks_ > 0) {
        const dim3 threads_per_block(n_threads, n_threads, 1);
        const dim3 n_blocks((container.current_occupied_blocks_ + threads_per_block.x - 1) / threads_per_block.x,
                            (container.voxel_block_volume_ + threads_per_block.y - 1) / threads_per_block.y,
                            1);
        extractIsoSurfaceKernel<<<n_blocks, threads_per_block>>>(d_instance_, container.d_instance_);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(&this->num_triangles_, &this->d_trianglesCounter_[0], sizeof(uint), cudaMemcpyDeviceToHost));
        if (this->num_triangles_ > 0) {
          CUDA_CHECK(cudaMemcpy(
            &this->h_triangles_[0], &this->d_triangles_[0], this->num_triangles_ * sizeof(Triangle), cudaMemcpyDeviceToHost));
        }
        CUDA_CHECK(cudaMemset(&this->d_trianglesCounter_[0], 0, sizeof(uint)));
        CUDA_CHECK(cudaMemset(&this->d_triangles_[0], 0, this->max_num_triangles_mesh_ * sizeof(Triangle)));
      }
      std::cout << "MarchingCubesExtractor::extractIsoSurface | triangles extracted: " << this->num_triangles_ << std::endl;
    }
  } // namespace cugeoutils
} // namespace cupanutils
