#pragma once
#include "cuda_utils.cuh"
#include <type_traits>

namespace cupanutils {
  namespace cugeoutils {

    struct Voxel {
      __host__ __device__ Voxel() {
        sdf    = 0.f;
        rgb    = make_uchar3(0, 0, 0);
        weight = 0;
      }
      float sdf         = 0.f; // signed distance function
      float sum_squared = 0.f;
      uchar3 rgb        = make_uchar3(0, 0, 0); // color
      uchar weight      = 0;                    // accumulated sdf weight

      auto cista_members() const {
        return std::tie(sdf, sum_squared, rgb, weight);
      }
    };

    template <typename T>
    struct is_voxel_derived : std::is_base_of<Voxel, T> {};

    struct HashEntry {
      __host__ __device__ HashEntry() {
        pos        = make_int3(0, 0, 0);
        offset     = NO_OFFSET;
        ptr        = FREE_ENTRY;
        resolution = 0;
      }
      int3 pos;    // hash position (lower left corner of SDFBlock))
      uint offset; // offset for collisions
      int ptr;     // pointer into heap to SDFBlock
      int resolution;
    };

    struct RayCastSample {
      float sdf;
      float alpha;
      uint weight;
    };

    struct Vertex {
      __host__ __device__ Vertex() {
        p = make_float3(0.f, 0.f, 0.f);
        c = make_float3(0.f, 0.f, 0.f);
      }
      float3 p;
      float3 c;
    };

    struct Triangle {
      __host__ __device__ Triangle() {
        v0 = Vertex();
        v1 = Vertex();
        v2 = Vertex();
      }
      Vertex v0;
      Vertex v1;
      Vertex v2;
    };

    __forceinline__ __device__ float3 virtualVoxelPosToWorld(const float virtual_voxel_size, const int3& voxel_pos) {
      return make_float3(voxel_pos.x * virtual_voxel_size, voxel_pos.y * virtual_voxel_size, voxel_pos.z * virtual_voxel_size);
    }

    __forceinline__ __device__ float3 virtualVoxelPosToWorld(const float virtual_voxel_size, const float3& voxel_pos) {
      return make_float3(voxel_pos.x * virtual_voxel_size, voxel_pos.y * virtual_voxel_size, voxel_pos.z * virtual_voxel_size);
    }

#ifdef __CUDACC__
    __forceinline__ __device__ int3 virtualVoxelPosToSDFBlock(const int3 virtual_voxel_pos,
                                                              const float virtual_voxel_size,
                                                              const float3 voxel_extents,
                                                              const int block_size = sdf_block_size) {
      int3 voxel_pos = virtual_voxel_pos;
      float epsilon  = 1e-5;
      if (virtual_voxel_pos.x < 0)
        voxel_pos.x -= (block_size - 1); // i.e voxelBlock virtual_voxel_size -1
      if (virtual_voxel_pos.y < 0)
        voxel_pos.y -= (block_size - 1);
      if (virtual_voxel_pos.z < 0)
        voxel_pos.z -= (block_size - 1);

      const float3 pw = virtualVoxelPosToWorld(virtual_voxel_size, voxel_pos);

      const float3 metric_block_size = make_float3((voxel_extents.x * sdf_block_size * virtual_voxel_size),
                                                   (voxel_extents.y * sdf_block_size * virtual_voxel_size),
                                                   (voxel_extents.z * sdf_block_size * virtual_voxel_size));

      int3 sdf_block_pos;
      sdf_block_pos.x =
        (pw.x >= 0) ? floorf((pw.x + epsilon) / metric_block_size.x) : ceilf((pw.x - epsilon) / metric_block_size.x);
      sdf_block_pos.y =
        (pw.y >= 0) ? floorf((pw.y + epsilon) / metric_block_size.y) : ceilf((pw.y - epsilon) / metric_block_size.y);
      sdf_block_pos.z =
        (pw.z >= 0) ? floorf((pw.z + epsilon) / metric_block_size.z) : ceilf((pw.z - epsilon) / metric_block_size.z);

      return sdf_block_pos;
    }
#endif

    __forceinline__ __host__ __device__ uint linearizeVoxelPos(const int3& pos, const int block_size = sdf_block_size) {
      return pos.z * block_size * block_size + pos.y * block_size + pos.x;
    }

    __forceinline__ __host__ __device__ uint virtualVoxelPosToSDFBlockIndex(const int3& virtual_voxel_pos,
                                                                            const int block_size = sdf_block_size) {
      const int scaling_factor = sdf_block_size / block_size;
      int3 local_voxel_pos     = make_int3(
        virtual_voxel_pos.x % sdf_block_size, virtual_voxel_pos.y % sdf_block_size, virtual_voxel_pos.z % sdf_block_size);

      if (local_voxel_pos.x < 0)
        local_voxel_pos.x += sdf_block_size;
      if (local_voxel_pos.y < 0)
        local_voxel_pos.y += sdf_block_size;
      if (local_voxel_pos.z < 0)
        local_voxel_pos.z += sdf_block_size;

      local_voxel_pos.x /= scaling_factor;
      local_voxel_pos.y /= scaling_factor;
      local_voxel_pos.z /= scaling_factor;

      return linearizeVoxelPos(local_voxel_pos, sdf_block_size);
    }

    __forceinline__ __host__ __device__ uint3 delinearizeVoxelPos(const uint index, const int block_size = sdf_block_size) {
      const uint size2 = block_size * block_size;
      const uint x     = (index % block_size);
      const uint y     = ((index % (size2)) / block_size);
      const uint z     = (index / (size2));
      return make_uint3(x, y, z);
    }

    __forceinline__ __host__ __device__ int3 SDFBlockToVirtualVoxelPos(const int3& sdf_block) {
      return make_int3(sdf_block.x * sdf_block_size, sdf_block.y * sdf_block_size, sdf_block.z * sdf_block_size);
    }

#ifdef __CUDACC__
    __forceinline__ __host__ __device__ int3 worldPointToVirtualVoxelPos(const float virtual_voxel_size, const float3& point) {
      float3 p       = point / virtual_voxel_size;
      float epsilon  = 1e-5;
      float3 aprox_p = p + make_float3(sign(p)) * 0.5f;
      aprox_p.x      = (aprox_p.x >= 0) ? floorf(aprox_p.x + epsilon) : ceilf(aprox_p.x - epsilon);
      aprox_p.y      = (aprox_p.y >= 0) ? floorf(aprox_p.y + epsilon) : ceilf(aprox_p.y - epsilon);
      aprox_p.z      = (aprox_p.z >= 0) ? floorf(aprox_p.z + epsilon) : ceilf(aprox_p.z - epsilon);
      return make_int3(aprox_p);
    }

    __forceinline__ __device__ float3 worldPointToVirtualVoxelPosFloat(const float virtual_voxel_size, const float3& point) {
      return point / virtual_voxel_size;
    }

    __forceinline__ __device__ int3 worldPointToSDFBlock(const float virtual_voxel_size,
                                                         const float3 voxel_extents,
                                                         const float3& point) {
      return virtualVoxelPosToSDFBlock(worldPointToVirtualVoxelPos(virtual_voxel_size, point), virtual_voxel_size, voxel_extents);
    }

    __forceinline__ __device__ float3 SDFBlockToWorldPoint(const float virtual_voxel_size, const int3& sdf_block) {
      return virtualVoxelPosToWorld(virtual_voxel_size, SDFBlockToVirtualVoxelPos(sdf_block));
    }

    template <typename T>
    // ! merges two voxels (v0 the currently stored voxel, v1 is the input voxel)
    __forceinline__ __device__ void combineVoxel(const T& v0, const T& v1, const int integration_weight_max, T& out) {
      // interpolate colors
      const float3 c0  = make_float3(v0.rgb.x, v0.rgb.y, v0.rgb.z);
      const float3 c1  = make_float3(v1.rgb.x, v1.rgb.y, v1.rgb.z);
      const float3 res = 0.5f * c0 + 0.5f * c1;
      out.rgb.x        = (uchar) (res.x + 0.5f);
      out.rgb.y        = (uchar) (res.y + 0.5f);
      out.rgb.z        = (uchar) (res.z + 0.5f);

      // merge sdf and weight
      out.sdf    = (v0.sdf * v0.weight + v1.sdf * v1.weight) / (v0.weight + v1.weight);
      out.weight = min(integration_weight_max, v0.weight + v1.weight);
    }

    //! returns the truncation of the SDF for a given distance value
    __forceinline__ __host__ __device__ float
    getTruncation(const float z, const float sdf_truncation, const float sdf_truncation_scale) {
      return sdf_truncation + sdf_truncation_scale * z;
    }

    // ! delete stuff, voxel and hash entries
    __forceinline__ __device__ void deleteHashEntry(HashEntry& hashEntry) {
      hashEntry.pos    = make_int3(0, 0, 0);
      hashEntry.offset = NO_OFFSET;
      hashEntry.ptr    = FREE_ENTRY;
    }

    template <typename T>
    __forceinline__ __device__ void deleteVoxel(T& v) {
      v.sdf         = 0.f;
      v.rgb         = make_uchar3(0, 0, 0);
      v.sum_squared = 0.f;
      v.weight      = 0;
    }

    __forceinline__ __device__ uint linearizeChunkPos(const int3& chunk_pos,
                                                      const int3& min_grid_pos,
                                                      const int3& grid_dimensions) {
      const int3 p = chunk_pos - min_grid_pos;
      return p.z * grid_dimensions.x * grid_dimensions.y + p.y * grid_dimensions.x + p.x;
    }

    __forceinline__ __device__ int3 worldToChunks(const float3& pw, const float3& voxel_extents) {
      float3 p;
      p.x = pw.x / voxel_extents.x;
      p.y = pw.y / voxel_extents.y;
      p.z = pw.z / voxel_extents.z;

      float3 s;
      s.x = (float) sign(p.x);
      s.y = (float) sign(p.y);
      s.z = (float) sign(p.z);

      return make_int3(p + s * 0.5f);
    }

#endif

  } // namespace cugeoutils
} // namespace cupanutils
