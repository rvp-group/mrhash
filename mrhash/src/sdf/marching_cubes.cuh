#pragma once
#include "mesh_extractor.cuh"

namespace cupanutils {
  namespace cugeoutils {

    template <typename T>
    class MarchingCubesExtractor : public MeshExtractor<T> {
    public:
      MarchingCubesExtractor(const float marching_cubes_threshold,
                             const bool viewer_active,
                             const uint max_num_triangles_mesh,
                             const float vertices_merging_threshold) :
        marching_cubes_threshold_(marching_cubes_threshold),
        MeshExtractor<T>(viewer_active, max_num_triangles_mesh, vertices_merging_threshold) {
        CUDA_CHECK(cudaMalloc((void**) &d_instance_, sizeof(MarchingCubesExtractor)));
        CUDA_CHECK(cudaMemcpy(d_instance_, this, sizeof(MarchingCubesExtractor), cudaMemcpyHostToDevice));
      }

      ~MarchingCubesExtractor() {
        CUDA_CHECK(cudaFree(d_instance_));
      }

      __device__ void extractIsoSurfaceAtPosition(VoxelContainer<T>* container,
                                                  const float3& pw,
                                                  const float3& positive_scaling,
                                                  const float3& negative_scaling);

      __host__ void extractIsoSurface(VoxelContainer<T>& container) override;

    private:
      /**
       * @brief Checks if the eight corners of a voxel position are at different resolutions.
       *
       * @param pf The position of the voxel.
       * @param positive_scaling The scaling factor for the positive direction.
       * @param negative_scaling The scaling factor for the negative direction.
       * @return true if the eight corners are at different resolutions, false otherwise.
       *
       * @note This function is used to check if the eight corners of a voxel's position are at different resolutions.
       */
      __device__ bool
      checkVertexVoxels(const VoxelContainer<T>* container, const float3& pf, float3& positive_scaling, float3& negative_scaling);

      float marching_cubes_threshold_     = 0.f;
      MarchingCubesExtractor* d_instance_ = nullptr;
    };

    template class MarchingCubesExtractor<Voxel>;
    using GeometricMarchingCubes = MarchingCubesExtractor<Voxel>;
  } // namespace cugeoutils
} // namespace cupanutils
