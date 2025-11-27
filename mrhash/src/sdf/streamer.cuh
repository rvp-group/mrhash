#pragma once

#include "utils/cista.h"
#include "voxel_data_structures.cuh"

namespace data = cista::offset;

namespace cupanutils {
  namespace cugeoutils {

    struct Vector3iHash {
      std::size_t operator()(const Eigen::Vector3i& v) const noexcept {
        size_t h1 = std::hash<int>()(v.x());
        size_t h2 = std::hash<int>()(v.y());
        size_t h3 = std::hash<int>()(v.z());
        return h1 ^ (h2 << 1) ^ (h3 << 2); // Combine hashes
      }
    };

    template <typename T>
    struct SDFBlock {
      data::vector<T> data;

      SDFBlock(const int num_voxels) {
        data.reserve(num_voxels);
      }

      static uint3 delinearizeVoxelIndex(uint idx, const int block_size = sdf_block_size) {
        uint x = idx % block_size;
        uint y = (idx % (block_size * block_size)) / block_size;
        uint z = idx / (block_size * block_size);
        return make_uint3(x, y, z);
      }

      auto cista_members() {
        return std::tie(data);
      }
    };

    class SDFBlockDesc {
    public:
      __host__ __device__ SDFBlockDesc() : pos(make_int3(0, 0, 0)), ptr(-1), resolution(0) {
      }

      __host__ __device__ SDFBlockDesc(const HashEntry& entry) {
        pos        = entry.pos;
        ptr        = entry.ptr;
        resolution = entry.resolution;
      }

      bool operator<(const SDFBlockDesc& other) const {
        if (pos.x == other.pos.x) {
          if (pos.y == other.pos.y) {
            return pos.z < other.pos.z;
          }
          return pos.y < other.pos.y;
        }
        return pos.x < other.pos.x;
      }

      bool operator==(const SDFBlockDesc& other) const {
        return pos.x == other.pos.x && pos.y == other.pos.y && pos.z == other.pos.z;
      }

      struct HashSDFBlockDesc {
        size_t operator()(const SDFBlockDesc& other) const {
          const int3& v    = other.pos;
          const size_t res = ((size_t) v.x * p0) ^ ((size_t) v.y * p1) ^ ((size_t) v.z * p2);
          return res;
        }
      };

      auto cista_members() {
        return std::tie(pos, ptr, resolution);
      }

      int3 pos;
      int ptr;
      int resolution;
    } __align__(16);

    template <typename T>
    class ChunkDesc {
    public:
      ChunkDesc(const uint& initial_chunk_list_size) {
        vecSDFBlock_ = data::vector<SDFBlock<T>>();
        vecSDFBlock_.reserve(initial_chunk_list_size);
        vecChunkDesc_ = data::vector<SDFBlockDesc>();
        vecChunkDesc_.reserve(initial_chunk_list_size);
      }

      ChunkDesc(const ChunkDesc& chunk_desc) {
        vecSDFBlock_  = chunk_desc.vecSDFBlock_;
        vecChunkDesc_ = chunk_desc.vecChunkDesc_;
      }

      // Copy assignment operator
      ChunkDesc& operator=(const ChunkDesc& chunk_desc) {
        if (this != &chunk_desc) {
          vecSDFBlock_  = chunk_desc.vecSDFBlock_;
          vecChunkDesc_ = chunk_desc.vecChunkDesc_;
        }
        return *this;
      }

      // Move constructor
      ChunkDesc(ChunkDesc&& chunk_desc) noexcept {
        vecSDFBlock_  = std::move(chunk_desc.vecSDFBlock_);
        vecChunkDesc_ = std::move(chunk_desc.vecChunkDesc_);
      }

      // Move assignment operator
      ChunkDesc& operator=(ChunkDesc&& chunk_desc) noexcept {
        if (this != &chunk_desc) {
          vecSDFBlock_  = std::move(chunk_desc.vecSDFBlock_);
          vecChunkDesc_ = std::move(chunk_desc.vecChunkDesc_);
        }
        return *this;
      }

      // Destructor (default is fine, but explicit for completeness)
      ~ChunkDesc() = default;

      void addSDFBlock(const SDFBlockDesc& desc, const SDFBlock<T>& data) {
        vecChunkDesc_.push_back(desc);
        vecSDFBlock_.push_back(data);
      }

      uint getNElements() const {
        return (uint) vecSDFBlock_.size();
      }

      const SDFBlockDesc& getSDFBlockDesc(uint i) const {
        return vecChunkDesc_[i];
      }

      const SDFBlock<T>& getSDFBlock(uint i) const {
        return vecSDFBlock_[i];
      }

      void clear() {
        vecChunkDesc_.clear();
        vecSDFBlock_.clear();
      }

      bool isStreamedOut() const {
        return vecSDFBlock_.size() > 0;
      }

      const data::vector<SDFBlockDesc>& getSDFBlockDescs() const {
        return vecChunkDesc_;
      }

      const data::vector<SDFBlock<T>>& getSDFBlocks() const {
        return vecSDFBlock_;
      }

      auto cista_members() {
        return std::tie(vecSDFBlock_, vecChunkDesc_);
      }

      data::vector<SDFBlock<T>> vecSDFBlock_;
      data::vector<SDFBlockDesc> vecChunkDesc_;
    };

    template <typename T, typename Enable = void>
    class Streamer {
      Streamer() {
        std::cerr << "Streamer is not implemented for this type" << std::endl;
      }
    };

    template <typename T>
    class Streamer<T, std::enable_if_t<is_voxel_derived<T>::value>> {
    public:
      Streamer() {
      }
      Streamer(VoxelContainer<T>* c,
               const bool write_timings,
               const std::string& memory_allocation_filepath,
               const std::string& profiler_name) :
        container_(c),
        memory_allocation_filepath_(memory_allocation_filepath),
        streaming_profiler_(profiler_name, write_timings) {
      }

      ~Streamer() {
        destroy();
      }

      // Copy constructor - deleted because of raw CUDA pointers
      // Deep copying CUDA memory requires explicit handling
      Streamer(const Streamer&) = delete;

      // Copy assignment operator - deleted
      Streamer& operator=(const Streamer&) = delete;

      //  debugging
      void calculateMemoryUsage() const;
      double debugCheckForDuplicates() const;

      void create(const Eigen::Vector3f& voxel_extents,
                  const uint max_num_sdf_block_integrate_from_global_hash,
                  unsigned int initial_chunk_list_size);

      void clearGrid();
      void destroy();

      // ! stream out - in, wrapper for both
      void stream(const Eigen::Vector3f& camera_position, const float radius);

      // ! stream out - to host
      void streamOutToHostPass0(const float3& camera_pose, const float radius);
      void integrateFromGlobalHashPass1(const float radius, const float3& camera_position);
      void integrateFromGlobalHashPass2(const uint num_SDF_block_desc);

      void streamAllOut(); // usually at the end
      void integrateFromGlobalHashPass1(const int num_pass);
      void integrateFromGlobalHashPass2(const uint num_SDF_block_desc, const int num_pass);

      void streamOutToCPUPass1CPU();
      void integrateInChunkGrid(const SDFBlockDesc* h_SDFBlockDescOutput_, const T* h_SDFBlockOutput_);

      // ! stream in - to device
      uint integrateInHash(const Eigen::Vector3f& camera_pose, float radius);
      void streamInToGPU(const Eigen::Vector3f& camera_position, const float radius);
      void streamInToGPUChunk(const Eigen::Vector3i& chunk);
      void streamInToGPUChunkNeighborhood(const Eigen::Vector3i& chunk, const int radius);
      void chunkToGlobalHashPass1(const uint num_sdf_blocks_descs,
                                  const uint heap_count_prev,
                                  const SDFBlockDesc* d_SDFBlockDescs,
                                  uint* d_blocks_ptr);
      void chunkToGlobalHashPass2(const uint num_sdf_blocks_descs,
                                  const uint heap_count_prev,
                                  const SDFBlockDesc* d_SDFBlockDescs,
                                  const T* d_SDFBlocks,
                                  uint* d_blocks_ptr);

      // ! some other utils
      inline void clearSDFBlockCounter() {
        uint src = 0;
        CUDA_CHECK(cudaMemcpy(d_SDF_block_counter_, &src, sizeof(uint), cudaMemcpyHostToDevice));
      }

      inline uint getSDFBlockCounter() const {
        uint dest;
        CUDA_CHECK(cudaMemcpy(&dest, d_SDF_block_counter_, sizeof(uint), cudaMemcpyDeviceToHost));
        return dest;
      }

      inline Eigen::Vector3i worldToChunks(const Eigen::Vector3f& pw) const {
        Eigen::Vector3f p;

        p.x() = pw.x() / voxel_extents_.x();
        p.y() = pw.y() / voxel_extents_.y();
        p.z() = pw.z() / voxel_extents_.z();

        Eigen::Vector3f s = (Eigen::Vector3f) p.array().sign();
        return (p + s * 0.5f).cast<int>();
      }

      inline Eigen::Vector3i worldPointToVirtualVoxelPos(const Eigen::Vector3f& pw) {
        const Eigen::Vector3f p = pw / container_->virtual_voxel_size_;

        const Eigen::Vector3f s = (Eigen::Vector3f) p.array().sign();
        return (p + s * 0.5f).cast<int>();
      }

      inline Eigen::Vector3i worldPointToSDFBlockPos(const Eigen::Vector3f& pw) {
        return virtualVoxelPosToSDFBlock(worldPointToVirtualVoxelPos(pw));
      }

      inline uint linearizeVoxelPos(const Eigen::Vector3i& pos) {
        return pos.z() * sdf_block_size * sdf_block_size + pos.y() * sdf_block_size + pos.x();
      }

      inline uint worldPointToSDFBlockIndex(const Eigen::Vector3f& pw, const int block_size = sdf_block_size) {
        Eigen::Vector3i virtual_voxel_pos = worldPointToVirtualVoxelPos(pw);
        Eigen::Vector3i local_voxel_pos;
        local_voxel_pos.x() = virtual_voxel_pos.x() % block_size;
        local_voxel_pos.y() = virtual_voxel_pos.y() % block_size;
        local_voxel_pos.z() = virtual_voxel_pos.z() % block_size;
        if (local_voxel_pos.x() < 0)
          local_voxel_pos.x() += block_size;
        if (local_voxel_pos.y() < 0)
          local_voxel_pos.y() += block_size;
        if (local_voxel_pos.z() < 0)
          local_voxel_pos.z() += block_size;

        return linearizeVoxelPos(local_voxel_pos);
      }

      inline Eigen::Vector3i virtualVoxelPosToSDFBlock(Eigen::Vector3i virtual_voxel_pos) {
        if (virtual_voxel_pos.x() < 0)
          virtual_voxel_pos.x() -= sdf_block_size - 1; // i.e voxelBlock virtual_voxel_size -1
        if (virtual_voxel_pos.y() < 0)
          virtual_voxel_pos.y() -= sdf_block_size - 1;
        if (virtual_voxel_pos.z() < 0)
          virtual_voxel_pos.z() -= sdf_block_size - 1;
        return Eigen::Vector3i(
          virtual_voxel_pos.x() / sdf_block_size, virtual_voxel_pos.y() / sdf_block_size, virtual_voxel_pos.z() / sdf_block_size);
      }

      inline Eigen::Vector3f chunkToWorld(const Eigen::Vector3i& chunk_pose) const {
        Eigen::Vector3f res;
        res << chunk_pose.x() * voxel_extents_.x(), chunk_pose.y() * voxel_extents_.y(), chunk_pose.z() * voxel_extents_.z();
        return res;
      }

      inline const Eigen::Vector3i meterToNumberOfChunksCeil(const float f) const {
        return Eigen::Vector3i(
          (int) ceil(f / voxel_extents_.x()), (int) ceil(f / voxel_extents_.y()), (int) ceil(f / voxel_extents_.z()));
      }

      float getChunkRadiusInMeter() const {
        return voxel_extents_.norm() / 2.0f;
      }

      bool containSDFBlocksChunk(const Eigen::Vector3i& chunk) const {
        auto it = grid_.find(chunk);
        return ((it != grid_.end()) && (grid_.at(chunk)->isStreamedOut()));
      }

      bool containSDFBlocksChunkNeighborhood(const Eigen::Vector3i& chunk,
                                             const Eigen::Vector3i& min_grid_pos,
                                             const Eigen::Vector3i& max_grid_pos,
                                             const int radius) const {
        const Eigen::Vector3i start_chunk = Eigen::Vector3i(std::max(chunk.x() - radius, min_grid_pos.x()),
                                                            std::max(chunk.y() - radius, min_grid_pos.y()),
                                                            std::max(chunk.z() - radius, min_grid_pos.z()));
        const Eigen::Vector3i end_chunk   = Eigen::Vector3i(std::min(chunk.x() + radius, max_grid_pos.x()),
                                                          std::min(chunk.y() + radius, max_grid_pos.y()),
                                                          std::min(chunk.z() + radius, max_grid_pos.z()));

        for (int x = start_chunk.x(); x < end_chunk.x(); ++x) {
          for (int y = start_chunk.y(); y < end_chunk.y(); ++y) {
            for (int z = start_chunk.z(); z < end_chunk.z(); ++z) {
              if (containSDFBlocksChunk(Eigen::Vector3i(x, y, z)))
                return true;
            }
          }
        }
        return false;
      }

      bool isChunkInSphere(const Eigen::Vector3i& chunk, const Eigen::Vector3f& center, float radius) const {
        const Eigen::Vector3f world_pose = chunkToWorld(chunk); // chunk center
        const float l                    = (world_pose - center).norm();
        if (l <= std::abs(radius - chunk_radius_))
          return true;
        return false;
      }

      void serializeData(const std::string& filename_hash  = "./data/hash_points.ply",
                         const std::string& filename_voxel = "./data/voxel_points.ply") const;

      std::pair<Eigen::Vector3i, Eigen::Vector3i> computeBounds() {
        Eigen::Vector3i min_grid_pos(
          std::numeric_limits<int>::max(), std::numeric_limits<int>::max(), std::numeric_limits<int>::max());
        Eigen::Vector3i max_grid_pos(
          std::numeric_limits<int>::lowest(), std::numeric_limits<int>::lowest(), std::numeric_limits<int>::lowest());
        for (const auto& [chunk_pos, _] : grid_) {
          min_grid_pos = min_grid_pos.cwiseMin(chunk_pos);
          max_grid_pos = max_grid_pos.cwiseMax(chunk_pos);
        }
        return {min_grid_pos, max_grid_pos};
      }

      const std::unordered_map<Eigen::Vector3i, std::unique_ptr<ChunkDesc<T>>, Vector3iHash>& getGrid() const {
        return grid_;
      }

      inline void printStatistics() const {
        uint num_SDF_blocks = 0;
        for (const auto& [chunk_pos, chunk_ptr] : grid_) {
          num_SDF_blocks += chunk_ptr->getNElements();
        }
        std::cout << "Streamer | total number of blocks in RAM: " << num_SDF_blocks << std::endl;
      }

      // protected:
      VoxelContainer<T>* container_ = nullptr;

      std::unordered_map<Eigen::Vector3i, std::unique_ptr<ChunkDesc<T>>, Vector3iHash> grid_;

      // output host and device
      SDFBlockDesc* h_SDFBlockDescOutput_ = nullptr;
      T* h_SDFBlockOutput_                = nullptr;
      SDFBlockDesc* d_SDFBlockDescOutput_ = nullptr;
      T* d_SDFBlockOutput_                = nullptr;

      // input
      SDFBlockDesc* d_SDFBlockDescInput_ = nullptr;
      T* d_SDFBlockInput_                = nullptr;

      uint* d_SDF_block_counter_ = nullptr;

      uint curr_stream_out_blocks_                       = 0;
      uint curr_stream_in_blocks_                        = 0;
      uint max_num_sdf_block_integrate_from_global_hash_ = 0;

      uint initial_chunk_list_size_;

      Eigen::Vector3f voxel_extents_;

      float chunk_radius_;
      bool stream_in_done_;

      // asynchronous streams gpu
      cudaEvent_t start_event_, stop_event_;

      std::string memory_allocation_filepath_;
      CUDAProfiler streaming_profiler_;
      float elapsed_time = 0.f;
    };

    template class Streamer<Voxel>;
    using GeometricStreamer = Streamer<Voxel>;

  } // namespace cugeoutils
} // namespace cupanutils
