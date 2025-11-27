#pragma once

#include "streamer.cuh"
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace cupanutils {
  namespace cugeoutils {

    inline constexpr const char* default_serializer_filename = "./data/grid.bin";
    inline constexpr unsigned int max_chunk_size_serializer  = 1024ULL * 1024ULL * 100ULL; // 100MB safety limit
    template <typename T>
    struct Serializer {
      static void serialize(const std::unordered_map<Eigen::Vector3i, std::unique_ptr<ChunkDesc<T>>, Vector3iHash>& grid,
                            const std::string& filename = default_serializer_filename) {
        std::cout << "Serializer::serialize | writing to " << filename << std::endl;

        std::ofstream out(filename, std::ios::binary);
        if (!out.is_open()) {
          throw std::runtime_error("Serializer::serialize | Failed to open file for writing: " + filename);
        }

        for (const auto& [chunk_pos, chunk_ptr] : grid) {
          auto buffer         = cista::serialize(*chunk_ptr);
          uint64_t chunk_size = buffer.size();

          out.write(reinterpret_cast<const char*>(&chunk_size), sizeof(chunk_size));
          out.write(reinterpret_cast<const char*>(chunk_pos.data()), sizeof(Eigen::Vector3i));
          out.write(reinterpret_cast<const char*>(buffer.data()), buffer.size());

          if (!out.good()) {
            throw std::runtime_error("Serializer::serialize | Write failed to: " + filename);
          }
        }

        std::cout << "Serializer::serialize | written " << filename << std::endl;
      }

      static void deserialize(std::unordered_map<Eigen::Vector3i, std::unique_ptr<ChunkDesc<T>>, Vector3iHash>& grid,
                              const std::string& filename = default_serializer_filename) {
        std::cout << "Serializer::deserialize | from " << filename << std::endl;

        std::ifstream in(filename, std::ios::binary);
        if (!in.is_open()) {
          throw std::runtime_error("Serializer::deserialize | Failed to open file for reading: " + filename);
        }

        std::vector<uint8_t> buffer;
        uint64_t chunk_size = 0;

        // Use read() return value for proper EOF detection in binary streams
        while (in.read(reinterpret_cast<char*>(&chunk_size), sizeof(chunk_size))) {
          // Safety check: prevent memory exhaustion from corrupted files
          if (chunk_size > max_chunk_size_serializer) {
            throw std::runtime_error("Serializer::deserialize | Corrupted file: chunk size too large (" +
                                     std::to_string(chunk_size) + " bytes)");
          }

          buffer.resize(chunk_size);

          Eigen::Vector3i chunk_pos;
          in.read(reinterpret_cast<char*>(chunk_pos.data()), sizeof(Eigen::Vector3i));
          in.read(reinterpret_cast<char*>(buffer.data()), chunk_size);

          if (!in.good()) {
            throw std::runtime_error("Serializer::deserialize | Read failed from: " + filename);
          }

          // Deserialize: cista returns a pointer into buffer, so we must copy the data
          ChunkDesc<T>* chunk_desc = cista::deserialize<ChunkDesc<T>>(buffer);
          grid[chunk_pos]          = std::make_unique<ChunkDesc<T>>(*chunk_desc);
        }

        std::cout << "Serializer::deserialize | read " << filename << std::endl;
      }
    };

  } // namespace cugeoutils
} // namespace cupanutils
