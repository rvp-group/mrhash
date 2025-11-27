#pragma once
#include "params.h"
#include <cstdint>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <sys/time.h>

static void HandleError(cudaError_t err, const char* file, int line) {
  // CUDA error handling from the "CUDA by example" book
  if (err != cudaSuccess) {
    printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
    exit(EXIT_FAILURE);
  }
}

#define CUDA_CHECK(err) (HandleError(err, __FILE__, __LINE__))

inline int getDeviceInfo() {
  int n_devices = 0;
  // int available_shmem = 0;
  cudaGetDeviceCount(&n_devices);
  // check for devices
  for (int i = 0; i < n_devices; ++i) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    printf("device number: %d\n", i);
    printf("  device name: %s\n", prop.name);
    printf("  memory clock rate [KHz]: %d\n", prop.memoryClockRate);
    printf("  memory bus width [bits]: %d\n", prop.memoryBusWidth);
    printf("  peak memory bandwidth [GB/s]: %f\n", 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
    printf("  shared mem size per block [KB]: %lui\n", prop.sharedMemPerBlock);
    std::cout << "_______________________________________________" << std::endl;
    // available_shmem = prop.sharedMemPerBlock;
  }
  if (n_devices > 1) {
    std::cout << "multiple devices found, using devices number 0" << std::endl;
    std::cout << "_______________________________________________" << std::endl;
  }
  std::cout << std::endl;
  return n_devices;
}

namespace cupanutils {
  namespace cugeoutils {
    // ! redefining some useful numeric limits for device code
    template <class T>
    struct numeric_limits {
      typedef T type;
      __host__ __device__ static type min() {
        return type();
      }
      __host__ __device__ static type max() {
        return type();
      }
    };

    template <>
    struct numeric_limits<unsigned long> {
      typedef unsigned long type;
      __host__ __device__ static type min() {
        return 0;
      }
      __host__ __device__ static type max() {
        return ULONG_MAX;
      }
    };

    template <>
    struct numeric_limits<unsigned long long> {
      typedef unsigned long long type;
      __host__ __device__ static type min() {
        return 0;
      }
      __host__ __device__ static type max() {
        return UINT64_MAX;
      }
    };

    template <>
    struct numeric_limits<float> {
      typedef float type;
      __host__ __device__ static type min() {
        return 1.175494351e-38f;
      }
      __host__ __device__ static type max() {
        return 3.402823466e+38f;
      }
    };

    template <>
    struct numeric_limits<double> {
      typedef double type;
      __host__ __device__ static type min() {
        return 2.2250738585072014e-308;
      }
      __host__ __device__ static type max() {
        return 1.7976931348623158e+308;
      }
    };

    struct Profiler {
      double elapsed_ = 0;
      int num_events_ = 0;
      std::string name_;

      Profiler(const std::string& name) : name_(name) {
      }

      inline void print(std::ostream& os) {
        std::cout << "HostProfiler | " << name_ << " -- total [ms]: " << elapsed_ << " num events: " << num_events_
                  << " avg [ms]: " << elapsed_ / num_events_ << std::endl;
      }

      struct Event {
        Profiler& profiler_;
        struct timeval t_start_;

        Event(Profiler& prof) : profiler_(prof) {
          gettimeofday(&t_start_, 0);
        }
        inline ~Event() {
          struct timeval t_end_;
          struct timeval duration_;
          gettimeofday(&t_end_, 0);
          timersub(&t_end_, &t_start_, &duration_);
          profiler_.elapsed_ += duration_.tv_sec * 1e3f + duration_.tv_usec * 1e-3;
          ++profiler_.num_events_;
        }
      };
    };

    struct CUDAProfiler {
      std::ofstream outfile;
      double elapsed_ = 0;
      int num_events_ = 0;
      std::string name_;
      bool bwrite_;

      CUDAProfiler() {
      }

      CUDAProfiler(const std::string& name, const bool bwrite = false) : name_(name), bwrite_(bwrite) {
        if (bwrite_) {
          std::string filepath = "./" + name + ".txt";
          outfile.open(filepath);
          outfile << "#CUDAProfiler: " << name << std::endl;
          std::cout << "CUDAProfiler | saving timings to: " << filepath << std::endl;
        }
      }

      ~CUDAProfiler() {
        if (outfile.is_open())
          outfile.close();
      }

      inline void write(const int num_elements = 0) {
        outfile << std::to_string(elapsed_) << " " << std::to_string(num_events_);
        if (num_elements > 0)
          outfile << " " << std::to_string(elapsed_ / (double) num_elements) << " " << std::to_string(num_elements);
        else
          outfile << " 0 0";
        outfile << "\n";
      }

      inline void print(const int num_elements = 0) {
        std::cout << "CUDAProfiler | " << name_ << " -- total [ms]: " << elapsed_ << " num elements: " << num_elements;
        if (num_elements > 0)
          std::cerr << " avg [ms]: " << elapsed_ / (double) num_elements << std::endl;
        std::cerr << std::endl;
      }

      struct CUDAEvent {
        CUDAProfiler& profiler_;
        cudaEvent_t t_start_, t_stop_;

        CUDAEvent(CUDAProfiler& prof) : profiler_(prof) {
          CUDA_CHECK(cudaEventCreate(&t_start_));
          CUDA_CHECK(cudaEventCreate(&t_stop_));
          CUDA_CHECK(cudaEventRecord(t_start_));
        }

        inline ~CUDAEvent() {
          CUDA_CHECK(cudaEventRecord(t_stop_));
          CUDA_CHECK(cudaEventSynchronize(t_stop_));
          float duration_ms = 0;
          CUDA_CHECK(cudaEventElapsedTime(&duration_ms, t_start_, t_stop_));
          profiler_.elapsed_ = duration_ms;
          profiler_.num_events_++;
          CUDA_CHECK(cudaEventDestroy(t_start_));
          CUDA_CHECK(cudaEventDestroy(t_stop_));
        }
      };
    };

  } // namespace cugeoutils
} // namespace cupanutils