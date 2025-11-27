
#include <sdf/cuda_matrix.cuh>
#include <iostream>
#include <vector>

#include "test_utils.cuh"

using namespace cupanutils::cugeoutils;

template <typename Mat>
__global__ void checkFieldsDevice(uint32_t* fields_, const Mat* mat_) {
  fields_[0] = mat_->rows();
  fields_[1] = mat_->cols();
  fields_[2] = mat_->size();
}

template <typename Entry, typename Mat>
__global__ void copyFieldsToDevice(Entry* entries_, const Mat* mat_) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= mat_->size())
    return;
  entries_[tid] = mat_->at<1>(tid);
}

TEST(FLOAT, DualMatrixFloat) {
  using Entry       = float;
  using CUDAMatrixT = DualMatrix_<Entry>;
  // check creation and resizing
  uint32_t rows = 300;
  uint32_t cols = 300;
  CUDAMatrixT mat(rows, cols);
  mat  = CUDAMatrixT();
  rows = 400;
  cols = 300;
  mat.resize(rows, cols);
  mat.fill(0.f);

  uint32_t* mat_fields_device = nullptr;
  uint32_t* mat_fields_host   = new uint32_t[3];
  memset(mat_fields_host, 0, sizeof(uint32_t) * 3);
  CUDA_CHECK(cudaMalloc((void**) &mat_fields_device, sizeof(uint32_t) * 3));
  checkFieldsDevice<CUDAMatrixT><<<1, 1>>>(mat_fields_device, mat.deviceInstance());
  CUDA_CHECK(cudaMemcpy(mat_fields_host, mat_fields_device, sizeof(uint32_t) * 3, cudaMemcpyDeviceToHost));

  uint32_t dev_rows     = mat_fields_host[0];
  uint32_t dev_cols     = mat_fields_host[1];
  uint32_t dev_capacity = mat_fields_host[2];

  ASSERT_EQ(rows, mat.rows());
  ASSERT_EQ(cols, mat.cols());
  ASSERT_EQ(rows * cols, mat.size());
  ASSERT_EQ(dev_rows, mat.rows());
  ASSERT_EQ(dev_cols, mat.cols());
  ASSERT_EQ(dev_capacity, mat.size());

  // check only reconstruction and resizing
  rows = 700;
  cols = 100;

  mat = CUDAMatrixT();
  mat.resize(rows, cols);
  mat.fill(0.f);

  checkFieldsDevice<CUDAMatrixT><<<1, 1>>>(mat_fields_device, mat.deviceInstance());
  CUDA_CHECK(cudaMemcpy(mat_fields_host, mat_fields_device, sizeof(uint32_t) * 3, cudaMemcpyDeviceToHost));

  dev_rows     = mat_fields_host[0];
  dev_cols     = mat_fields_host[1];
  dev_capacity = mat_fields_host[2];

  CUDA_CHECK(cudaFree(mat_fields_device));
  delete[] mat_fields_host;

  ASSERT_EQ(rows, mat.rows());
  ASSERT_EQ(cols, mat.cols());
  ASSERT_EQ(rows * cols, mat.size());
  ASSERT_EQ(dev_rows, mat.rows());
  ASSERT_EQ(dev_cols, mat.cols());
  ASSERT_EQ(dev_capacity, mat.size());

  // checking that host manipulation of entries is valid in device
  Entry* buff_entries_host = new Entry[mat.size()];
  Entry* buff_entries_dev  = nullptr;
  CUDA_CHECK(cudaMalloc((void**) &buff_entries_dev, sizeof(Entry) * mat.size()));

  // modify entries of matrix in host
  for (uint32_t i = 0; i < mat.size(); ++i) {
    mat.at(i) = (rand()) / (static_cast<float>(RAND_MAX / 1.f));
  }

  // copy to device
  mat.toDevice();
  // check that host and device contain the same element
  copyFieldsToDevice<Entry, CUDAMatrixT><<<mat.nBlocks(), mat.nThreads()>>>(buff_entries_dev, mat.deviceInstance());
  cudaDeviceSynchronize();

  CUDA_CHECK(cudaMemcpy(buff_entries_host, buff_entries_dev, sizeof(Entry) * mat.size(), cudaMemcpyDeviceToHost));

  for (uint32_t i = 0; i < mat.size(); ++i) {
    ASSERT_EQ(mat.at(i), buff_entries_host[i]);
  }

  CUDA_CHECK(cudaFree(buff_entries_dev));
  delete[] buff_entries_host;

  return;
}

struct DummyEntry {
  DummyEntry() {
    field1 = 0.f;
  }
  float field1    = 0.f;
  uint32_t field2 = 0;
};

template <typename Mat>
__global__ void checkMemPaddingEntryDevice(int* add1_, int* add2_, int* add3_, const Mat* mat_) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= mat_->size())
    return;
  const auto elem = mat_->at<1>(tid);
  int* add1       = (int*) &elem;
  int* add2       = (int*) &(elem.field1);
  int* add3       = (int*) &(elem.field2);
  // printf("add: %p %p %p\n", add1, add2, add3);
  // printf("diff: %i %i %i",add3-add1,add2-add1,add3-add2);
  add1_[tid] = add3 - add1;
  add2_[tid] = add2 - add1;
  add3_[tid] = add3 - add2;
}

TEST(DUMMY, DualMatrixDummyEntry) {
  using Entry       = DummyEntry;
  using CUDAMatrixT = DualMatrix_<Entry>;
  // check creation and resizing
  uint32_t rows = 300;
  uint32_t cols = 300;
  CUDAMatrixT mat(rows, cols);
  mat  = CUDAMatrixT();
  rows = 400;
  cols = 300;
  mat.resize(rows, cols);
  mat.fill(DummyEntry());

  uint32_t* mat_fields_device = nullptr;
  uint32_t* mat_fields_host   = new uint32_t[3];
  memset(mat_fields_host, 0, sizeof(uint32_t) * 3);
  CUDA_CHECK(cudaMalloc((void**) &mat_fields_device, sizeof(uint32_t) * 3));
  checkFieldsDevice<CUDAMatrixT><<<1, 1>>>(mat_fields_device, mat.deviceInstance());
  CUDA_CHECK(cudaMemcpy(mat_fields_host, mat_fields_device, sizeof(uint32_t) * 3, cudaMemcpyDeviceToHost));

  uint32_t dev_rows     = mat_fields_host[0];
  uint32_t dev_cols     = mat_fields_host[1];
  uint32_t dev_capacity = mat_fields_host[2];

  ASSERT_EQ(rows, mat.rows());
  ASSERT_EQ(cols, mat.cols());
  ASSERT_EQ(rows * cols, mat.size());
  ASSERT_EQ(dev_rows, mat.rows());
  ASSERT_EQ(dev_cols, mat.cols());
  ASSERT_EQ(dev_capacity, mat.size());

  // check only reconstruction and resizing
  rows = 700;
  cols = 100;

  mat = CUDAMatrixT();
  mat.resize(rows, cols);
  mat.fill(DummyEntry());

  checkFieldsDevice<CUDAMatrixT><<<1, 1>>>(mat_fields_device, mat.deviceInstance());
  CUDA_CHECK(cudaMemcpy(mat_fields_host, mat_fields_device, sizeof(uint32_t) * 3, cudaMemcpyDeviceToHost));

  dev_rows     = mat_fields_host[0];
  dev_cols     = mat_fields_host[1];
  dev_capacity = mat_fields_host[2];

  CUDA_CHECK(cudaFree(mat_fields_device));
  delete[] mat_fields_host;

  ASSERT_EQ(rows, mat.rows());
  ASSERT_EQ(cols, mat.cols());
  ASSERT_EQ(rows * cols, mat.size());
  ASSERT_EQ(dev_rows, mat.rows());
  ASSERT_EQ(dev_cols, mat.cols());
  ASSERT_EQ(dev_capacity, mat.size());

  // checking that host manipulation of entries is valid in device
  Entry* buff_entries_host = new Entry[mat.size()];
  Entry* buff_entries_dev  = nullptr;
  CUDA_CHECK(cudaMalloc((void**) &buff_entries_dev, sizeof(Entry) * mat.size()));

  // modify entries of matrix in host
  for (uint32_t i = 0; i < mat.size(); ++i) {
    // mat.at(i).setDepth(static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / max_depth)));
    // mat.at(i).setIntensity(static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / 1.f)));
    // mat.at(i).setNormal(Vector3f::Random());
    mat.at(i).field1 = (rand()) / (static_cast<float>(RAND_MAX / 1.f));
    mat.at(i).field2 = i;
  }

  // copy to device
  mat.toDevice();
  // check that host and device contain the same element
  copyFieldsToDevice<Entry, CUDAMatrixT><<<mat.nBlocks(), mat.nThreads()>>>(buff_entries_dev, mat.deviceInstance());
  cudaDeviceSynchronize();

  CUDA_CHECK(cudaMemcpy(buff_entries_host, buff_entries_dev, sizeof(Entry) * mat.size(), cudaMemcpyDeviceToHost));

  for (uint32_t i = 0; i < mat.size(); ++i) {
    ASSERT_EQ(mat.at(i).field1, buff_entries_host[i].field1);
    ASSERT_EQ(mat.at(i).field2, buff_entries_host[i].field2);
    // ASSERT_EQ(mat.at(i).depth(), buff_entries_host[i].depth());
    // ASSERT_EQ(mat.at(i).intensity(), buff_entries_host[i].intensity());
    // ASSERT_EQ(mat.at(i).normal(), buff_entries_host[i].normal());
  }

  cudaFree(buff_entries_dev);
  delete[] buff_entries_host;

  // checking for struct padding
  int* adds_dev[3] = {nullptr, nullptr, nullptr};
  for (int i = 0; i < 3; ++i) {
    CUDA_CHECK(cudaMalloc((void**) &adds_dev[i], sizeof(int) * mat.size()));
  }
  int* adds[3] = {new int[mat.size()], new int[mat.size()], new int[mat.size()]};
  checkMemPaddingEntryDevice<CUDAMatrixT>
    <<<mat.nBlocks(), mat.nThreads()>>>(adds_dev[0], adds_dev[1], adds_dev[2], mat.deviceInstance());
  cudaDeviceSynchronize();

  for (int i = 0; i < 3; ++i) {
    CUDA_CHECK(cudaMemcpy(adds[i], adds_dev[i], sizeof(int) * mat.size(), cudaMemcpyDeviceToHost));
  }

  for (int i = 0; i < 0; ++i) {
    const auto elem = mat.at(i);
    int* add1       = (int*) &elem;
    int* add2       = (int*) &(elem.field1);
    int* add3       = (int*) &(elem.field2);

    const int diff1 = add3 - add1;
    const int diff2 = add2 - add1;
    const int diff3 = add3 - add2;

    ASSERT_EQ(adds[0][i], diff1);
    ASSERT_EQ(adds[1][i], diff2);
    ASSERT_EQ(adds[2][i], diff3);
  }

  for (int i = 0; i < 3; ++i) {
    cudaFree(adds_dev[i]);
    delete[] adds[i];
  }

  return;
}

using namespace Eigen;

struct EigenEntry {
  EigenEntry() {
    field2.setIdentity();
  }
  Vector3f field1   = Vector3f::Zero();
  Isometry3d field2 = Isometry3d::Identity();
};

TEST(EIGEN, DualMatrixEigenEntry) {
  using Entry       = EigenEntry;
  using CUDAMatrixT = DualMatrix_<Entry>;
  // check creation and resizing
  uint32_t rows = 300;
  uint32_t cols = 300;
  CUDAMatrixT mat(rows, cols);
  mat  = CUDAMatrixT();
  rows = 400;
  cols = 300;
  mat.resize(rows, cols);
  mat.fill(EigenEntry());

  uint32_t* mat_fields_device = nullptr;
  uint32_t* mat_fields_host   = new uint32_t[3];
  memset(mat_fields_host, 0, sizeof(uint32_t) * 3);
  CUDA_CHECK(cudaMalloc((void**) &mat_fields_device, sizeof(uint32_t) * 3));
  checkFieldsDevice<CUDAMatrixT><<<1, 1>>>(mat_fields_device, mat.deviceInstance());
  CUDA_CHECK(cudaMemcpy(mat_fields_host, mat_fields_device, sizeof(uint32_t) * 3, cudaMemcpyDeviceToHost));

  uint32_t dev_rows     = mat_fields_host[0];
  uint32_t dev_cols     = mat_fields_host[1];
  uint32_t dev_capacity = mat_fields_host[2];

  ASSERT_EQ(rows, mat.rows());
  ASSERT_EQ(cols, mat.cols());
  ASSERT_EQ(rows * cols, mat.size());
  ASSERT_EQ(dev_rows, mat.rows());
  ASSERT_EQ(dev_cols, mat.cols());
  ASSERT_EQ(dev_capacity, mat.size());

  // check only reconstruction and resizing
  rows = 700;
  cols = 100;

  mat = CUDAMatrixT();
  mat.resize(rows, cols);
  mat.fill(EigenEntry());

  checkFieldsDevice<CUDAMatrixT><<<1, 1>>>(mat_fields_device, mat.deviceInstance());
  CUDA_CHECK(cudaMemcpy(mat_fields_host, mat_fields_device, sizeof(uint32_t) * 3, cudaMemcpyDeviceToHost));

  dev_rows     = mat_fields_host[0];
  dev_cols     = mat_fields_host[1];
  dev_capacity = mat_fields_host[2];

  CUDA_CHECK(cudaFree(mat_fields_device));
  delete[] mat_fields_host;

  ASSERT_EQ(rows, mat.rows());
  ASSERT_EQ(cols, mat.cols());
  ASSERT_EQ(rows * cols, mat.size());
  ASSERT_EQ(dev_rows, mat.rows());
  ASSERT_EQ(dev_cols, mat.cols());
  ASSERT_EQ(dev_capacity, mat.size());

  // checking that host manipulation of entries is valid in device
  Entry* buff_entries_host = new Entry[mat.size()];
  Entry* buff_entries_dev  = nullptr;
  CUDA_CHECK(cudaMalloc((void**) &buff_entries_dev, sizeof(Entry) * mat.size()));

  // modify entries of matrix in host
  for (uint32_t i = 0; i < mat.size(); ++i) {
    mat.at(i).field1                   = Vector3f::Random();
    mat.at(i).field2.translation().x() = (static_cast<double>(RAND_MAX / 1.0));
  }

  // copy to device
  mat.toDevice();
  // check that host and device contain the same element
  copyFieldsToDevice<Entry, CUDAMatrixT><<<mat.nBlocks(), mat.nThreads()>>>(buff_entries_dev, mat.deviceInstance());
  cudaDeviceSynchronize();

  CUDA_CHECK(cudaMemcpy(buff_entries_host, buff_entries_dev, sizeof(Entry) * mat.size(), cudaMemcpyDeviceToHost));

  for (uint32_t i = 0; i < mat.size(); ++i) {
    ASSERT_EQ(mat.at(i).field1, buff_entries_host[i].field1);
    ASSERT_EQ(mat.at(i).field2.translation().x(), buff_entries_host[i].field2.translation().x());
  }

  cudaFree(buff_entries_dev);
  delete[] buff_entries_host;

  // checking for struct padding
  int* adds_dev[3] = {nullptr, nullptr, nullptr};
  for (int i = 0; i < 3; ++i) {
    CUDA_CHECK(cudaMalloc((void**) &adds_dev[i], sizeof(int) * mat.size()));
  }
  int* adds[3] = {new int[mat.size()], new int[mat.size()], new int[mat.size()]};
  checkMemPaddingEntryDevice<CUDAMatrixT>
    <<<mat.nBlocks(), mat.nThreads()>>>(adds_dev[0], adds_dev[1], adds_dev[2], mat.deviceInstance());
  cudaDeviceSynchronize();

  for (int i = 0; i < 3; ++i) {
    CUDA_CHECK(cudaMemcpy(adds[i], adds_dev[i], sizeof(int) * mat.size(), cudaMemcpyDeviceToHost));
  }

  for (int i = 0; i < 0; ++i) {
    const auto elem = mat.at(i);
    int* add1       = (int*) &elem;
    int* add2       = (int*) &(elem.field1);
    int* add3       = (int*) &(elem.field2);

    const int diff1 = add3 - add1;
    const int diff2 = add2 - add1;
    const int diff3 = add3 - add2;

    ASSERT_EQ(adds[0][i], diff1);
    ASSERT_EQ(adds[1][i], diff2);
    ASSERT_EQ(adds[2][i], diff3);
  }

  for (int i = 0; i < 3; ++i) {
    cudaFree(adds_dev[i]);
    delete[] adds[i];
  }

  return;
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
