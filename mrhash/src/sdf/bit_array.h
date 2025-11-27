
#pragma once
#include <cstring>
#include <iostream>
#include <stdexcept>

template <class T>
class BitArray {
public:
  BitArray() {
    data_     = nullptr;
    num_bits_ = 0;
  }

  BitArray(unsigned int num_bits) {
    num_bits_ = num_bits;

    const unsigned int nInts = (num_bits_ + num_bits_in_T_ - 1) / num_bits_in_T_;
    data_                    = new T[nInts];
    for (unsigned int i = 0; i < nInts; i++) {
      data_[i] = T();
    }
  }

  bool isBitSet(unsigned int index) const {
    if (index < num_bits_) {
      return ((data_[index / num_bits_in_T_] & (0x1 << (index % num_bits_in_T_))) != 0x0);
    } else {
      throw std::out_of_range("BitArray::isBitSet - index out of bounds");
    }
  }

  void setBit(unsigned int index) {
    if (index < num_bits_) {
      data_[index / num_bits_in_T_] |= (0x1 << (index % num_bits_in_T_));
    } else {
      throw std::out_of_range("BitArray::setBit - index out of bounds");
    }
  }

  void resetBit(unsigned int index) {
    if (index < num_bits_) {
      data_[index / num_bits_in_T_] &= ~(0x1 << (index % num_bits_in_T_));
    } else {
      throw std::out_of_range("BitArray::resetBit - index out of bounds");
    }
  }

  void reset() {
    const unsigned int nInts = (num_bits_ + num_bits_in_T_ - 1) / num_bits_in_T_;
    for (unsigned int i = 0; i < nInts; i++) {
      data_[i] = T();
    }
  }

  const T* getRawData() const {
    return data_;
  }

  unsigned int getNBits() const {
    return num_bits_;
  }

  unsigned int getByteWidth() const {
    return sizeof(T) * ((num_bits_ + num_bits_in_T_ - 1) / num_bits_in_T_);
  }

  BitArray(const BitArray<T>& other) : data_(nullptr), num_bits_(0) {
    *this = other;
  }

  void operator=(const BitArray<T>& other) {
    if (this != &other) {
      delete[] data_;
      num_bits_ = other.getNBits();
      data_     = new T[other.getByteWidth() / sizeof(T)];
      memcpy(&data_[0], &(other.getRawData()[0]), other.getByteWidth());
    }
  }

  // Move constructor
  BitArray(BitArray<T>&& other) noexcept : data_(other.data_), num_bits_(other.num_bits_) {
    other.data_     = nullptr;
    other.num_bits_ = 0;
  }

  // Move assignment operator
  BitArray<T>& operator=(BitArray<T>&& other) noexcept {
    if (this != &other) {
      delete[] data_;
      data_           = other.data_;
      num_bits_       = other.num_bits_;
      other.data_     = nullptr;
      other.num_bits_ = 0;
    }
    return *this;
  }

  ~BitArray() {
    delete[] data_;
  }

private:
  static const unsigned int num_bits_in_T_ = 8 * sizeof(T);
  unsigned int num_bits_;
  T* data_;
};
