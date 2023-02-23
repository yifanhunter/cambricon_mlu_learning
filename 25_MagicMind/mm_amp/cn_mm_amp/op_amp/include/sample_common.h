/*************************************************************************
 * Copyright (C) [2020-2023] by Cambricon, Inc.
 *************************************************************************/
#ifndef SAMPLE_COMMON_H_
#define SAMPLE_COMMON_H_
#include "cnrt.h"
#include "mm_runtime.h"
#include <iostream>
#include <fstream>
#include <cmath>
#include <fstream>
#include <algorithm>
#include <queue>
#include <string.h>
#include <sstream>
#include <iterator>
#include <vector>
#include <chrono>
#include <random>
#include <type_traits>

#define MM_CHECK(status)                               \
  do {                                                 \
    auto ret = (status);                               \
    if (ret != magicmind::Status::OK()) {              \
      std::cout << "mm failure: " << ret << std::endl; \
      abort();                                         \
    }                                                  \
  } while (0)
// TO DO print line
#define LOGINFO(fmt, args...) fprintf(stdout, "[MMINFO]  " fmt "\n", ##args)

#define PTR_CHECK(ptr)                         \
  do {                                         \
    if (ptr == nullptr) {                      \
      std::cout << "mm failure " << std::endl; \
      abort();                                 \
    }                                          \
  } while (0)

struct InferDeleter {
  template <typename T>
  void operator()(T *obj) const {
    if (obj) {
      obj->Destroy();
    }
  }
};
template <typename T>
using sample_unique_ptr = std::unique_ptr<T, InferDeleter>;

template <class T, bool is_integral = std::is_integral<T>::value>
struct RandDist {};

template <typename T>
struct RandDist<T, false> {
  typedef std::uniform_real_distribution<T> Dist;
};

template <typename T>
struct RandDist<T, true> {
  typedef std::uniform_int_distribution<T> Dist;
};

template <class T>
std::vector<T> GenRand(uint64_t len, T begin, T end, unsigned int seed) {
  //CHECK_LE(begin, end);
  std::vector<T> ret(len);
  std::default_random_engine eng(seed);
  typename RandDist<T>::Dist dist(begin, end);
  for (size_t idx = 0; idx < len; ++idx) {
    ret[idx] = dist(eng);
  }
  return ret;
}

inline void getRandFloat(float *input, int64_t len, int64_t scale) {
  for (int i = 0; i < len; i++) {
    input[i] = (rand() % (scale * 1000)) / 1000.0 - scale / 2.0;
  }
}

inline void getRandInt(int32_t *input, int64_t len, int64_t scale) {
  for (int i = 0; i < len; i++) {
    input[i] = rand() % (scale + 1);
  }
}
static std::map<magicmind::DataType, cnrtDataType_t> kDtypeMap{
    {magicmind::DataType::QINT8, CNRT_INT8},      {magicmind::DataType::QINT16, CNRT_INT16},
    {magicmind::DataType::INT8, CNRT_INT8},       {magicmind::DataType::INT16, CNRT_INT16},
    {magicmind::DataType::INT32, CNRT_INT32},     {magicmind::DataType::UINT8, CNRT_UINT8},
    {magicmind::DataType::FLOAT16, CNRT_FLOAT16}, {magicmind::DataType::FLOAT32, CNRT_FLOAT32},
};

inline cnrtDataType_t ConvertDataType(magicmind::DataType dtype) {
  auto iter = kDtypeMap.find(dtype);
  if (iter != kDtypeMap.end()) {
    return iter->second;
  } else {
    LOGINFO("Invalid DataType.");
    return CNRT_INVALID;
  }
}

static std::map<magicmind::TensorLocation, std::string> kTensorLocationMap{
    {magicmind::TensorLocation::kHost, "kHost"},
    {magicmind::TensorLocation::kMLU, "kMLU"},
    {magicmind::TensorLocation::kRemoteHost, "kRemoteHost"}};

std::string inline TensorLocationEnumToString(magicmind::TensorLocation loc) {
  auto iter = kTensorLocationMap.find(loc);
  if (iter != kTensorLocationMap.end()) {
    return iter->second;
  } else {
    return "INVALID";
  }
}

template <class T>
inline void getMinAndMax(T *input, size_t size, cnrtDataType_t dtype, double *min, double *max) {
  if (input == NULL || size == 0 || min == NULL || max == NULL) {
    std::cout << "invalid input paramter!" << std::endl;
    return;
  }
  for (size_t index = 0; index < size; ++index) {
    if (input[index] > *max) {
      *max = input[index];
    }
    if (input[index] < *min) {
      *min = input[index];
    }
  }
}

template <typename T>
struct cmpPair {
  bool operator()(const std::pair<int32_t, T> &a, const std::pair<int32_t, T> &b) {
    return a.second > b.second;
  }
};

// find top dst.size() in src
template <typename T>
bool Topk(std::vector<int32_t> &dst, std::vector<T> src) {
  if (dst.size() == 0 || src.size() == 0)
    return false;
  std::priority_queue<std::pair<int32_t, T>, std::vector<std::pair<int32_t, T>>, cmpPair<T>>
      noise_words;
  uint32_t min_size = (src.size() < dst.size()) ? src.size() : dst.size();
  for (uint32_t i = 0; i < min_size; ++i) {
    noise_words.push(std::make_pair(i, src[i]));
  }
  if (src.size() > dst.size()) {
    for (uint32_t i = min_size; i < src.size(); ++i) {
      if (src[i] < noise_words.top().second) {
        continue;
      } else {
        noise_words.pop();
        noise_words.push(std::make_pair(i, src[i]));
      }
    }
  }
  for (uint32_t i = 0; i < min_size; ++i) {
    dst[min_size - i - 1] = noise_words.top().first;
    noise_words.pop();
  }
  return true;
}

// read binary files to data_ptr
// file size should be set
template <typename T>
void getDataFromFile(const std::string path,
                     std::vector<std::string> files,
                     T *data_ptr,
                     int datasize) {
  char *temp_data = (char *)data_ptr;
  for (auto file : files) {
    std::string temp_path = path + file;
    std::ifstream inFile(temp_path, std::ios::in | std::ios::binary);
    if (!inFile) {
      LOGINFO("Open file %s failed.", temp_path.c_str());
      continue;
    }
    inFile.read(temp_data, datasize);
    inFile.close();
    temp_data += datasize;
  }
}

inline void split(const std::string &str, std::vector<std::string> &tokens) {
  tokens.clear();
  std::istringstream iss(str);
  std::copy(std::istream_iterator<std::string>(iss), std::istream_iterator<std::string>(),
            std::back_inserter(tokens));
}

inline bool read_label(const char *label_txt,
                       std::vector<std::string> &images,
                       std::vector<int> &labels) {
  std::string line;
  std::ifstream flabel(label_txt);
  if (!flabel) {
    printf("can not open label file %s\n", label_txt);
    return false;
  }
  std::vector<std::string> tokens;
  while (getline(flabel, line)) {
    split(line, tokens);
    if (tokens.size() != 2) {
      return false;
    } else {
      // ILSVRC2012_val_00000001.JPEG->ILSVRC2012_val_00000001
      images.push_back(tokens[0].substr(0, tokens[0].length() - 5));
      labels.push_back(std::stoi(tokens[1]));
    }
  }
  flabel.close();
  if (images.size() == 0)
    return false;
  return true;
}

// --------------------------------------------------------
// For bert demo usage
// --------------------------------------------------------
#include "mm_builder.h"
#include "mm_network.h"
#include <cassert>
using namespace magicmind;
using ConstNodeMap = std::map<std::string, IConstNode *>;

inline void parseDimsByName(std::ifstream &input, const std::string &name, Dims &d) {
  int nb_dims;
  input >> nb_dims;
  std::vector<int64_t> dims(nb_dims, 0);
  for (int it = 0; it < nb_dims; it++) {
    input >> dims[it];
  }
  d = Dims(dims);
  assert(input.peek() == ' ');
  input.get();
}

template <typename T>
inline void loadData(std::ifstream &input, std::string &name, T *&data, int &nbWeights, Dims &d) {
  int32_t type;
  int32_t nbDims;
  int32_t dim;
  input >> name >> std::dec >> type;
  parseDimsByName(input, name, d);
  nbWeights = d.GetElementCount();
  if (data == nullptr) {
    data = new T[nbWeights];
  }
  // data = new T[nbWeights]; // should be freed by caller
  char *bytes = reinterpret_cast<char *>(data);
  input.read(bytes, nbWeights * sizeof(T));

  assert(input.peek() == '\n');
  input.get();
}

inline void loadWeights(const std::string &wts_path, INetwork *network, ConstNodeMap &nodeMap) {
  std::ifstream input(wts_path, std::ios_base::binary);
  int32_t count;
  input >> count;
  // std::cout << "Number of parameters: " << count << std::endl;

  for (int it = 0; it < count; it++) {
    int param_size = 0;
    std::string name;
    float *data = nullptr;
    Dims d;
    loadData<float>(input, name, data, param_size, d);
    IConstNode *node = network->AddIConstNode(DataType::FLOAT32, d, data);
    nodeMap[name]    = node;
  }
}

inline void loadInputs(const std::string &wts_path, std::vector<void *> &input_cpu_ptrs) {
  std::ifstream input(wts_path, std::ios_base::binary);
  int32_t count;
  input >> count;
  // std::cout << "Number of Inputs: " << count << std::endl;
  assert(count % 3 == 0);

  for (int it = 0; it < count; it++) {
    int param_size = 0;
    std::string name;
    int *data = reinterpret_cast<int *>(input_cpu_ptrs[it]);
    Dims d;
    loadData<int>(input, name, data, param_size, d);
    input_cpu_ptrs[it] = reinterpret_cast<void *>(data);
  }
}

inline void loadOutputs(const std::string &wts_path, float *input_cpu_ptrs) {
  std::ifstream input(wts_path, std::ios_base::binary);
  int32_t count;
  input >> count;
  // std::cout << "Number of Outputs: " << count << std::endl;
  assert(count % 1 == 0);
  int param_size = 0;
  std::string name;
  Dims d;
  loadData<float>(input, name, input_cpu_ptrs, param_size, d);
}

#endif  // SAMPLE_COMMON_H_
