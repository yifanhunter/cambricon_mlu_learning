/*************************************************************************
 * Copyright (C) [2020-2023] by Cambricon, Inc.
 *************************************************************************/
#ifndef SAMPLE_BUFFER_H_
#define SAMPLE_BUFFER_H_

#include "sample_common.h"
/*!
 * @class MMBuffer
 *
 * @brief a class creates input/output mlu/cpu buffers according to input/output irttensor,
 * and support memcpy sync/async bwtween mlu and cpu buffers.
 *
 * @par Requirements
 * - cnrt.h, interface_runtime.h
 *
 * @par See Also
 * - IRTTensor
 */
class MMBuffer {
 public:
  MMBuffer(){};
  // for enqueue(input_tensors,output_tensors)
  bool Init(std::vector<magicmind::IRTTensor *> &input_tensors,
            std::vector<magicmind::IRTTensor *> &output_tensors);
  // for enqueue(input_tensors,&output_tensors)
  bool Init(std::vector<magicmind::IRTTensor *> &input_tensors);
  // get ptrs
  std::vector<void *> InputCpuBuffers() const { return cpu_input_ptrs_; }
  std::vector<void *> OutputCpuBuffers() const { return cpu_output_ptrs_; }
  std::vector<void *> InputMluBuffers() const { return mlu_input_ptrs_; }
  std::vector<void *> OutputMluBuffers() const { return mlu_output_ptrs_; }

  // memcpy
  void D2H(cnrtQueue_t queue);
  void H2D(cnrtQueue_t queue);
  void D2H();
  void H2D();

  void Destroy();
  std::string DebugString();

 private:
  uint32_t input_num_  = 0;
  uint32_t output_num_ = 0;
  std::vector<void *> mlu_input_ptrs_;
  std::vector<void *> mlu_output_ptrs_;
  std::vector<void *> cpu_input_ptrs_;
  std::vector<void *> cpu_output_ptrs_;
  std::vector<magicmind::IRTTensor *> input_tensors_;
  std::vector<magicmind::IRTTensor *> output_tensors_;
  void MallocBuffers(std::vector<magicmind::IRTTensor *> &tensors,
                     std::vector<void *> &cpu_ptrs,
                     std::vector<void *> &mlu_ptrs);
};

#endif  // SAMPLE_BUFFER_H_
