#include <iostream>
#include <memory>
#include <vector>
#include "cnrt.h"
#include "mm_builder.h"
#include "mm_network.h"
#include "mm_runtime.h"
#include "sample_common.h"

#define Setfp16conv

using namespace magicmind;
/*
 * To construct network as follow:
 * input/bias/filter(fp32, dims)
 *            |
 *          Conv
 *            |
 *       conv_output
 *            |
 *          ReLU
 *            |
 *         ReLU_output
 *            |
 *          Conv1 (you can set precision fp32 to fp16)
 *            |
 *        Conv1_output
 *            |
 *          ReLU1
 *            |
 *         ReLU1_output
 */
void ConstructAddNetwork(const char *model_name) {
  // init
  auto builder = sample_unique_ptr<magicmind::IBuilder>(magicmind::CreateIBuilder());
  auto network = sample_unique_ptr<magicmind::INetwork>(magicmind::CreateINetwork());

  // create input tensor
  magicmind::Range range = {-1.0f, 1.0f};
  auto input_tensor = network->AddInput(DataType::FLOAT32, Dims({1,224,224,3}));
  input_tensor->SetDynamicRange(range, false);
  PTR_CHECK(input_tensor);
  MM_CHECK(input_tensor->SetDynamicRange(range, false));

  // create filter tensor
  auto filter_dim = magicmind::Dims({64, 7, 7, 3});

  std::vector<float> filter_float = GenRand(filter_dim.GetElementCount(), -1.0f, 1.0f, 0);
  // set filter
  auto filter = network->AddIConstNode(DataType::FLOAT32, filter_dim, filter_float.data());
  PTR_CHECK(filter);
  auto filter_tensor = filter->GetOutput(0);
  PTR_CHECK(filter_tensor);
  MM_CHECK(filter_tensor->SetDynamicRange(range, false));
  filter_tensor->SetDynamicRange(range, false);

  // create bias tensor
  auto bias_dim = magicmind::Dims({64});
  std::vector<float> bias_float = GenRand(bias_dim.GetElementCount(), -1.0f, 1.0f, 0);
  magicmind::ITensor *bias_tensor = nullptr;
  auto bias = network->AddIConstNode(DataType::FLOAT32, bias_dim, bias_float.data());
  PTR_CHECK(bias);
  bias_tensor = bias->GetOutput(0);

  // create conv + relu node
  auto conv = network->AddIConvNode(input_tensor, filter_tensor, bias_tensor);
  PTR_CHECK(conv);
  int stride[2] = {2, 2};
  int pad[4] = {3, 3, 3, 3};
  int dilation[2] = {1, 1};

  MM_CHECK(conv->SetStride(stride[0], stride[1]));
  MM_CHECK(conv->SetPad(pad[0], pad[1], pad[2], pad[3]));
  MM_CHECK(conv->SetDilation(dilation[0], dilation[1]));
  MM_CHECK(conv->SetPaddingMode(magicmind::IPaddingMode::EXPLICIT));
  auto conv_output = conv->GetOutput(0);
  PTR_CHECK(conv_output);
  // conv output tensor datatype should be set same with bias tensor
  MM_CHECK(conv->SetOutputType(0, DataType::FLOAT32));
  // relu output tensor datatype will be same with input tensor
  auto relu = network->AddIActivationNode(conv_output, magicmind::IActivation::RELU);
  PTR_CHECK(relu);
#ifdef Setfp16conv
  MM_CHECK(relu->SetOutputType(0, DataType::FLOAT16));
#else
  MM_CHECK(relu->SetOutputType(0, DataType::FLOAT32));
#endif

  // create filter1 tensor
  auto filter1_dim = magicmind::Dims({128, 3, 3, 64});

  std::vector<float> filter1_float = GenRand(filter1_dim.GetElementCount(), -1.0f, 1.0f, 0);
  // set filter
  auto filter1 = network->AddIConstNode(DataType::FLOAT32, filter1_dim, filter1_float.data());
  PTR_CHECK(filter1);
  auto filter1_tensor = filter1->GetOutput(0);
  PTR_CHECK(filter1_tensor);
  MM_CHECK(filter1_tensor->SetDynamicRange(range, false));
  filter1_tensor->SetDynamicRange(range, false);

  // create bias1 tensor
  auto bias1_dim = magicmind::Dims({128});
  std::vector<float> bias1_float = GenRand(bias1_dim.GetElementCount(), -1.0f, 1.0f, 0);
  magicmind::ITensor *bias1_tensor = nullptr;
  auto bias1 = network->AddIConstNode(DataType::FLOAT32, bias1_dim, bias1_float.data());
  PTR_CHECK(bias1);
  bias1_tensor = bias1->GetOutput(0);

  // create conv1 relu1 node
  auto conv1 = network->AddIConvNode(relu->GetOutput(0), filter1_tensor, bias1_tensor);
  PTR_CHECK(conv1);
  int stride1[2] = {2, 2};
  int pad1[4] = {0, 0, 0, 0};
  int dilation1[2] = {1, 1};

  MM_CHECK(conv1->SetStride(stride1[0], stride1[1]));
  MM_CHECK(conv1->SetPad(pad1[0], pad1[1], pad1[2], pad1[3]));
  MM_CHECK(conv1->SetDilation(dilation1[0], dilation1[1]));
  MM_CHECK(conv1->SetPaddingMode(magicmind::IPaddingMode::EXPLICIT));

#ifdef Setfp16conv
  MM_CHECK(conv1->SetPrecision(0, DataType::FLOAT16));
  MM_CHECK(conv1->SetPrecision(1, DataType::FLOAT16));
  MM_CHECK(conv1->SetPrecision(2, DataType::FLOAT16));
#endif

  auto conv1_output = conv1->GetOutput(0);
  PTR_CHECK(conv1_output);
  // conv output tensor datatype should be set same with bias tensor
#ifdef Setfp16conv
  MM_CHECK(conv1->SetOutputType(0, DataType::FLOAT16));
#else
  MM_CHECK(conv1->SetOutputType(0, DataType::FLOAT32));
#endif

  auto relu1 = network->AddIActivationNode(conv1_output, magicmind::IActivation::RELU);
  PTR_CHECK(relu1);
  MM_CHECK(relu1->SetOutputType(0, DataType::FLOAT32));

  // set outputs nodes
  network->MarkOutput(relu1->GetOutput(0));

  IBuilderConfig *config_ptr = CreateIBuilderConfig();
  config_ptr->SetMLUArch({"mtp_372"});
  config_ptr->ParseFromString(R"({"debug_config": {"print_ir": {"print_level": 1}}})");
  config_ptr->ParseFromString(R"({"graph_shape_mutable": false})");
  config_ptr->ParseFromString(R"({"precision_config": {"precision_mode": "force_float32"}})");

  // create model
  auto model = sample_unique_ptr<magicmind::IModel>(
      builder->BuildModel("op_amp", network.get(), config_ptr));
  PTR_CHECK(model);
  // save model to file
  MM_CHECK(model->SerializeToFile(model_name));
}

int main(int argc, char *argv[]) {
  if (argc != 2) {
    LOGINFO("Usage: %s model_name \n", argv[0]);
    return -1;
  }
  std::string model_name = argv[1];

  ConstructAddNetwork(model_name.c_str());
  return 0;
}
