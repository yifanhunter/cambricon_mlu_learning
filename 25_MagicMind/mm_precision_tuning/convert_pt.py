import torch
import torchvision

model = torchvision.models.resnet50()
model.load_state_dict(torch.load("./resnet50-19c8e357.pth"))

model.eval()  # 使用trace之前，一定要执行model.eval()，以保证trace测试而不是训练的逻辑
input = torch.randn(1, 3, 224, 224)
trace_model = torch.jit.trace(model, input)
torch.jit.save(trace_model, "resnet50_mm.pt")
