#!/usr/bin/env python
# coding: utf-8

# # Mini Quantization Demo
#
# LLM 量化入门 demo — 纯 PyTorch 实现
#
# 覆盖内容:
# 1. 非对称线性量化 (INT8)
# 2. 对称线性量化 (INT8)
# 3. 逐通道对称量化 (Per-Channel)
# 4. 量化误差可视化
#
# 所有函数在 `helper.py` 中，notebook 只做调用演示。

# ## 0. 环境准备
# In[70]:
import msvc_runtime  # Python 3.15 Windows 修复, 必须在 torch 之前
import torch
import importlib
import helper
from torch import nn
importlib.reload(helper)
import torch.nn.functional as F
from helper import (
    linear_q_with_scale_and_zero_point,
    linear_dequantization,
    get_q_scale_and_zero_point,
    linear_quantizaton,
    get_q_scale_symmetric,
    linear_q_symmetric,
    linear_q_symmetric_per_channel,
    linear_q_symmetric_per_group,
    linear_dequantization_per_group,
    plot_quantization_errors,
    quantized_linear_W8A32_without_bias,
    w8_a16_forward,
    W8A16LinearLayer,
    replace_linear_with_target,
    replace_linear_with_target_and_quantize,
    DummyModel,
    plot_detection_results,
)

# ## 1. 非对称线性量化 (INT8)
#
# 原理:
# - `scale = (r_max - r_min) / (q_max - q_min)`
# - `zero_point = round(q_min - r_min / scale)`
# - 量化: `q = clamp(round(x / scale + zero_point), q_min, q_max)`
# - 反量化: `x ≈ scale * (q - zero_point)`
#
# 适合场景: 激活值 (ReLU 后全正, 分布不对称)
# In[2]:
r_tensor = torch.randn((4, 4))
r_tensor
# In[3]:
quantized_tensor, scale, zero_point = linear_quantizaton(r_tensor)
print(f"scale:       {scale}")
print(f"zero_point:  {zero_point}")
print(f"quantized:\n{quantized_tensor}")
# In[4]:
dequantized_tensor = linear_dequantization(quantized_tensor, scale, zero_point)
dequantized_tensor
# In[5]:
(r_tensor - dequantized_tensor).square().mean()
# In[6]:
plot_quantization_errors(r_tensor, quantized_tensor, dequantized_tensor)

# ## 2. 对称线性量化 (INT8)
#
# 原理:
# - `scale = max(|x|) / q_max`  (zero_point = 0)
# - 量化: `q = clamp(round(x / scale), -q_max, q_max)`
# - 反量化: `x ≈ scale * q`
#
# 适合场景: 权重 (均值 ≈ 0, 分布对称)
#
# 优点: 省掉 zero_point 存储, 计算更简单
# In[7]:
test_tensor = torch.randn((4, 4))
print(f"symmetric scale: {get_q_scale_symmetric(test_tensor)}")
# In[8]:
quantizated_tensor, scale = linear_q_symmetric(test_tensor)
quantizated_tensor
# In[9]:
dequantized_tensor = linear_dequantization(quantizated_tensor, scale, zero_point=0)
dequantized_tensor
# In[10]:
plot_quantization_errors(test_tensor, quantizated_tensor, dequantized_tensor)

# ### 对称量化在大动态范围数据上的表现
#
# 当数据范围很大且不均匀时, 对称量化的误差会明显增大。
# In[11]:
test_tensor = torch.tensor(
    [[191.6, -13.5, 728.6],
    [92.14, 295.5, -184],
    [0, 684.6, 245.5]]
)
# In[12]:
quantizated_tensor, scale = linear_q_symmetric(test_tensor)
dequantized_tensor = linear_dequantization(quantizated_tensor, scale, zero_point=0)
plot_quantization_errors(test_tensor, quantizated_tensor, dequantized_tensor)

# ## 3. 逐通道对称量化 (Per-Channel)
#
# 原理: 对每个通道 (行/列) 独立计算 scale, 而非全局统一 scale。
#
# 优点: 当不同通道动态范围差异大时, 精度远优于逐 tensor 量化。
#
# LLM 权重量化常用这种方式 (沿 output channel 维度)。
# In[13]:
# 用大动态范围数据, 对比 per-tensor vs per-channel
test_tensor = torch.tensor(
    [[191.6, -13.5, 728.6],
    [92.14, 295.5, -184],
    [0, 684.6, 245.5]]
)

# per-tensor (全局一个 scale)
q_per_tensor, s_tensor = linear_q_symmetric(test_tensor)
dq_per_tensor = linear_dequantization(q_per_tensor, s_tensor, zero_point=0)
mse_per_tensor = (test_tensor - dq_per_tensor).square().mean()
print(f"Per-Tensor MSE:  {mse_per_tensor.item():.4f}")

# per-channel (每行独立 scale)
q_per_channel, s_channel = linear_q_symmetric_per_channel(test_tensor, dim=0)
dq_per_channel = linear_dequantization(q_per_channel, s_channel, zero_point=0)
mse_per_channel = (test_tensor - dq_per_channel).square().mean()
print(f"Per-Channel MSE: {mse_per_channel.item():.4f}")

# per-channel (每列独立scale)
q_per_channel, s_channel = linear_q_symmetric_per_channel(test_tensor, dim=1)
dq_per_channel = linear_dequantization(q_per_channel, s_channel, zero_point=0)
mse_per_channel = (test_tensor - dq_per_channel).square().mean()
print(f"Per-Channel MSE: {mse_per_channel.item():.4f}")
# In[14]:
plot_quantization_errors(test_tensor, q_per_channel, dq_per_channel)

# ## 4. 总结
#
# | 方法 | scale | zero_point | 适合场景 | 精度 |
# |------|-------|------------|----------|------|
# | 非对称量化 | `(r_max-r_min)/(q_max-q_min)` | ≠0 | 激活值 (分布不对称) | 高 |
# | 对称量化 | `max(\|x\|)/q_max` | 0 | 权重 (分布对称) | 中 |
# | 逐通道量化 | 每通道独立 | 0 | LLM 权重 | 最高 |
#
# 核心思想: **量化 = 把连续浮点映射到有限离散电平, 用更少 bit 近似表示原始数据**
#
# 误差来源: 舍入 + 截断 → 可通过 QAT (量化感知训练) 进一步缓解。
# In[15]:
# linear_q_symmetric_per_group -> helper.py
# In[16]:
# linear_dequantization_per_group -> helper.py
# In[17]:
test_tensor = torch.rand((6,6))
# In[18]:
group_size = 6
# In[19]:
quantized_tensor, scale = linear_q_symmetric_per_group(test_tensor, group_size)
# In[20]:
dequantized_tensor = linear_dequantization_per_group(quantized_tensor, scale, group_size)
# In[21]:
plot_quantization_errors(test_tensor, quantized_tensor, dequantized_tensor)
# In[22]:
# quantized_linear_W8A32_without_bias -> helper.py
# In[23]:
input = torch.tensor([1, 2, 3], dtype = torch.float32)
# In[24]:
weight = torch.tensor([[-2, -1.13, 0.42],
                      [-1.51, 0.25, 1.62],
                      [0.12, 1.35, 2.15]])
# In[25]:
q_w, s_w = linear_q_symmetric(weight)
# In[26]:
q_w
# In[27]:
s_w
# In[28]:
output = quantized_linear_W8A32_without_bias(input, q_w, s_w, 0)
# In[29]:
print(f"This is the W8A32 output: {output}")
# In[30]:
fp32_output =  torch.nn.functional.linear(input, weight)
# In[31]:
print(f"This is the output if we don't quantize: {fp32_output}")
# In[32]:
# w8_a16_forward -> helper.py
# In[33]:
# W8A16LinearLayer -> helper.py
# In[34]:
module = W8A16LinearLayer(4, 8)
# In[35]:
random_matrix = torch.randn((4, 8), dtype = torch.bfloat16)
# In[36]:
random_matrix
# In[37]:
module.quantize(random_matrix)
# In[38]:
module.int8_weights
# In[39]:
module.scales.shape
# In[40]:
module.int8_weights.shape
# In[41]:
dequantized_weights = module.int8_weights*module.scales.unsqueeze(1)
# In[42]:
dequantized_weights
# In[43]:
(random_matrix - dequantized_weights).square().mean()
# In[44]:
# replace_linear_with_target -> helper.py
# In[45]:
# DummyModel -> helper.py
# In[46]:
model_1 = DummyModel()
model_2 = DummyModel()
# In[47]:
replace_linear_with_target(model_1, W8A16LinearLayer, ["lm_head"])
# In[48]:
print(model_1)
# In[49]:
# replace_linear_with_target_and_quantize -> helper.py
# In[50]:
model_3 = DummyModel()
# In[51]:
replace_linear_with_target_and_quantize(model_3, W8A16LinearLayer, ["lm_head"])
print(model_3)
# In[52]:
import msvc_runtime
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
model_id = "Salesforce/codegen-350M-mono"
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype = torch.bfloat16, low_cpu_mem_usage = True, local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
# In[53]:
pipe = pipeline("text-generation", model = model, tokenizer = tokenizer)
# In[54]:
print(pipe("def hello_word():", max_new_tokens = 20, do_sample = False))
# In[ ]:
# In[84]:
replace_linear_with_target_and_quantize(model, W8A16LinearLayer, ["lm_head"])
# In[ ]:
# In[85]:
print(pipe("def hello_word():", max_new_tokens = 20, do_sample = False)[0]["generated_text"])
# In[72]:
from transformers import AutoImageProcessor, AutoModelForObjectDetection
from PIL import Image
import timm
model_id = "facebook/detr-resnet-50"

processor = AutoImageProcessor.from_pretrained(model_id, local_files_only=True)
model = AutoModelForObjectDetection.from_pretrained(model_id, local_files_only=True)
# In[73]:
previous_memory_footprint = model.get_memory_footprint()
previous_memory_footprint
# In[87]:
image = Image.open("dinner.jpg")
inputs = processor(images = image, return_tensors = "pt")
with torch.no_grad():
    outputs = model(**inputs)
results = processor.post_process_object_detection(
    outputs, threshold=0.5, target_sizes=torch.tensor([image.size[::-1]])
  )
annotated_image = plot_detection_results(image, results, model.config.id2label, threshold=0.5)
annotated_image
# In[ ]:
# In[86]:
replace_linear_with_target_and_quantize(model, W8A16LinearLayer, ["0", "1", "2", "class_labels_classifier"])
# In[ ]:
# In[90]:
new_memory_footprint = model.get_memory_footprint()
# In[91]:
new_memory_footprint
# In[92]:
previous_memory_footprint - new_memory_footprint
# In[93]:
import msvc_runtime
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "facebook/opt-125m"

tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
      model_id,
      dtype=torch.float32,        # 125M 小模型用 FP32 即可
      local_files_only=True,      # 跳过网络，用本地缓存
  )
# In[94]:
replace_linear_with_target_and_quantize(model, W8A16LinearLayer, ["lm_head"])
# In[96]:
quantized_state_dict = model.state_dict()
torch.save(quantized_state_dict, "quantized_state_dict.pth")
# In[97]:
from huggingface_hub import HfApi, create_repo
YOUR_HF_UERNAME = "Yotoio"
your_repo_id = f"{YOUR_HF_UERNAME}/opt-125m-quantized-dlai"

api = HfApi()

# create_repo(your_repo_id)

api.upload_file(
  path_or_fileobj="quantized_state_dict.pth",
  path_in_repo="quantized_state_dict.pth",
  repo_id=your_repo_id,
  repo_type="model",
)
# In[ ]:
