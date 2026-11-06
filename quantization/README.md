# Quantization · 量化学习项目

基于自己编写的 `helper.py` 和 Notebook 导出脚本，使用普通 CPU 电脑学习线性量化、误差分析和权重量化的工程细节。

## 文件与入口

| 文件 | 用途 |
| --- | --- |
| [mini_quantization.ipynb](mini_quantization.ipynb) | 学习主入口：按主题整理的讲解、代码单元和观察任务 |
| [helper.py](helper.py) | 量化辅助模块；保留既有量化实现，新增位打包 / 解包函数 |
| [original/mini_quantization_export.py](original/mini_quantization_export.py) | 保留用户导出脚本的代码、演示顺序和原注释，作为整理前对照；空行做了归整 |
| [requirements.txt](requirements.txt) | CPU 基础实验与 Jupyter 依赖 |
| [requirements-models.txt](requirements-models.txt) | CodeGen、DETR、OPT 等可选模型实验依赖 |
| [experiments/TEMPLATE.md](experiments/TEMPLATE.md) | 可复现实验记录模板 |
| [VALIDATION.md](VALIDATION.md) | 这次整理的验证范围与结果 |
| [tests/test_bit_packing.py](tests/test_bit_packing.py) | 位打包的字节排列、往返恢复和输入边界验证 |

## 学习范围

| 阶段 | 实验 | 要回答的问题 |
| --- | --- | --- |
| 1 | 非对称 / 对称 INT8 量化 | scale、zero-point 怎样决定整数映射和重建误差？ |
| 2 | Per-tensor / Per-channel / Per-group | 量化粒度变化时，误差与 scale 元数据有什么取舍？ |
| 2.1 | 位打包 / 解包 | 怎样把低比特码值真正压入字节，并无损恢复？ |
| 3 | W8A32 与量化线性层 | 权重存储为 INT8 后，前向计算实际使用什么 dtype？ |
| 4 | 递归替换 `nn.Linear` | 替换了哪些层，为什么保留 `lm_head`？ |
| 5（可选） | CodeGen / DETR / OPT | 如何在真实模型中观察输出与权重占用变化？ |
| 6（可选） | 保存权重 / Hugging Face 上传 | 一个 state_dict 包含什么，恢复还需要什么结构与配置？ |

## 在本地运行

进入本目录，创建虚拟环境并安装基础依赖：

```bash
python -m venv .venv
```

激活环境：

| 系统 / 终端 | 命令 |
| --- | --- |
| Windows PowerShell | `.venv\Scripts\Activate.ps1` |
| Linux / macOS | `source .venv/bin/activate` |

```bash
python -m pip install -r requirements.txt
python -m jupyterlab mini_quantization.ipynb
```

选择该虚拟环境的 Python 内核，按顺序运行 Notebook。基础实验只构造小张量，不需要 GPU 或下载模型。Notebook 固定随机种子为 42，并把可选模型实验与远端上传开关默认设为 `False`。

Windows 下保留了原脚本的 `msvc_runtime` 导入，并放在 `torch` 之前；其他系统跳过该导入。依赖中使用平台条件，仅 Windows 安装对应运行库。Python 版本请以 [PyTorch 安装说明](https://pytorch.org/get-started/locally/)支持的环境为准，原脚本中的“Python 3.15”注释不代表本项目完成了该版本兼容性验证。

## 位打包与解包

Notebook 的「4.1 低比特权重打包与解包」使用 8 个 2-bit 码值，打包后为 `[228, 27]` 两个字节，再验证解包结果与输入一致。

`pack_weights` 和 `unpack_weights` 支持 1、2、4、8 bit，输入是一维 `torch.uint8`。打包输入必须在 `[0, 2**bits - 1]`，长度必须能填满字节；不自动补零。每字节按低位优先存放，输出保留输入 device。

打包是无损存储转换，不进行浮点量化或反量化。示例中的 8 B → 2 B 仅统计码值，不包含 scale 等元数据。

安装基础依赖后，在本目录运行：

```bash
python -m unittest discover -s tests -v
```

## 可选模型实验

安装额外依赖后，按需开启一个模型开关：

```bash
python -m pip install -r requirements-models.txt
```

| 开关 | 默认值 | 前提 |
| --- | --- | --- |
| `RUN_CODEGEN` | `False` | 本地已有 `Salesforce/codegen-350M-mono` 的模型与 tokenizer |
| `RUN_DETR` | `False` | 本地已有 `facebook/detr-resnet-50`；设置 `IMAGE_PATH` 指向自己的图片 |
| `RUN_OPT` | `False` | 本地已有 `facebook/opt-125m`；量化权重写入 `artifacts/` |
| `UPLOAD_TO_HF` | `False` | 已运行 OPT，填写 `HF_REPO_ID`，并在本地完成 Hugging Face 认证和目标模型仓库准备 |

模型加载保持 `local_files_only=True`，不会自动下载权重。原始导出脚本末尾含有直接调用 Hugging Face 上传的代码；整理后的 Notebook 将它改为显式开关，且不会代为创建模型仓库或执行上传。

## 本次整理的改动

- 恢复 Markdown 与 Python 代码单元，清空执行编号及输出，移除空白占位单元。
- 将分组量化、量化线性层、模型替换和三个模型示例补成独立章节，把总结放到最后。
- 为环境导入加入 Windows 判断，固定随机种子，提供模型实验和上传开关。
- 将单个线性层示例的声明改为 `in_features=8, out_features=4`，与 `(4, 8)` 权重一致；该示例仍使用原始 BF16 权重。
- 对比行通道和列通道时，分别标注 `dim=0` 与 `dim=1`，避免两行都只显示 “Per-Channel”。
- 初次归档保留了用户上传的 helper；后续增加 `pack_weights` 和 `unpack_weights`。既有函数与类保留当前仓库版本，包括 `linear_quantizaton` 这个原有拼写。

## 计算约定与工程边界

原始 helper 使用：

```text
q = clamp(round(x / scale + zero_point), qmin, qmax)
x_hat = scale * (q - zero_point)
MSE = mean((x - x_hat)^2)
```

取整发生在加 zero-point 之后。中点取整时，这与“先 round(x / scale)，再加整数 zero-point”的约定可能不同，应在以后对接其他实现时明确比较。

原始对称量化计算 `scale = max(abs(x)) / qmax`，再使用 dtype 的完整整数范围裁剪；INT8 对应 `[-128, 127]`。按该 scale 估计时不应把它与严格窄范围 `[-127, 127]` 的实现混为一谈。

`W8A16LinearLayer` 的名字沿用原实现，实际激活 dtype 由输入决定：示例中的 BF16 不是 FP16。其前向过程先把 INT8 权重转换为输入 dtype，再调用浮点 `F.linear`；本项目没有实现原生 INT8 GEMM 内核，不能从文件变小直接推断 CPU 推理加速。

不同量化方法没有固定的“高 / 中 / 最高”精度排名。比较应使用相同输入与指标，同时观察裁剪、粒度和元数据开销。DETR 示例只展示检测结果与参数 / buffer 占用差异，尚不构成准确率、延迟或峰值内存评估。

## 已知问题与下一步验收

以下为代码审查发现的待验证项，尚未修复：

- [ ] 常数张量、全零张量、全零通道 / 分组的 scale 为 0 时如何处理。
- [ ] 非对称参数估计对全正、全负输入的表示范围；zero-point 直接截断可能导致不必要的裁剪。
- [ ] `quantized_linear_W8A32_without_bias` 目前使用 `q_w * s_w + z_w`；若 `z_w` 是整数 zero-point，通用反量化应为 `(q_w - z_w) * s_w`。当前示例只传 0。
- [ ] `W8A16LinearLayer.quantize` 的零 scale、INT8 转换前裁剪、输入形状检查。
- [ ] `linear_q_symmetric_per_group` 对维数、group_size 和非连续张量的校验；现有断言顺序和 `.view()` 有适用条件。
- [ ] 逐通道 scale 的 device / dtype 一致性；当前学习范围为 CPU。
- [ ] 层替换排除项目前按叶子模块名匹配，不能直接当成完整层路径规则。
- [ ] 序列化后重建量化层结构并成功加载 state_dict；只有 `.pth` 文件并不等于可直接加载的完整 Transformers 模型仓库。

完成一个问题后，补上最小可复现输入、修复、测试和自己的解释，再勾选对应项。实验记录使用 [模板](experiments/TEMPLATE.md)，保留同一输入、参数、dtype、随机种子与实际测试命令。
