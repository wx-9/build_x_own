"""
量化器 helper 模块
从 miniquantizarion.ipynb 抽取的函数集合
"""

import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
#  量化 / 反量化
# ============================================================

def linear_q_with_scale_and_zero_point(tensor, scale, zero_point, dtype=torch.int8):
    """带 scale 和 zero_point 的线性量化"""
    scaled_and_shifted_tensor = tensor / scale + zero_point
    rounded_tensor = torch.round(scaled_and_shifted_tensor)
    q_min = torch.iinfo(dtype).min
    q_max = torch.iinfo(dtype).max
    q_tensor = rounded_tensor.clamp(q_min, q_max).to(dtype)
    return q_tensor


def linear_dequantization(q_tensor, scale, zero_point):
    """线性反量化"""
    return scale * (q_tensor.float() - zero_point)


def get_q_scale_and_zero_point(tensor, dtype=torch.int8):
    """计算非对称量化的 scale 和 zero_point"""
    r_min = tensor.min().item()  
    r_max = tensor.max().item()
    q_min = torch.iinfo(dtype).min
    q_max = torch.iinfo(dtype).max

    scale = (r_max - r_min) / (q_max - q_min)
    zero_point = q_min - (r_min / scale)
    if zero_point > q_max:
        zero_point = q_max
    elif zero_point < q_min:
        zero_point = q_min
    else:
        zero_point = int(round(zero_point))

    return scale, zero_point


def linear_quantizaton(tensor, dtype=torch.int8):
    """非对称线性量化 (一站式)"""
    scale, zero_point = get_q_scale_and_zero_point(tensor, dtype)
    quantizated_tensor = linear_q_with_scale_and_zero_point(tensor, scale, zero_point, dtype)
    return quantizated_tensor, scale, zero_point


def get_q_scale_symmetric(tensor, dtype=torch.int8):
    """计算对称量化的 scale (zero_point=0)"""
    r_max = tensor.abs().max().item()
    q_max = torch.iinfo(dtype).max
    scale = r_max / q_max
    return scale


def linear_q_symmetric(r_tensor, dtype=torch.int8):
    """对称线性量化 (zero_point=0)"""
    scale = get_q_scale_symmetric(r_tensor, dtype)
    quantizated_tensor = linear_q_with_scale_and_zero_point(r_tensor, scale, 0, dtype)
    return quantizated_tensor, scale

def linear_q_symmetric_per_channel(tensor, dim, dtype=torch.int8):
    """逐通道对称量化: 沿 dim 维度每条计算独立 scale"""
    output_dim = tensor.shape[dim]
    scale = torch.zeros(output_dim)
    for index in range(output_dim):
        sub_tensor = tensor.select(dim, index)
        scale[index] = get_q_scale_symmetric(sub_tensor, dtype)
    scale_shape = [1] * tensor.dim()
    scale_shape[dim] = -1
    scale = scale.view(scale_shape)
    quantized_tensor = linear_q_with_scale_and_zero_point(tensor, scale, zero_point=0, dtype=dtype)
    return quantized_tensor, scale

def linear_q_symmetric_per_group(tensor, group_size, dtype = torch.int8):
    t_shape = tensor.shape
    assert t_shape[1] % group_size == 0
    assert tensor.dim() == 2
    tensor = tensor.view(-1, group_size)
    quantized_tensor, scale = linear_q_symmetric_per_channel(tensor, dim = 0, dtype = dtype)
    quantized_tensor = quantized_tensor.view(t_shape)
    return quantized_tensor, scale

def linear_dequantization_per_group(quantized_tensor, scale, group_size):
    q_shape = quantized_tensor.shape
    quantized_tensor = quantized_tensor.view(-1, group_size)
    dequantized_tensor = linear_dequantization(quantized_tensor, scale, 0)
    dequantized_tensor = dequantized_tensor.view(q_shape)
    return dequantized_tensor
# ============================================================
#  可视化
# ============================================================

def plot_quantization_errors(tensor, quantizated_tensor, dequantizated_tensor):
    """可视化量化误差: 原始 / 量化 / 反量化 / 绝对误差 四联图"""
    # 1. 转换为 NumPy
    original = tensor.detach().cpu().numpy()
    quantized = quantizated_tensor.detach().cpu().numpy()
    dequantized = dequantizated_tensor.detach().cpu().numpy()

    # 2. 检查形状
    if not (original.shape == quantized.shape == dequantized.shape):
        raise ValueError("All tensors must have the same shape")

    # 3. 支持一维和二维 Tensor
    if original.ndim == 1:
        original = original.reshape(1, -1)
        quantized = quantized.reshape(1, -1)
        dequantized = dequantized.reshape(1, -1)
    elif original.ndim != 2:
        raise ValueError("Only 1D or 2D tensors are supported")

    if original.size == 0:
        raise ValueError("Input tensors must not be empty")

    # 4. 计算误差
    error = np.abs(
        original.astype(np.float64) -
        dequantized.astype(np.float64)
    )

    matrices = [original, quantized, dequantized, error]
    titles = [
        "Original (FP32)",
        "Quantized (INT8)",
        "Dequantized (FP32)",
        "Absolute Error"
    ]

    rows, cols = original.shape

    fig, axes = plt.subplots(
        1, 4,
        figsize=(max(16, cols * 2.8), max(4, rows * 0.65 + 1))
    )

    # 原始矩阵和反量化矩阵统一颜色范围
    vmin = min(original.min(), dequantized.min())
    vmax = max(original.max(), dequantized.max())

    for idx, (ax, matrix, title) in enumerate(
        zip(axes, matrices, titles)
    ):
        kwargs = {}

        if idx in (0, 2):
            kwargs = {"vmin": vmin, "vmax": vmax}

        im = ax.imshow(
            matrix,
            cmap="viridis",
            aspect="equal",
            **kwargs
        )

        ax.set_title(title)

        # 显示每个元素的值
        for i in range(rows):
            for j in range(cols):
                value = matrix[i, j]

                if idx == 1:
                    label = f"{int(value)}"
                else:
                    label = f"{float(value):.3f}"

                rgba = im.cmap(im.norm(value))
                brightness = (
                    0.299 * rgba[0] +
                    0.587 * rgba[1] +
                    0.114 * rgba[2]
                )

                color = "black" if brightness > 0.5 else "white"

                ax.text(
                    j, i, label,
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=8
                )

        ax.set_xticks(np.arange(cols))
        ax.set_yticks(np.arange(rows))

        # 单元格网格线
        ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.5)
        ax.tick_params(which="minor", bottom=False, left=False)

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()

    print(f"Max Absolute Error : {error.max():.6f}")
    print(f"Mean Absolute Error: {error.mean():.6f}")
    print(f"MSE                : {np.mean(error ** 2):.6f}")


# ============================================================
#  目标检测可视化
# ============================================================

def plot_detection_results(image, results, id2label, threshold=0.5, alpha=0.25):
    """在原图上绘制检测框、标签和置信度，返回标注后的 PIL Image

    Args:
        image: PIL Image, 原始图片
        results: processor.post_process_object_detection 的输出
        id2label: dict, label_id -> label name
        threshold: 置信度阈值
        alpha: 框内填充透明度 (0=透明, 1=不透明)
    """
    from PIL import Image as PILImage, ImageDraw, ImageFont

    annotated = image.copy().convert("RGBA")
    w, h = annotated.size
    min_dim = min(w, h)

    # 自适应线宽: 细线避免遮挡小目标
    line_width = max(1, min_dim // 400)
    font_size = max(12, min_dim // 45)

    try:
        font = ImageFont.truetype("arial.ttf", size=font_size)
    except (OSError, IOError):
        font = ImageFont.load_default()

    palette = [
        (230, 25, 75), (60, 180, 75), (255, 225, 25), (67, 99, 216),
        (245, 130, 49), (145, 30, 180), (66, 212, 244), (240, 50, 230),
        (191, 239, 69), (250, 190, 190), (70, 153, 144), (220, 159, 255),
        (154, 99, 36), (128, 0, 0), (170, 255, 195), (128, 128, 0),
        (0, 0, 117), (169, 169, 169), (230, 190, 168), (255, 216, 177),
        (61, 90, 108),
    ]

    # 用两个图层: fill_layer 做半透明填充, draw_layer 画线和文字
    fill_layer = PILImage.new("RGBA", annotated.size, (0, 0, 0, 0))
    fill_draw = ImageDraw.Draw(fill_layer)

    for result in results:
        for score, label_id, box in zip(result["scores"], result["labels"], result["boxes"]):
            if score.item() < threshold:
                continue
            label = id2label[label_id.item()]
            x1, y1, x2, y2 = [round(v, 2) for v in box.tolist()]
            r, g, b = palette[label_id.item() % len(palette)]

            # 半透明填充
            fill_draw.rectangle([x1, y1, x2, y2], fill=(r, g, b, int(255 * alpha)))

    # 合并填充层到原图
    annotated = PILImage.alpha_composite(annotated, fill_layer)
    draw = ImageDraw.Draw(annotated)

    # 第二遍: 画框和标签 (在填充之上)
    for result in results:
        for score, label_id, box in zip(result["scores"], result["labels"], result["boxes"]):
            if score.item() < threshold:
                continue
            label = id2label[label_id.item()]
            x1, y1, x2, y2 = [round(v, 2) for v in box.tolist()]
            r, g, b = palette[label_id.item() % len(palette)]
            color_str = f"#{r:02X}{g:02X}{b:02X}"

            # 细描边框
            draw.rectangle([x1, y1, x2, y2], outline=color_str, width=line_width)

            # 标签
            text = f"{label} {score:.2f}"
            bbox = draw.textbbox((x1, y1), text, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            pad_x, pad_y = 3, 1
            bg_w = text_w + pad_x * 2
            bg_h = text_h + pad_y * 2

            if y1 - bg_h - 2 >= 0:
                bg_y1 = y1 - bg_h - 2
                bg_y2 = y1 - 2
            else:
                bg_y1 = y1
                bg_y2 = y1 + bg_h

            draw.rectangle([x1, bg_y1, x1 + bg_w, bg_y2], fill=(r, g, b, 220))
            draw.text((x1 + pad_x, bg_y1 + pad_y), text, fill="white", font=font)

    return annotated.convert("RGB")


# ============================================================
#  W8A32 / W8A16 量化线性层
# ============================================================

def quantized_linear_W8A32_without_bias(input, q_w, s_w, z_w):
    """W8A32 量化线性层推理 (无 bias)"""
    assert input.dtype == torch.float32
    assert q_w.dtype == torch.int8
    dequantized_tensor = q_w.to(torch.float32) * s_w + z_w
    output = torch.nn.functional.linear(input, dequantized_tensor)
    return output


def w8_a16_forward(weight, input, scales, bias=None):
    """W8A16 前向传播: INT8 权重 + FP16 激活"""
    casted_weights = weight.to(input.dtype)
    output = F.linear(input, casted_weights) * scales
    if bias is not None:
        output = output + bias
    return output


class W8A16LinearLayer(nn.Module):
    """INT8 权重 + FP16 激活的量化线性层"""
    def __init__(self, in_features, out_features, bias=True, dtype=torch.float32):
        super().__init__()
        self.register_buffer(
            "int8_weights",
            torch.randint(-128, 127, (out_features, in_features), dtype=torch.int8)
        )
        self.register_buffer(
            "scales",
            torch.randn(out_features, dtype=dtype)
        )
        if bias:
            self.register_buffer("bias",
                                 torch.randn((1, out_features), dtype=dtype))
        else:
            self.bias = None

    def quantize(self, weight):
        w_fp32 = weight.clone().to(torch.float32)
        scales = w_fp32.abs().max(dim=-1).values / 127
        scales = scales.to(weight.dtype)
        int8_weights = torch.round(weight / scales.unsqueeze(1)).to(torch.int8)
        self.int8_weights = int8_weights
        self.scales = scales

    def forward(self, input):
        return w8_a16_forward(self.int8_weights, input, self.scales, self.bias)


# ============================================================
#  模型层替换
# ============================================================

def replace_linear_with_target(module, target_class, module_name_to_exclude):
    """递归替换 module 中的 nn.Linear 为 target_class (不量化)"""
    for name, child in module.named_children():
        if isinstance(child, nn.Linear) and not any([x == name for x in module_name_to_exclude]):
            old_bias = child.bias
            new_module = target_class(child.in_features, child.out_features,
                                      old_bias is not None, child.weight.dtype)
            setattr(module, name, new_module)
            if old_bias is not None:
                getattr(module, name).bias = old_bias
        else:
            replace_linear_with_target(child, target_class, module_name_to_exclude)


def replace_linear_with_target_and_quantize(module, target_class, module_name_to_exclude):
    """递归替换 module 中的 nn.Linear 为 target_class 并量化权重"""
    for name, child in module.named_children():
        if isinstance(child, nn.Linear) and not any([x == name for x in module_name_to_exclude]):
            old_bias = child.bias
            old_weight = child.weight
            new_module = target_class(child.in_features, child.out_features,
                                      old_bias is not None, child.weight.dtype)
            setattr(module, name, new_module)
            getattr(module, name).quantize(old_weight)
            if old_bias is not None:
                getattr(module, name).bias = old_bias
        else:
            replace_linear_with_target_and_quantize(child, target_class, module_name_to_exclude)


class DummyModel(torch.nn.Module):
    """测试用模型: 包含 embedding + 2 个 linear + lm_head"""
    def __init__(self):
        super().__init__()
        self.emb = torch.nn.Embedding(1, 1)
        self.linear_1 = nn.Linear(1, 1)
        self.linear_2 = nn.Linear(1, 1)
        self.lm_head = nn.Linear(1, 1, bias=False)
