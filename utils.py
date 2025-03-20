import cv2
import numpy as np
import math
import torch
import torch.nn as nn

class AverageMeter(object):
    """
    Computes and stores the average and current value.

    """
    def __init__(self):
        self.reset()

    def reset(self):
        """
        Reset the meter's values.

        """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """
        Update the meter with a new value.

        Args:
            val (float): New value to update the meter.
            n (int): Number of elements represented by the value.

        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def normalize_list(input_list):
    # 计算列表的总和
    total_sum = sum(input_list)
    # 归一化列表
    normalized_list = [x / total_sum for x in input_list]
    return normalized_list


def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def softmax(logits):
    exp_logits = [math.exp(logit) for logit in logits]
    sum_exp_logits = sum(exp_logits)
    return [exp_logit / sum_exp_logits for exp_logit in exp_logits]

def update_regressor_weights(regressor_losses, length):
    """
    更新回归器权重。
    
    参数:
    regressor_losses (dict): 键为回归器名称，值为对应回归器的总损失。
    length (int): 样本总数。
    
    返回:
    dict: 更新后的回归器权重。
    """
    # 将所有总损失除以样本总数
    normalized_losses = regressor_losses / length
    
    # 使用sigmoid函数进行归一化，并进行1-sigmoid操作
    updated_weights = 1 - sigmoid(normalized_losses)
    
    return updated_weights


def update_sample_weights(weight_of_dict, regressor_weight, loss_of_dict):
    """
    更新样本权重。
    
    参数:
    weight_of_dicts (dict): 键为回归器名称，值为样本权重字典。
    regressor_weights (dict): 键为回归器名称，值为对应回归器的权重。
    loss_of_dicts (dict): 键为回归器名称，值为样本损失字典。
    
    返回:
    dict: 更新后的样本权重字典。
    """
    # 提取当前回归器的样本权重和损失
    sample_weights = weight_of_dict
    sample_losses = loss_of_dict
    
    # 找到最大损失
    max_loss = max(sample_losses.values())
    
    # 更新损失字典，将所有损失除以最大损失
    normalized_losses = {sample: loss / max_loss for sample, loss in sample_losses.items()}
    
    # 更新样本权重
    updated_sample_weights = {}
    for sample, loss in normalized_losses.items():
        updated_weight = sample_weights[sample] * (regressor_weight ** (1 - loss))
        updated_sample_weights[sample] = updated_weight
    
    # 对新的样本权重进行归一化，确保权重和为1
    total_weight = sum(updated_sample_weights.values())
    normalized_sample_weights = {sample: weight / total_weight for sample, weight in updated_sample_weights.items()}
    
    return normalized_sample_weights


def move_to_gpu(data, device):
    if isinstance(data, dict):
        return {k: move_to_gpu(v, device) for k, v in data.items()}
    elif isinstance(data, torch.Tensor):
        return data.to(device)
    else:
        return data


def save_image(face_name, img):
    """
    保存图像为文件，支持调整通道顺序和数据类型。
    
    参数：
    - face_name: str，保存图像的文件路径
    - img: numpy.ndarray，输入的图像数据 (来自 data['face'][k].cpu().numpy())
    """
    # 如果形状为 (C, H, W)，调整为 (H, W, C)
    if len(img.shape) == 3 and img.shape[0] in [1, 3, 4]:  # 检查是否是 (C, H, W)
        img = img.transpose(1, 2, 0)  # 转换为 (H, W, C)

    # 如果形状为 (1, H, W)，去掉多余的维度
    if len(img.shape) == 3 and img.shape[0] == 1:
        img = img.squeeze(0)  # 转换为 (H, W)

    # 转换为 uint8 类型并归一化
    img = (img * 255.0).clip(0, 255).astype('uint8')

    # 如果是彩色图像，转换为 BGR 格式
    if len(img.shape) == 3 and img.shape[2] == 3:  # 检查是否是 RGB 图像
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # 保存图像
    cv2.imwrite(face_name, img)