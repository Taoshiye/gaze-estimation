import cv2
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

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


def gaze_to_vector(gaze):
    """
    将二维视线角 [yaw, pitch] 转换为三维单位向量。
    支持批量输入，输入形状为 [B, 2]，输出形状为 [B, 3]。
    """
    # 将角度转换为弧度
    yaw = torch.deg2rad(gaze[:, 0])  # 提取 yaw 并转换为弧度
    pitch = torch.deg2rad(gaze[:, 1])  # 提取 pitch 并转换为弧度

    # 计算三维单位向量
    x = torch.cos(pitch) * torch.cos(yaw)
    y = torch.cos(pitch) * torch.sin(yaw)
    z = torch.sin(pitch)

    # 堆叠成 [B, 3] 的形状
    return torch.stack([x, y, z], axis=1)

def compute_rotation_matrix(g1, g2):
    """
    通过两个三维单位向量 g1 和 g2 计算旋转矩阵 R。
    支持批量输入，输入形状为 [B, 2]，输出形状为 [B, 3, 3]。
    """
    # 将 g1 和 g2 转换为三维单位向量
    v1 = gaze_to_vector(g1)  # 形状 [B, 3]
    v2 = gaze_to_vector(g2)  # 形状 [B, 3]

    # 计算旋转轴
    k = torch.cross(v1, v2, dim=1)  # 形状 [B, 3]
    k_norm = torch.norm(k, dim=1, keepdim=True)  # 计算范数，形状 [B, 1]
    k = k / (k_norm + 1e-14)  # 归一化，避免除以零

    # 计算旋转角度
    dot_product = torch.sum(v1 * v2, dim=1)  # 点积，形状 [B]
    theta = torch.acos(torch.clamp(dot_product, -1.0, 1.0))  # 形状 [B]

    # 使用罗德里格斯公式计算旋转矩阵
    K = torch.zeros((g1.shape[0], 3, 3), device=g1.device)  # 形状 [B, 3, 3]
    K[:, 0, 1] = -k[:, 2]
    K[:, 0, 2] = k[:, 1]
    K[:, 1, 0] = k[:, 2]
    K[:, 1, 2] = -k[:, 0]
    K[:, 2, 0] = -k[:, 1]
    K[:, 2, 1] = k[:, 0]

    I = torch.eye(3, device=g1.device).unsqueeze(0).repeat(g1.shape[0], 1, 1)  # 形状 [B, 3, 3]
    sin_theta = theta.unsqueeze(1).unsqueeze(1)  # 形状 [B, 1, 1]
    cos_theta = torch.cos(theta).unsqueeze(1).unsqueeze(1)  # 形状 [B, 1, 1]

    # 计算旋转矩阵 R
    R = I + sin_theta * K + (1 - cos_theta) * torch.bmm(K, K)  # 形状 [B, 3, 3]

    return R


def compute_rotation_degree(R, g1):
    """
    通过旋转矩阵 R 和原始视线角 g1 求出旋转后的视线角 g2。
    存在bug：旋转角如果超过范围，计算会出错。yaw[-180,180],pitch[-90,90]
    """
    # 将 g1 转换为三维单位向量
    v1 = gaze_to_vector(g1)  # 形状 [B, 3]

    # 验证旋转矩阵 R 是否正确将 v1 旋转到 v2
    v2_rotated = torch.bmm(R, v1.unsqueeze(-1)).squeeze(-1)  # 形状 [B, 3]
    # 归一化 v2_rotated（单位向量）
    v2_rotated_norm = F.normalize(v2_rotated, p=2, dim=1)  # p=2 表示 L2 范数，dim=1 对每个样本的 3D 向量归一化

    # 从旋转后的向量中提取新的 yaw 和 pitch
    pitch2 = torch.asin(v2_rotated_norm[:, 2])  # 形状 [B]
    cos_pitch2 = torch.cos(pitch2) + 1e-10
    yaw2 = torch.atan2(v2_rotated_norm[:, 1]/cos_pitch2, v2_rotated_norm[:, 0]/cos_pitch2)

    g2_rot = torch.rad2deg(torch.stack([yaw2, pitch2], dim=1))  # 形状 [B, 2]

    # # 确保角度在 -180 到 180 之间
    # g2_rot = (g2_rot + 180) % 360 - 180

    return g2_rot


def inverse_rotation(R, g2):
    """
    通过旋转矩阵 R 和旋转后的视线角 g2 反向求出原始视线角 g1。
    """
    # 将 g2 转换为三维单位向量
    v2 = gaze_to_vector(g2)  # 形状 [B, 3]

    # 使用旋转矩阵 R 的逆矩阵将 v2 旋转回 v1
    R_inv = torch.inverse(R)  # 计算逆矩阵
    R_transpose = R.transpose(1, 2) # 计算 R 的转置
    v1_rotated = torch.bmm(R_transpose, v2.unsqueeze(-1)).squeeze(-1)  # 形状 [B, 3]
    # 归一化 v1_rotated（单位向量）
    v1_rotated_norm = F.normalize(v1_rotated, p=2, dim=1)  # p=2 表示 L2 范数，dim=1 对每个样本的 3D 向量归一化

    # 从旋转后的向量中提取新的 yaw 和 pitch
    pitch1 = torch.asin(v1_rotated_norm[:, 2])  # 形状 [B]
    cos_pitch1 = torch.cos(pitch1) + 1e-10
    yaw1 = torch.atan2(v1_rotated_norm[:, 1]/cos_pitch1, v1_rotated_norm[:, 0]/cos_pitch1)

    # 将弧度转换回角度
    g1_inverted = torch.rad2deg(torch.stack([yaw1, pitch1], dim=1))  # 形状 [B, 2]

    return g1_inverted


if __name__ == "__main__":
    # 示例使用
    g1 = torch.tensor([[179, 86], [10, 20], [30, 40]])  # 初始视线角 [yaw, pitch]，形状 [3, 2]
    g2 = torch.tensor([[181, 92], [30, 40], [60, 70]])  # 旋转后的视线角 [yaw + theta, pitch + theta]，形状 [3, 2]

    # 计算旋转矩阵 R
    R = compute_rotation_matrix(g1, g2)
    g2_rotated = compute_rotation_degree(R, g1)
    g1_inverted = inverse_rotation(R, g2)

    print("旋转矩阵 R:")
    print(R)
    print("\n旋转前的视线角 g1:")
    print(g1)
    print("\n旋转后的视线角 g2:")
    print(g2)
    print("\n旋转后的视线角 g2_rotated:")
    print(g2_rotated)
    print("\n反向求出的视线角 g1_inverted:")
    print(g1_inverted)
