import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2


def rotategazewithmatrix(g1, theta, axis='z'):
    """
    通过三维旋转矩阵将二维视线角 g1 [yaw, pitch] 转换为 g2 [yaw theta, pitch theta]。
    参数:
    g1 (list or np.array): 初始视线角，格式为 [yaw, pitch]，单位为度。
    theta (float): 旋转角度，单位为度，范围为 -15 到 15。
    axis (str): 旋转轴，'z' 表示绕 z 轴旋转，'y' 表示绕 y 轴旋转。
    返回:
    g2 (np.array): 旋转后的视线角，格式为 [yaw, pitch]，单位为度。
    """
    # 将角度转换为弧度
    yaw, pitch = np.radians(g1)
    thetarad = np.radians(theta)

    # 定义旋转矩阵
    if axis == 'z':
        # 将 g1 转换为三维单位向量
        v1 = np.array([
            np.cos(pitch) * np.cos(yaw),
            np.cos(pitch) * np.sin(yaw),
            np.sin(pitch)
        ])

        R = np.array([
            [np.cos(thetarad), -np.sin(thetarad), 0],
            [np.sin(thetarad), np.cos(thetarad), 0],
            [0, 0, 1]
        ])

        # 应用旋转矩阵
        v2 = np.dot(R, v1)

        # 从旋转后的向量中提取新的 yaw 和 pitch
        pitch2 = pitch # 绕z轴旋转pitch角不变
        yaw2 = np.arctan2(v2[1]/np.cos(pitch2), v2[0]/np.cos(pitch2))
        # yaw2 = np.arctan2(v2[1], v2[0])
        # pitch2 = np.arcsin(v2[2])
    elif axis == 'y':
        # 将 g1 转换为三维单位向量
        v1 = np.array([
            np.cos(yaw) * np.sin(pitch),
            np.sin(yaw),
            np.cos(yaw) * np.cos(pitch)
        ])

        R = np.array([
            [np.cos(thetarad), 0, np.sin(thetarad)],
            [0, 1, 0],
            [-np.sin(thetarad), 0, np.cos(thetarad)]
        ])

        # 应用旋转矩阵
        v2 = np.dot(R, v1)

        # 从旋转后的向量中提取新的 yaw 和 pitch
        yaw2 = yaw
        pitch2 = np.arctan2(v2[0]/np.cos(yaw2), v2[2]/np.cos(yaw2))
        # yaw2 = np.arcsin(v2[1])
        # pitch2 = np.arctan2(v2[0], v2[2])
    else:
        raise ValueError("axis must be 'z' or 'y'")

    # 将弧度转换回角度
    g2 = np.degrees([yaw2, pitch2])

    # # 确保角度在 -180 到 180 之间
    # g2 = (g2 + 180) % 360 - 180

    return g2


"""利用世界坐标系"""
def gaze_to_yaw_vector(gaze):
    """
    将二维视线角 [yaw, pitch] 转换为三维单位向量。
    支持批量输入，输入形状为 [B, 2]，输出形状为 [B, 3]。
    """
    yaw = gaze[:, 0]  # 提取 yaw 并转换为弧度
    pitch = gaze[:, 1]  # 提取 pitch 并转换为弧度

    # 计算三维单位向量
    x = torch.cos(pitch) * torch.cos(yaw)
    y = torch.cos(pitch) * torch.sin(yaw)
    z = torch.sin(pitch)

    # 堆叠成 [B, 3] 的形状
    return torch.stack([x, y, z], axis=1)

def gaze_to_pitch_vector(gaze):
    """
    将二维视线角 [yaw, pitch] 转换为三维单位向量。
    支持批量输入，输入形状为 [B, 2]，输出形状为 [B, 3]。
    """
    yaw = gaze[:, 0]  # 提取 yaw 并转换为弧度
    pitch = gaze[:, 1]  # 提取 pitch 并转换为弧度

    # 计算三维单位向量
    x = torch.cos(yaw) * torch.sin(pitch)
    y = torch.sin(yaw)
    z = torch.cos(yaw) * torch.cos(pitch)

    # 堆叠成 [B, 3] 的形状
    return torch.stack([x, y, z], axis=1)


def inverse_rotation(g2, R_yaw, R_pitch):
    """
    通过旋转矩阵 R 和旋转后的视线角 g2 反向求出原始视线角 g1。
    """
    # 首先还原yaw角
    # 将 g2 转换为三维单位向量
    g2 = torch.deg2rad(g2)
    v2_yaw = gaze_to_yaw_vector(g2)  # 形状 [B, 3]
    R_yaw_inv = torch.inverse(R_yaw)  # 计算逆矩阵
    v1_yaw_rotated = torch.bmm(R_yaw_inv, v2_yaw.unsqueeze(-1)).squeeze(-1)  # 形状 [B, 3]
    # 归一化 v1_rotated（单位向量）
    v1_yaw_rotated_norm = F.normalize(v1_yaw_rotated, p=2, dim=1)  # p=2 表示 L2 范数，dim=1 对每个样本的 3D 向量归一化

    # 从旋转后的向量中提取新的 yaw 和 pitch
    # pitch1 = torch.asin(v1_yaw_rotated_norm[:, 2])  # 形状 [B]
    # pitch1 = torch.asin(v2_yaw[:, 2])  # 形状 [B]
    pitch1 = g2[:, 1]  # 形状 [B]
    cos_pitch1 = torch.cos(pitch1) + 1e-10
    yaw1 = torch.atan2(v1_yaw_rotated_norm[:, 1]/cos_pitch1, v1_yaw_rotated_norm[:, 0]/cos_pitch1)

    g1_yaw_inverted = torch.stack([yaw1, pitch1], dim=1)  # 形状 [B, 2]
    # 将弧度转换回角度
    g1_inverted = torch.rad2deg(g1_yaw_inverted)  # 形状 [B, 2]

    # 然后还原pitch角
    # 将 g2 转换为三维单位向量
    v2_pitch = gaze_to_pitch_vector(g1_yaw_inverted)  # 形状 [B, 3]
    R_pitch_inv = torch.inverse(R_pitch)  # 计算逆矩阵
    v1_pitch_rotated = torch.bmm(R_pitch_inv, v2_pitch.unsqueeze(-1)).squeeze(-1)  # 形状 [B, 3]
    # 归一化 v1_rotated（单位向量）
    v1_pitch_rotated_norm = F.normalize(v1_pitch_rotated, p=2, dim=1)  # p=2 表示 L2 范数，dim=1 对每个样本的 3D 向量归一化

    # 从旋转后的向量中提取新的 yaw 和 pitch
    # yaw2 = torch.asin(v1_pitch_rotated_norm[:, 1])  # 形状 [B]
    # yaw2 = torch.asin(v2_pitch[:, 1])  # 形状 [B]
    yaw2 = g1_yaw_inverted[:, 0]  # 形状 [B]
    cos_yaw2 = torch.cos(yaw2) + 1e-10
    pitch2 = torch.atan2(v1_pitch_rotated_norm[:, 0]/cos_yaw2, v1_pitch_rotated_norm[:, 2]/cos_yaw2)

    # 将弧度转换回角度
    g1_inverted_ = torch.rad2deg(torch.stack([yaw2, pitch2], dim=1))  # 形状 [B, 2]

    return g1_inverted_


def multi_inverse_rotation(theta, g2, R_z, R_y):
    """
    通过多次逆旋转，将旋转后的视线角 g2 反向求出原始视线角 g1。
    """
    # 计算旋转角度的整数部分
    num_rotations = int(np.abs(theta) // 2)

    for _ in range(num_rotations):
        g1 = inverse_rotation(g2, R_z, R_y)
        g2 = g1
    
    return g1


# 示例使用
if __name__ == "__main__":
    # 示例使用
    g1 = torch.tensor([[-2, 0], [10, 20], [100, 89]])  # 初始视线角 [yaw, pitch]，形状 [3, 2]
    g2 = torch.tensor([[0, 0], [10, 20], [-170, -80]])  # 旋转后的视线角 [yaw + theta, pitch + theta]，形状 [3, 2]
    degree = -2
    if degree <0:
        thetarad = torch.deg2rad(torch.tensor(-2))  # 旋转角度
    elif degree >0:
        thetarad = torch.deg2rad(torch.tensor(2))  # 旋转角度


    # 旋转矩阵
    R_z = torch.tensor([
        [torch.cos(thetarad), -torch.sin(thetarad), 0],
        [torch.sin(thetarad), torch.cos(thetarad), 0],
        [0, 0, 1]
    ])
    # 使用unsqueeze增加一个维度，然后使用expand复制
    R_z_tensor = R_z.unsqueeze(0).expand(g2.size(0), -1, -1)

    R_y = torch.tensor([
        [torch.cos(thetarad), 0, torch.sin(thetarad)],
        [0, 1, 0],
        [-torch.sin(thetarad), 0, torch.cos(thetarad)]
    ])
    # 使用unsqueeze增加一个维度，然后使用expand复制
    R_y_tensor = R_y.unsqueeze(0).expand(g2.size(0), -1, -1)

    print("g1:", g1)
    print("g2:", g2)
    g1_inv = multi_inverse_rotation(degree, g2, R_z_tensor, R_y_tensor)
    # 绕 z 轴旋转，只改变 yaw
    print("还原后的视线角 g1_inv:", g1_inv)