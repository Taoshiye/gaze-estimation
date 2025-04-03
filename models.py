import torch
import torch.nn as nn
from resnet import resnet18, resnet50
import torch.nn.functional as F


"""视线预测多层感知器"""
class mlp_gazeEs(nn.Module):
    def __init__(self, channel, flag=True, drop_p=0.2):
        super(mlp_gazeEs, self).__init__()
        self.flag = flag
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.drop = nn.Dropout(drop_p)
        self.fc1 = nn.Linear(channel, 1000)
        self.fc2 = nn.Linear(1000, 256)
        self.fc3_g = nn.Linear(256, 2)
        self.fc3_m = nn.Linear(256, 9)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = nn.ReLU()(self.fc1(x))
        x = self.fc2(x)
        x = self.drop(x)
        if self.flag:
            x_g = self.fc3_g(x)
            x_m = self.fc3_m(x)
            return x_g, x_m
        else:
            x_g = self.fc3_g(x)
            return x_g
    

"""弱回归器"""
class fc_gazeEs(nn.Module):
    def __init__(self, channel, drop_p=0.5):
        super(fc_gazeEs, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.drop = nn.Dropout(drop_p)
        self.fc = nn.Linear(channel, 2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.avgpool(x)
        x_1d = torch.flatten(x, start_dim=1)
        x_1d = self.drop(x_1d)
        x = self.fc(x_1d)
        return x



class GazeRes(nn.Module):
    def __init__(self, backbone = "res18"):
        super(GazeRes, self).__init__()
        self.img_feature_dim = 512  # the dimension of the CNN feature to represent each frame
        if backbone == "res18":
            self.base_model = resnet18(pretrained=True)
        elif backbone == "res50":
            self.base_model = resnet50(pretrained=True)

        self.gazeEs = mlp_gazeEs(self.img_feature_dim)

    def forward(self, x_in):
        base_out = self.base_model(x_in)
        output = self.gazeEs(base_out)
        angular_output = output

        return angular_output, base_out


class RotConGE(nn.Module):
    def __init__(self, backbone = "res18", degree_list = [2], drop_p=0.5):
        super(RotConGE, self).__init__()
        self.degree = degree_list
        self.img_feature_dim = 512  # the dimension of the CNN feature to represent each frame
        if backbone == "res18":
            self.base_model = resnet18(pretrained=True)
        elif backbone == "res50":
            self.base_model = resnet50(pretrained=True)

        self.gazeEs = mlp_gazeEs(self.img_feature_dim, flag=False)
        # Create a ModuleList for the rotate degree outputs
        self.rotate_gazeEs = nn.ModuleList([
          mlp_gazeEs(self.img_feature_dim) for _ in range(len(degree_list))
        ])

    def forward(self, x_in):
        base_features = self.base_model(x_in)
        output_gaze = self.gazeEs(base_features)

        angular_output = {}
        rotation_matrix = {}
        for i, layer in enumerate(self.rotate_gazeEs):
            degree = self.degree[i]
            output_degree_gaze, output_rotation_matrix = layer(base_features)
            angular_output[f"out_{degree}_gaze"] = output_degree_gaze
            rotation_matrix[f"rot_{degree}_matrix"] = output_rotation_matrix

        return output_gaze, angular_output, rotation_matrix


"""利用世界坐标系下，固定旋转角度则旋转矩阵固定的特性。将旋转矩阵作为全连接层的参数对旋转角度进行还原。"""
class mlp_matgazeEs(nn.Module):
    def __init__(self, channel, flag=True, drop_p=0.5):
        super(mlp_matgazeEs, self).__init__()
        self.flag = flag
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.drop = nn.Dropout(drop_p)
        self.fc1 = nn.Linear(channel, 1000)
        self.fc2 = nn.Linear(1000, channel)
        self.fc3_g = nn.Linear(channel, 2)
        self.fc3_m1 = nn.Linear(channel, 9)
        self.fc3_m2 = nn.Linear(channel, 9)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = nn.ReLU()(self.fc1(x))
        x = self.fc2(x)
        x = self.drop(x)
        if self.flag:
            x_g = self.fc3_g(x)
            x_m1 = self.fc3_m1(x)
            x_m2 = self.fc3_m2(x)
            return x_g, x_m1, x_m2
        else:
            x_g = self.fc3_g(x)
            return x_g


class fc_matgazeEs(nn.Module):
    def __init__(self, channel, drop_p=0.5):
        super(fc_matgazeEs, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.drop = nn.Dropout(drop_p)
        self.fc = nn.Linear(channel, 2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.avgpool(x)
        x_1d = torch.flatten(x, start_dim=1)
        x_1d = self.drop(x_1d)
        x_g = self.fc(x_1d)
        return x_g


class fc_rotMat(nn.Module):
    def __init__(self, channel, drop_p=0.5):
        super(fc_rotMat, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.drop = nn.Dropout(drop_p)
        self.fc_y = nn.Linear(channel, 9)
        self.fc_p = nn.Linear(channel, 9)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.avgpool(x)
        x_1d = torch.flatten(x, start_dim=1)
        x_1d = self.drop(x_1d)
        x_my = self.fc_y(x_1d)
        x_mp = self.fc_p(x_1d)
        return x_my, x_mp


class MatRotGE(nn.Module):
    def __init__(self, backbone = "res18", degree_list = [2]):
        super(MatRotGE, self).__init__()
        self.degree = degree_list
        self.img_feature_dim = 512  # the dimension of the CNN feature to represent each frame
        if backbone == "res18":
            self.base_model = resnet18(pretrained=True)
        elif backbone == "res50":
            self.base_model = resnet50(pretrained=True)

        self.gazeEs = fc_matgazeEs(self.img_feature_dim)
        # Create a ModuleList for the rotate degree outputs
        self.rotate_gazeEs = nn.ModuleList([
          fc_matgazeEs(self.img_feature_dim) for _ in range(len(degree_list))
        ])

    def forward(self, x_in):
        base_features = self.base_model(x_in)
        output_gaze = self.gazeEs(base_features)

        angular_output = {}
        for i, layer in enumerate(self.rotate_gazeEs):
            degree = self.degree[i]
            output_degree_gaze = layer(base_features)
            angular_output[f"out_{degree}_gaze"] = output_degree_gaze

        return output_gaze, angular_output


class UncertaintyWPseudoLabelLoss(nn.Module):
    def __init__(self, lamda_pseudo = 0.0001):
        super(UncertaintyWPseudoLabelLoss, self).__init__()
        self.lamda_pseudo = lamda_pseudo 

    def forward(self, gaze, pseudo):
        std = torch.std(gaze, dim=0) # gaze [6, b, 2]
        pseudo = pseudo.unsqueeze(0)
        return torch.mean(std) + self.lamda_pseudo * torch.mean(torch.abs(gaze - pseudo) / std.detach())


class MinimumVarianceLoss(nn.Module):
    def __init__(self):
        super(MinimumVarianceLoss, self).__init__()

    def forward(self, gaze):
        # std = torch.std(gaze, dim=0) # gaze [6, b, 2]
        std = torch.var(gaze, dim=0) # gaze [6, b, 2]
        return torch.mean(std)


"""一种复杂回归"""
class GAZEnet(nn.Module):
    def __init__(self, channel):
        super(GAZEnet, self).__init__()
        self.cbam = CBAM(channel)
        self.conv2d = nn.Conv2d(channel, channel, 3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(channel)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

    def forward(self, x):
        x_cbam = self.cbam(x)
        x_1 = self.relu(self.bn(self.conv2d(x_cbam)))
        x_2 = self.relu(self.bn(self.conv2d(x_1)))
        x_3 = self.relu(self.bn(self.conv2d(x_2)))
        out = self.maxpool(x_3)

        return out


# CBAM模块
class CBAM(nn.Module):
    def __init__(self, channel):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(channel)
        self.sa = SpatialAttention()

    def forward(self, x):
        x_ca = self.ca(x)
        out = x_ca * x
        out_sa = self.sa(out)
        out = out_sa * out
        return out
    

class ChannelAttention(nn.Module):
    # ratio代表第一个全连接的通道下降倍数
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        # 使用自适应池化缩减map的大小，保持通道不变
        # 全局最大池化 [b,c,h,w]==>[b,c,1,1]  当前数据下[16*8,512,7,7]==>[16*8,512,1,1]
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 全局平均池化 [b,c,h,w]==>[b,c,1,1]
        self.max_pool = nn.AdaptiveMaxPool2d(1)


        # 第一个全连接层, 通道数下降4倍（可以换成1x1的卷积，效果相同）
        # 第二个全连接层, 恢复通道数（可以换成1x1的卷积，效果相同）
        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
                                nn.BatchNorm2d(in_planes // ratio),
                                nn.ReLU(),
                                nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        out = self.sigmoid(out)
        return out


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        out = self.sigmoid(x)
        return out