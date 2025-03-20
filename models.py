import torch
import torch.nn as nn
from resnet import resnet18, resnet50
import torch.nn.functional as F


"""视线预测多层感知器"""
class mlp_gazeEs(nn.Module):
    def __init__(self, channel, drop_p=0.5):
        super(mlp_gazeEs, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.drop = nn.Dropout(drop_p)
        self.fc1 = nn.Linear(channel, 1000)
        self.fc2 = nn.Linear(1000, 256)
        self.fc3 = nn.Linear(256, 2)

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
        x = self.fc3(self.drop(x))
        return x
    

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

        self.gazeEs = fc_gazeEs(self.img_feature_dim)
        # Create a ModuleList for the rotate degree outputs
        self.rotate_gazeEs = nn.ModuleList([
          fc_gazeEs(self.img_feature_dim) for _ in range(len(degree_list))
        ])

    def forward(self, x_in):
        base_features = self.base_model(x_in)
        output_gaze = self.gazeEs(base_features)

        angular_sub_output = {}
        for i, layer in enumerate(self.rotate_gazeEs):
            degree = self.degree[i]
            output_degree_gaze = layer(base_features)
            angular_sub_output[f"out_{degree}_gaze"] = torch.sub(output_degree_gaze, degree)

        return output_gaze, angular_sub_output, base_features



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


class GAZEnet(nn.Module):
    def __init__(self, drop_p=0.5):
        super(GAZEnet, self).__init__()
        self.img_feature_dim = 512

        self.gazeNet = nn.Sequential(
            CBAM(512),
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.last_layer = nn.Linear(self.img_feature_dim, 2)
        self.drop = nn.Dropout(drop_p)

    def forward(self, x):
        x = self.gazeNet(x)
        x = self.avgpool(x)
        base_out = torch.flatten(x, start_dim=1)
        output = self.drop(base_out)
        out = self.last_layer(output)
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