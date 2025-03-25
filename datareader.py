import random
import numpy as np
import cv2 
import os
from torch.utils.data import Dataset, DataLoader, Subset
import torch
import re
import torchvision.transforms as T
import math
from PIL import Image

class RandomGammaTransform:
    """
    自定义随机伽马亮度变换，适用于 PyTorch 数据增强流水线。
    """
    def __init__(self, gamma_values):
        """
        初始化随机伽马变换。param gamma_values: 可选的 gamma 值范围 (list 或 tuple)。
        """
        self.gamma_values = gamma_values

    def __call__(self, img):
        """
        应用随机伽马变换.param img: 输入图像 (PIL.Image)。return: 经过伽马变换后的图像 (PIL.Image)。
        """
        # 随机选择一个 gamma 值
        gamma = random.choice(self.gamma_values)
        # 将 PIL 图像转换为 NumPy 数组
        img_np = np.array(img)
        # 构建伽马查找表
        gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]
        gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
        # 应用查表变换
        img_np = cv2.LUT(img_np, gamma_table)
        # 转换回 PIL 图像
        return Image.fromarray(img_np)


class loader(Dataset):
    def __init__(self, path, root, header):
        self.header = header
        self.angle = 180
        self.binwidth = 4
        self.path = path
        self.root = root
        # build transforms
        self.transforms = T.Compose([
        T.ToTensor()  # 转换为 PyTorch 张量
        ])
        self.transforms_inc = T.Compose([
        T.ToPILImage(),  # 将 NumPy 图像转换为 PIL 图像
        RandomGammaTransform(gamma_values=[1.05, 1.1, 1.15, 1.2, 1.25]),  # 随机亮度改变
        T.ToTensor()  # 转换为 PyTorch 张量
        ])
        self.transforms_dec = T.Compose([
        T.ToPILImage(),  # 将 NumPy 图像转换为 PIL 图像
        RandomGammaTransform(gamma_values=[0.75, 0.8, 0.85, 0.9, 0.95]),  # 随机亮度改变
        T.ToTensor()  # 转换为 PyTorch 张量
        ])

        self.lines = []

        if isinstance(path, list):
            for f_item in self.path:
                with open(f_item) as f:
                    line = f.readlines()
                    if header: line.pop(0)
                    if len(line) > 0:
                        for i in range(len(line)):
                            newlines = self.__readdatalines(line[i])
                            if newlines is not None:
                                self.lines.append(newlines)  
        else:
            with open(self.path) as f:
                    line = f.readlines()
                    if header: line.pop(0)
                    if len(line) > 0:
                        for i in range(len(line)):
                            newlines = self.__readdatalines(line[i])
                            if newlines is not None:
                                self.lines.append(newlines)


    def __readdatalines(self, line):
        global dataset_type
        newlines = []
        lines = line.strip().split(" ")
        newlines.append(lines[0].replace("\\", "/"))  # face img file
        if dataset_type == "gaze360":
            newlines.append(lines[5])  # gaze360是5
        elif dataset_type == "eth" or dataset_type == "gazeCapture":
            gaze2d = lines[1] # ETH-XGaze是1
            # ETH-XGaze和gazeCapture数据集中label是[pitch, yaw]，顺序与其他数据集不同，需要转换
            strs = gaze2d.split(",")
            gaze2d = strs[1] + "," + strs[0]
            newlines.append(gaze2d)
        elif dataset_type == "diap":
            newlines.append(lines[6])  # diap是6
        elif dataset_type == "mpiiface":
            newlines.append(lines[7])  # mpiigaze是7
        label = np.array(newlines[1].split(",")).astype("float")
        # yaw的范围[-180,180],pitch的范围 [-90,90]
        if abs((label[0]*180/np.pi)) <= self.angle and abs((label[1]*180/np.pi)) <= self.angle/2:
            return newlines
        else:
            return None
    

    def __len__(self):
        return len(self.lines)
    
    def __getitem__(self, idx):
        lines = self.lines[idx]

        path_root = self.root
        face_name = lines[0]
        gaze2d = lines[1]
        info_name = face_name

        face_img = cv2.imread(os.path.join(path_root, face_name))
        # 转换为 RGB 格式
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        # 垂直翻转
        face_flip_img = cv2.flip(face_img, 1)

        # 增加高斯模糊，滤波，使图像平滑
        if dataset_type == "eth":
          face_img = cv2.GaussianBlur(face_img, (13, 13), 0)
        if dataset_type == "gaze360":
          face_img = cv2.GaussianBlur(face_img, (3, 3), 0)

        img = self.transforms(face_img)
        img_flip = self.transforms(face_flip_img)
        img_inc = self.transforms_inc(face_img)
        img_dec = self.transforms_dec(face_img)

        gaze2d_array = np.array(gaze2d.split(",")).astype("float")
        gaze2d_label = torch.from_numpy(gaze2d_array).type(torch.FloatTensor)
        gaze2d_label = gaze2d_label* 180 / np.pi

        info = {"face": img,
                "face_flip": img_flip,
                "face_inc": img_inc,
                "face_dec": img_dec,
                "gaze": gaze2d_label,
                "name": info_name}

        return info

            


def txtload(labelpath, imagepath, batch_size, shuffle=True, num_workers=0, header=True, data_type="mpiiface"):
    global dataset_type
    dataset_type = data_type
    dataset = loader(labelpath, imagepath, header)
    print(f"[Read Data]: Total num: {len(dataset)}")
    print(f"[Read Data]: Label path: {labelpath}")
    load = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return len(dataset), load


def random_sample_txtload(labelpath, imagepath, batch_size, num_samples, shuffle=True, num_workers=0, header=True, data_type="mpiiface", inc_gamma=[1.1], dec_gamma=[0.9]):
    global dataset_type
    dataset_type = data_type
    dataset = loader(labelpath, imagepath, header, inc_gamma, dec_gamma)
    print(f"[Read Data]: Total num: {len(dataset)}")
    print(f"[Read Data]: Label path: {labelpath}")
    # 随机选择样本的索引
    random_indices = random.sample(range(len(dataset)), num_samples)
    # 根据随机索引创建子集
    subset_dataset = Subset(dataset, random_indices)
    print(f"[Read Data]: Total sample num: {len(subset_dataset)}")
    load = DataLoader(subset_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return load

