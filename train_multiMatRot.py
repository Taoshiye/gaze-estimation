import argparse
import shutil
import cv2
import models
import datareader
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
import sys
import os
import matplotlib.pyplot as plt
import yaml
import torch.backends.cudnn as cudnn
from datetime import datetime
import logging
from torch.utils.tensorboard import SummaryWriter
import os
import utils
import rotate
os.environ["CUDA_VISIBLE_DEVICES"] = '1'


def compute_rot_matrix(batch, theta=2, mode='yaw'):
    """
    计算旋转矩阵标签
    """
    # 将角度转换为弧度，同时避免不必要的内存复制
    thetarad = torch.deg2rad(theta.clone().detach())  # 旋转角度
    
    if mode=='yaw':
        # 旋转矩阵
        R = torch.tensor([
            [torch.cos(thetarad), -torch.sin(thetarad), 0],
            [torch.sin(thetarad), torch.cos(thetarad), 0],
            [0, 0, 1]
        ])
    elif mode=='pitch':
        R = torch.tensor([
            [torch.cos(thetarad), 0, torch.sin(thetarad)],
            [0, 1, 0],
            [-torch.sin(thetarad), 0, torch.cos(thetarad)]
        ])
    else:
        print('mode not exit!')
    
    # 使用unsqueeze增加一个维度，然后使用expand复制
    R_tensor = R.unsqueeze(0).expand(batch, -1, -1)

    return R_tensor


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


def gazeto3d(gaze):
  gaze_gt = np.zeros([3])
  gaze_gt[0] = -np.cos(gaze[1]) * np.sin(gaze[0])
  gaze_gt[1] = -np.sin(gaze[1])
  gaze_gt[2] = -np.cos(gaze[1]) * np.cos(gaze[0])
  return gaze_gt

def angular(gaze, label):
  total = np.sum(gaze * label)
  return np.arccos(min(total/(np.linalg.norm(gaze)* np.linalg.norm(label)), 0.9999999))*180/np.pi


def train(args, dataset, valid_dataset):
    # 配置tensorboard
    tb_log_dir = os.path.join(args.savepath, 'tb_log')
    os.makedirs(tb_log_dir, exist_ok=True)
    writer = SummaryWriter(tb_log_dir)
    
    savepath = os.path.join(args.savepath, f"checkpoint")
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    # 加载模型
    logging.info("Model building")
    net = models.MatRotGE(args.backbone, args.degree)
    net.train()
    net.to(args.device)
    for name, param in net.named_parameters():
        if param.requires_grad:
            logging.info(name)

    # 计算参数量
    total_params = sum(p.numel() for p in net.parameters())
    trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    logging.info(f"模型总参数量: {total_params}")
    logging.info(f"可训练参数量: {trainable_params}")

    logging.info("optimizer building")
    # 定义损失函数
    lossfunc = getattr(nn, args.loss)().cuda()
    # 创建优化器字典
    optimizers = {
        'base_model': optim.Adam(net.base_model.parameters(), lr=args.lr, betas=(0.9, 0.95)),
        'gazeEs': optim.Adam(net.gazeEs.parameters(), lr=args.lr, betas=(0.9, 0.95)),
    }
    # 动态为 rotate_*_degree 模块创建优化器
    for i, degree in enumerate(args.degree):
        optimizers[f'rotate_{degree}_degree'] = optim.Adam(net.rotate_gazeEs[i].parameters(), lr=args.lr, betas=(0.9, 0.95))
    # 创建调度器字典
    schedulers = {}
    for key, optimizer in optimizers.items():
        schedulers[key] = optim.lr_scheduler.StepLR(optimizer, step_size=args.decaysteps, gamma=args.decayratio)

    # 开始训练
    logging.info("=====Traning=====")
    length = len(dataset)
    total = length * args.epoches
    cur = 0
    timebegin = time.time()
    val_loss_list = []
    with open(os.path.join(savepath, "train_log"), 'a') as outfile:
        for epoch in range(1, args.epoches + 1):
            logging.info(f"=====Traning/epoch={epoch}=====")
            for i, data in enumerate(dataset):
                torch.cuda.empty_cache()
                # Acquire data
                input = move_to_gpu(data['face'], args.device)
                target = move_to_gpu(data['gaze'], args.device)

                for k, item in enumerate(data):
                    if epoch == 1 and i == 0 and k < 1:
                        face_name = os.path.join(savepath, os.path.basename(data["name"][k])).replace('.jpg', '_face.jpg')
                        save_image(face_name, input[k].cpu().numpy())
        
                # forward
                gaze, gaze_dict, rotmat_dict = net(input)

                loss_gaze = lossfunc(gaze, target)
                loss_rot_gaze = []
                loss_degree_gaze_list = []
                loss_degree_rotmat_list = []
                for degree in args.degree:
                    # 旋转角度
                    theta = torch.tensor(np.sign(degree) * 2)
                    # 利用标签计算旋转矩阵损失
                    tg_yaw_rot_matrix = compute_rot_matrix(input.size(0), theta, mode='yaw')
                    tg_yaw_rot_matrix = tg_yaw_rot_matrix.to(args.device)
                    tg_pitch_rot_matrix = compute_rot_matrix(input.size(0), theta, mode='pitch')
                    tg_pitch_rot_matrix = tg_pitch_rot_matrix.to(args.device)
                    yaw_rot_matrix = rotmat_dict[f"rot_yaw_{degree}_matrix"]
                    yaw_rot_matrix = yaw_rot_matrix.view(yaw_rot_matrix.shape[0], 3, 3)
                    pitch_rot_matrix = rotmat_dict[f"rot_pitch_{degree}_matrix"]
                    pitch_rot_matrix = pitch_rot_matrix.view(pitch_rot_matrix.shape[0], 3, 3)
                    loss_degree_rotmat = lossfunc(yaw_rot_matrix, tg_yaw_rot_matrix) + lossfunc(pitch_rot_matrix, tg_pitch_rot_matrix)
                    loss_degree_rotmat_list.append(loss_degree_rotmat)
                    # 利用旋转矩阵还原旋转角为原角度
                    rot_gaze = gaze_dict[f"out_{degree}_gaze"]
                    inv_gaze = rotate.multi_inverse_rotation(degree, rot_gaze, yaw_rot_matrix, pitch_rot_matrix)
                    loss_degree_gaze = lossfunc(inv_gaze, target)
                    loss_degree_gaze_list.append(loss_degree_gaze)

                    loss_value = 0.1*loss_degree_gaze + 10*loss_degree_rotmat
                    loss_rot_gaze.append(loss_value)
                
                [optimizer.zero_grad() for name, optimizer in optimizers.items()]
                
                # 冻结 rotate_*_degree 所有全连接层
                for param in net.rotate_gazeEs.parameters():
                    param.requires_grad = False

                # backward，利用没有进行角度旋转的更新主干网络
                loss_gaze.backward(retain_graph=True)

                # 冻结 base_model和gazeEs
                for param in net.base_model.parameters():
                    param.requires_grad = False
                for param in net.gazeEs.parameters():
                    param.requires_grad = False
                
                # 确保对应的网络层参数可训练
                for param in net.rotate_gazeEs.parameters():
                    param.requires_grad = True
                # 使用 torch.autograd.backward 一次性反向传播
                torch.autograd.backward(loss_rot_gaze)  # 这里 grad_tensors 默认是 [1.0, 1.0, ...]

                # 解冻 base_model和fc_layer
                for param in net.base_model.parameters():
                    param.requires_grad = True
                for param in net.gazeEs.parameters():
                    param.requires_grad = True
                
                # 更新学习率
                [optimizer.step() for name, optimizer in optimizers.items()]
                [scheduler.step() for name, scheduler in schedulers.items()]
                cur += 1

                current_lr = optimizers['base_model'].param_groups[0]['lr']
                writer.add_scalar("learning_rate", current_lr, cur)
                """tensorboard writer"""
                writer.add_scalar("train/loss_gaze", loss_gaze, cur)
                for idx, degree in enumerate(args.degree):
                    writer.add_scalar(f"train/loss_degree_gaze/degree=[{degree}]", loss_degree_gaze_list[idx], cur)
                for idx, degree in enumerate(args.degree):
                    writer.add_scalar(f"train/loss_degree_rotmat/degree=[{degree}]", loss_degree_rotmat_list[idx], cur)

                # print logs
                if i % 20 == 0:
                    timeend = time.time()
                    resttime = (timeend - timebegin)/cur * (total-cur)/3600
                    log = f"[{epoch}/{args.epoches}]: [{i}/{length}] loss_gaze:{'%.8f' % loss_gaze} loss_degree_rotmat:{[f'{x:.8f}' for x in loss_degree_rotmat_list]} " \
                        f"loss_degree_gaze:{[f'{x:.8f}' for x in loss_degree_gaze_list]} lr:{current_lr}, rest time:{resttime:.2f}h"
                    logging.info(log)
                    outfile.write(log + "\n")
                    sys.stdout.flush()   
                    outfile.flush()

            checkpoint = {
                "net": net.state_dict(),
                'optimizer': optimizers,
                "scheduler": schedulers,
                "epoch": epoch
            }
            torch.save(checkpoint, os.path.join(savepath, f"Iter_{epoch}.pt"))

            valid_loss = valid(args, checkpoint, valid_dataset)
            val_loss_list.append(valid_loss)

    sequence = list(range(1, len(val_loss_list)+1))
    # 绘制折线图
    plt.figure(figsize=(10, 5))
    plt.plot(sequence, val_loss_list, marker='o')
    plt.title('gaze avg')
    plt.xlabel('Sequence')
    plt.ylabel('Value')
    plt.grid(True)
    plt.xticks(sequence)  # 确保所有序号都显示在x轴上
    # 保存折线图到特定目录
    plt.savefig(os.path.join(savepath, 'val_avg.png'))



def valid(args, ckpt, dataset):
    logging.info("=====Validing=====")
    net = models.MatRotGE(args.backbone, args.degree)
    net.to(args.device)
    net.load_state_dict(ckpt['net'])
    net.eval()

    accs = 0
    count = 0

    with torch.no_grad():
        for j, data in enumerate(dataset):
            input = move_to_gpu(data['face'], args.device)
            target = move_to_gpu(data['gaze'], args.device)
            target = target.float()*np.pi/180

            gaze, gaze_dict, rotmat_dict = net(input)
            for degree in args.degree:
                rot_gaze = gaze_dict[f"out_{degree}_gaze"]
                yaw_rot_matrix = rotmat_dict[f"rot_yaw_{degree}_matrix"]
                yaw_rot_matrix = yaw_rot_matrix.view(yaw_rot_matrix.shape[0], 3, 3)
                pitch_rot_matrix = rotmat_dict[f"rot_pitch_{degree}_matrix"]
                pitch_rot_matrix = pitch_rot_matrix.view(pitch_rot_matrix.shape[0], 3, 3)
                inv_gaze = rotate.multi_inverse_rotation(degree, rot_gaze, yaw_rot_matrix, pitch_rot_matrix)
                gaze_dict[f"out_{degree}_gaze"] = inv_gaze

                # gaze_dict[f"out_{degree}_gaze"] = torch.sub(rot_gaze, degree)
            gaze_list = torch.cat((gaze.unsqueeze(0), torch.stack(list(gaze_dict.values()), dim=0)), dim=0)
            pre_gaze = gaze_list.mean(dim=0)
            pre_gaze = pre_gaze*np.pi/180

            for k, (pre, gt) in enumerate(zip(pre_gaze, target)):
                pre = pre.cpu().detach().numpy()
                gt = gt.cpu().detach().numpy()
                count += 1
                accs += angular(gazeto3d(pre), gazeto3d(gt))

        avg = accs/count
        loger = f"[Total Num: {count}, avg: {avg}]"
        logging.info(loger)
    
    return avg


def test(args, dataset):
    logging.info("=====Testing=====")
    # ckptpath = os.path.join(args.savepath, f"checkpoint")
    ckptpath = '/media/ts/0074477d-81c5-4692-b399-7ccecf34dbb6/GazeEstimation/ts-record/ts-MultiRotCon_addGamma/20250401_101848/checkpoint'
    outputpath = os.path.join(args.savepath, f"evaluation")
    if not os.path.exists(outputpath):
        os.makedirs(outputpath)

    # 定义损失函数
    lossfunc = getattr(nn, args.loss)().cuda()
    
    avg_list = []
    best_epoch = 0
    best_avg = 100
    epoch = args.epoches
    with open(os.path.join(outputpath, "total_test.log"), 'a') as dataoutfile:
        for saveiter in range(1, epoch+1, 1):
            logging.info(f"Test {saveiter}")
            dataoutfile.write(f"the result of epoch {saveiter}:\n")
            net = models.MatRotGE(args.backbone, args.degree)
            statedict = torch.load(os.path.join(ckptpath, f"Iter_{saveiter}.pt"))
            logging.info(f"Loading from: {os.path.join(ckptpath, f"Iter_{saveiter}.pt")}")
            net.to(args.device)
            net.load_state_dict(statedict['net'])
            net.eval()

            length = len(dataset)
            accs = 0
            count = 0
            losssum =0
            losscount = 0

            with torch.no_grad():
                with open(os.path.join(outputpath, f"{saveiter}.log"), 'w') as outfile:
                    outfile.write("name results gts\n")
                    for j, data in enumerate(dataset):
                        input = move_to_gpu(data['face'], args.device)
                        target = move_to_gpu(data['gaze'], args.device)
                        target = target.float()*np.pi/180
                        names = data["name"]

                        gaze, gaze_dict, rotmat_dict = net(input)
                        result_dict  ={}
                        for degree in args.degree:
                            if degree == 6 or degree == -6:
                                continue
                            rot_gaze = gaze_dict[f"out_{degree}_gaze"]
                            yaw_rot_matrix = rotmat_dict[f"rot_yaw_{degree}_matrix"]
                            yaw_rot_matrix = yaw_rot_matrix.view(yaw_rot_matrix.shape[0], 3, 3)
                            pitch_rot_matrix = rotmat_dict[f"rot_pitch_{degree}_matrix"]
                            pitch_rot_matrix = pitch_rot_matrix.view(pitch_rot_matrix.shape[0], 3, 3)
                            inv_gaze = rotate.multi_inverse_rotation(degree, rot_gaze, yaw_rot_matrix, pitch_rot_matrix)
                            result_dict[f"out_{degree}_gaze"] = inv_gaze

                            # gaze_dict[f"out_{degree}_gaze"] = torch.sub(rot_gaze, degree)
                        gaze_list = torch.cat((gaze.unsqueeze(0), torch.stack(list(result_dict.values()), dim=0)), dim=0)
                        pre_gaze = gaze_list.mean(dim=0)
                        pre_gaze = pre_gaze*np.pi/180   

                        loss_gaze = lossfunc(pre_gaze, target)
                        losssum += loss_gaze
                        losscount += 1

                        for k, (pre, gt) in enumerate(zip(pre_gaze, target)):
                            if j == 1 and k < 1:
                                face_name = os.path.join(outputpath, os.path.basename(data["name"][k])).replace('.jpg', '_face.jpg')
                                save_image(face_name, data['face'][k].cpu().numpy())
                            pre = pre.cpu().detach().numpy()
                            gt = gt.cpu().detach().numpy()
                            count += 1
                            accs += angular(gazeto3d(pre), gazeto3d(gt))

                            name = [names[k]]
                            pre = [str(u) for u in pre] 
                            gt = [str(u) for u in gt] 
                            log = name + [",".join(pre)] + [",".join(gt)]
                            outfile.write(" ".join(log) + "\n")
                        
                        if losscount % 10 == 0:
                            log = f"[{saveiter}/{epoch}]: [{j}/{length}] avg: {accs/count} avgloss:{losssum/losscount}"
                            print(log)
                            dataoutfile.write(log + "\n")


                    avg = accs/count
                    if avg < best_avg:
                       best_avg = avg
                       best_epoch = saveiter
                    avg_list.append(avg)
                    loger = f"[{saveiter}] Total Num: {count}, avg: {avg}, avgloss:{losssum/losscount}"
                    best_loger = f"the best epoch: {best_epoch}, the best avg: {best_avg}"
                    outfile.write(loger)
                    dataoutfile.write(loger + '\n' + best_loger + '\n')
                    logging.info(loger)
                    logging.info(best_loger)
    
    
    sequence = list(range(1, len(avg_list)+1))
    # 绘制折线图
    plt.figure(figsize=(10, 5))
    plt.plot(sequence, avg_list, marker='o')
    plt.title('gaze avg')
    plt.xlabel('Sequence')
    plt.ylabel('Value')
    plt.grid(True)
    plt.xticks(sequence)  # 确保所有序号都显示在x轴上
    # 保存折线图到特定目录
    plt.savefig(os.path.join(outputpath, 'avg.png'))


  
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RotCon_addGamma Domain generalization')
    parser.add_argument('--backbone', type=str, 
                        default='res18', help='backbone')
    parser.add_argument('--degree', type=list, 
                        default=[-6, -4, -2, 2, 4, 6], help='旋转角度')
    parser.add_argument('--batch_size', type=int, 
                        default=128, help='源域训练 batch size')
    parser.add_argument('--lr', type=float, 
                        default=1e-4, help="源域训练初始 learning rate")
    parser.add_argument('--decaysteps', type=int, 
                        default=5000, help="学习率衰减步数，只源域")
    parser.add_argument('--decayratio', type=float, 
                        default=0.9, help="学习率衰减因子，只源域")
    parser.add_argument('--epoches', type=int, 
                        default=30, help='源域训练总 epoches')
    parser.add_argument('--source', type=str, 
                        default='gaze360', help='source dataset, eth/gaze360')
    parser.add_argument('--target', type=str, 
                        default='mpiiface', help='target dataset,  mpiiface/diap')
    parser.add_argument('--root', type=str, 
                        default="/media/ts/0074477d-81c5-4692-b399-7ccecf34dbb6/GazeEstimation/ts-record/ts-MultiRotCon_addGamma")
    parser.add_argument('--loss', 
                        default= "L1Loss", help="标签损失")
    parser.add_argument('--device', type=str, 
                        default="cuda", help='Set device')
    # use config file
    # ... parse other arguments ...
    args = parser.parse_args()
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    # save path for logs and models
    savepath = os.path.join(args.root, current_time)
    # savepath = os.path.join(args.root, 'debug')
    args.savepath = savepath
    os.makedirs(args.savepath, exist_ok=True)
    # 留存每个实验的说明
    with open(os.path.join(args.root, 'readme.txt'), 'a') as file:
        file.write(f'{current_time}: multiMatRot网络。加入旋转矩阵，直接跨域。利用世界坐标系下，固定旋转角度则旋转矩阵固定的特性。利用旋转矩阵还原角度计算损失。减少全连接层的数量。学习率衰减。仅测试，网络权重为20250401_101848的训练结果，测试时不使用 degree=±6 支路。' + '\n')
    # 复制文件
    train_code_path = os.path.abspath(__file__)
    shutil.copy(train_code_path, os.path.join(savepath, f'{os.path.basename(train_code_path)}'))

    # Load configuration
    config = yaml.load(open("datapath.yaml"), Loader=yaml.FullLoader)

    imagepath_source = config[args.source]["image"]
    labelpath_source = config[args.source]["label"]
    validpath_source = config[args.source]["valid"]
    imagepath_target = config[args.target]["image"]
    labelpath_target = config[args.target]["label"]
    args.labelpath_source = labelpath_source
    args.labelpath_target = labelpath_target

    # 配置logging
    logging.basicConfig(
        level=logging.INFO,  # 设置日志级别
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # 设置日志格式
        datefmt='%Y-%m-%d %H:%M:%S',  # 设置日期格式
        handlers=[
            logging.FileHandler(os.path.join(args.savepath, 'training_log.txt')),  # 将日志保存到文件
            logging.StreamHandler()  # 将日志打印到控制台
        ]
    )
    logging.info(f"trainning args:{args}")

    # read source data
    logging.info(f"======read source data={args.source}======")
    length_source_data, dataset_source = datareader.txtload(labelpath_source, imagepath_source, args.batch_size,
                                 shuffle=True, num_workers=2, header=True, data_type=args.source)
    args.length_s = length_source_data
    logging.info(f"Total Num of source: {args.length_s}")

    _, valid_dataset_source = datareader.txtload(validpath_source, imagepath_source, args.batch_size,
                                 shuffle=True, num_workers=2, header=True, data_type=args.source)
    # train(args, dataset_source, valid_dataset_source)

    # read target data for test
    logging.info(f"======read target data={args.target}======")
    folder_target = os.listdir(labelpath_target)
    folder_target.sort()
    labelpath_target_list = [os.path.join(labelpath_target, j) for j in folder_target]
    length_target_data, dataset_target = datareader.txtload(labelpath_target_list, imagepath_target, args.batch_size,
                                 shuffle=False, num_workers=2, header=True, data_type=args.target)
    args.length_t = length_target_data
    logging.info(f"Total Num of target: {args.length_t}")
    test(args, dataset_target)