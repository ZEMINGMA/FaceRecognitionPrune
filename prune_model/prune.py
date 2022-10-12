import torch
from torchsummary import summary
from torch import nn
import os
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from data import *
import torch.utils.data as data
import numpy as np
import cv2
import tools
import time
import os.path as osp

######################
# 剪枝的范例程序
# python prune.py --trained_model out/prune_net_100.pth --percent 0.8 --visual_threshold 0.3
######################

######################
# 准备工作
######################
parser = argparse.ArgumentParser(description='YOLO Detection Pruning')
# 训练的时候添加的L1正则化稀疏稀疏系数
parser.add_argument('--sparsity-regularization', '-sr', dest='sr', action='store_true',
                    help='train with channel sparsity regularization')
parser.add_argument('--s', type=float, default=0.0001,
                    help='scale sparse rate (default: 0.0001)')
# 用于微调的epoch次数
parser.add_argument('--epochs', type=int, default=50, metavar='N',
                    help='number of epochs to train (default: 50)')

# 预训练模型
# 可能保存的只有权值，load模型的时候没有对应的模型框架供加载
# 最好是保存整个模型，虽然会慢一点，但是方便加载，裁减之后的模型结构有变化
parser.add_argument('--trained_model', default='weights/custom/slim_yolo_v2',
                    type=str, help='Trained state_dict file path to open')
# 加载网络模型结构，如果是pruned的模型，会加载携带网络模型参数
parser.add_argument('-v', '--version', default='slim_yolo_v2',
                    help='slim_yolo_v2, pruned_slim_yolo_v2')
# 裁减的比例，默认为0.8
parser.add_argument('--percent', type=float, default=0.8,
                    help='scale sparse rate (default: 0.8)')

parser.add_argument('-d', '--dataset', default='voc',
                    help='voc, coco-val, custom')
parser.add_argument('-size', '--input_size', default=224, type=int,
                    help='input_size')
parser.add_argument('--conf_thresh', default=0.1, type=float,
                    help='Confidence threshold')
parser.add_argument('--nms_thresh', default=0.50, type=float,
                    help='NMS threshold')
parser.add_argument('--visual_threshold', default=0.3, type=float,
                    help='Final confidence threshold')
parser.add_argument('--cuda', action='store_true', default=True, 
                    help='use cuda.')

args = parser.parse_args()

def vis(img, bboxes, scores, cls_inds, thresh, class_colors, class_names, class_indexs=None, dataset='custom'):
    if dataset == 'voc' or dataset == "widerface" or dataset == "custom":
        for i, box in enumerate(bboxes):
            cls_indx = cls_inds[i]
            xmin, ymin, xmax, ymax = box
            if scores[i] > thresh:
                cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), class_colors[int(cls_indx)], 1)
                cv2.rectangle(img, (int(xmin), int(abs(ymin)-20)), (int(xmax), int(ymin)), class_colors[int(cls_indx)], -1)
                mess = '%s, %.2f' % (class_names[int(cls_indx)], scores[i])
                cv2.putText(img, mess, (int(xmin), int(ymin-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

    elif dataset == 'coco-val' and class_indexs is not None:
        for i, box in enumerate(bboxes):
            cls_indx = cls_inds[i]
            xmin, ymin, xmax, ymax = box
            if scores[i] > thresh:
                cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), class_colors[int(cls_indx)], 1)
                cv2.rectangle(img, (int(xmin), int(abs(ymin)-20)), (int(xmax), int(ymin)), class_colors[int(cls_indx)], -1)
                cls_id = class_indexs[int(cls_indx)]
                cls_name = class_names[cls_id]
                # mess = '%s: %.3f' % (cls_name, scores[i])
                mess = '%s, %.2f' % (cls_name, scores[i])
                cv2.putText(img, mess, (int(xmin), int(ymin-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

    return img
        

def test(net, device, testset, transform, thresh, class_colors=None, class_names=None, class_indexs=None, dataset='custom'):
    num_images = len(testset)
    for index in range(num_images):
        print('Testing image {:d}/{:d}....'.format(index+1, num_images))
        img, _ = testset.pull_image(index)
        h, w, _ = img.shape

        # to tensor
        x = torch.from_numpy(transform(img)[0][:, :, (2, 1, 0)]).permute(2, 0, 1)
        x = x.unsqueeze(0).to(device)

        t0 = time.time()
        # forward
        bboxes, scores, cls_inds = net(x)
        print("detection time used ", time.time() - t0, "s")
        
        # scale each detection back up to the image
        scale = np.array([[w, h, w, h]])
        # map the boxes to origin image scale
        bboxes *= scale

        img_processed = vis(img, bboxes, scores, cls_inds, thresh, class_colors, class_names, class_indexs, dataset)
        if not os.path.exists("out/test"):
            os.makedirs("out/test")
        cv2.imwrite(f"out/test/{index}.jpg", img_processed)
        # cv2.imshow('detection', img_processed)
        # cv2.waitKey(0)
        # print('Saving the' + str(index) + '-th image ...')
        # cv2.imwrite('test_images/' + args.dataset+ '3/' + str(index).zfill(6) +'.jpg', img)

class TestDatasets(data.Dataset):
    """VOC Detection Dataset Object

    input is image, target is annotation

    Arguments:
        root (string): filepath to VOCdevkit folder.
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name (string, optional): which dataset to load
            (default: 'VOC2007')
    """

    def __init__(self, root,
                 image_sets=['data/test_images2'],
                 transform=None, target_transform=WiderfaceAnnotationTransform(),
                 ):
        self.root = root
        self.image_set = image_sets
        self.transform = transform
        self.target_transform = target_transform
        self.ids = list()
        for name in image_sets:
            if os.path.exists(name): # dir
                rootpath = name
                for f_name in os.listdir(name):
                    if f_name.endswith(".jpg") or f_name.endswith(".jpeg"):
                        self.ids.append((osp.join(name, f_name), f_name))
            else: # VOC style data
                rootpath = self.root
                for line in open(osp.join(rootpath, 'ImageSets', 'Main', name + '.txt')):
                    self.ids.append( (osp.join(rootpath, "JPEGImages", line.strip()), line.strip()) )

    def __getitem__(self, index):
        return self.pull_image(index)

    def __len__(self):
        return len(self.ids)

    def reset_transform(self, transform):
        self.transform = transform

    def pull_image(self, index):
        path, img_id = self.ids[index]
        return cv2.imread(path, cv2.IMREAD_COLOR), img_id

if __name__ == "__main__":
    # get device
    if args.cuda:
        print('use cuda')
        cudnn.benchmark = True
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    input_size = [args.input_size, args.input_size]

    # dataset
    print('--- Test on custom ...')
    class_names = CUSTOM_CLASSES
    class_indexs = None
    num_classes = len(CUSTOM_CLASSES)
    dataset = CustomDetection(root=CUSTOM_ROOT, image_sets=['val'], transform=None)

    class_colors = [(np.random.randint(255),np.random.randint(255),np.random.randint(255)) for _ in range(num_classes)]

    # 直接加载两个一样的模型
    weight_path = args.trained_model
    net = torch.load(weight_path)
    net_new = torch.load(weight_path)
    
    net.to(device)
    net_new.to(device)
    prune_size = [224, 224]
    net.trainable = False
    net.set_grid(prune_size)
    net.eval()
    net_new.trainable = False
    net_new.set_grid(prune_size)
    net_new.eval()
    
    print('--- Finished loading model!')

    # 原始模型
#     summary(net.to(device), input_size=(3, 224, 224), device="cpu")

    ######################
    # 剪枝流程中
    ######################
    print('--- Pruning starts ...')
    idxs = []
    idxs.append(range(3)) # 上一层的filter -> 是三维输入图片，所以这里给定的数据是3
    for module in net.modules():
        # print(module) # 打印一下模型结构
        if type(module) is nn.BatchNorm2d:
            weight = module.weight.data # weight和bias
            n = weight.size(0)
            y,idx = torch.sort(weight)
            # 我人傻了，我自己写的0.1然后我注释写的0.8。。。,乌鱼子
            n = int(args.percent * n) # 筛选xxx%的数据
            idxs.append(idx[:n])
    idxs.append(range(65)) # 添加最后一层的维度
    i=1 # 11, 24, 50, 101 batch中已经筛选出来的维度
    # print(f"len: {len(idxs)} idxs:{idxs}")
    for module in net_new.modules():
        # 最后一层也是没有bn层的，直接跳过
        # print(f"pruning : {i} ...")
        # print(f"Module : {type(module)}")
        if type(module) is nn.Conv2d:
            # 跳过v2的 route_layer 和 reorg 层的数据，否则会报错
            # 在slim 的 backbone里面好像没有用到这两个层，但是有定义，所以还是需要删除掉
            # i = 13 和 14

            if i == 13 or i == 14:
                continue

            weight = module.weight.data.clone() # 保存旧的tensor需要需要开辟新的存储地址而不是引用，可以使用clone()进行深拷贝
            # print(f"{i} : weight_size:{weight.size()} \n idxs[i]:{idxs[i]} \n idxs[i-1]:{idxs[i-3 if i == 15 else i-1]}")
            weight = weight[idxs[i],:,:,:] # 本层的 filter 数据

            # i-3 if i == 15 else i-1 当为15层的时候，跳过前面几层
            weight = weight[:,idxs[i-3 if i == 15 else i-1],:,:] # 上一层 filter 数据
            module.bias.data = module.bias.data[idxs[i]] # 本层的偏置
            module.weight.data = weight

            # 最后一层没有bn层，所以在此处不会计数，所以在这增加一下数值
            # 为啥不放在这里直接计数？
            # 那就是bn层是最后的，可能bn层前面有好几个conv，当然我这里没有。。。
            # emmm第16层不用管hhh，bn那边已经加到了16
            # if i == 15:
                # i += 1  

        elif type(module) is nn.BatchNorm2d:
            weight = module.weight.data.clone()
            bias = module.bias.data.clone()
            running_mean = module.running_mean.data.clone() # 在训练阶段，running_mean和running_var在每次前向的时候更新一次，测试阶段使用eval()固定BN层的running_mean和running_var，此时这两个值为训练阶段最后一次前向时候的确定值
            running_var = module.running_var.data.clone()
            
            weight = weight[idxs[i]] # bn层是有四个参数的！！！
            bias = bias[idxs[i]]
            running_mean = running_mean[idxs[i]]
            running_var = running_var[idxs[i]]

            module.weight.data = weight
            module.bias.data = bias
            module.running_var.data = running_var
            module.running_mean.data = running_mean
            i += 1 # BN层为最后一层的数据
        elif type(module) is nn.Linear:
            # print(module.weight.data.size())
            # 输出的结果不变，修改原有的数据[output, input]
            module.weight.data = module.weight.data[:,idxs[-1]] # 直接获取最后一层的数据
    print('--- Pruning ends ...')
    
    # 打印新的模型结构
    summary(net_new.to(device), input_size=(3, 224, 224))

    # 测试验证的效果
    with torch.no_grad():
        test(net=net_new, 
            device=device, 
            testset=dataset,
            transform=BaseTransform(input_size),
            thresh=args.visual_threshold,
            class_colors=class_colors,
            class_names=class_names,
            class_indexs=class_indexs,
            dataset='custom'
            )
    
    net.trainable = True
    net.set_grid(prune_size)
    net.train()
    net_new.trainable = True
    net_new.set_grid(prune_size)
    net_new.train()
    # 保存剪枝后的模型文件
    torch.save(net_new, "out/pruned_" + str(args.trained_model).split(".")[0].split("/")[1] + "_percent_" + str(int(args.percent * 100)) + ".pth")
    print("--- Pruned model save OK ...")