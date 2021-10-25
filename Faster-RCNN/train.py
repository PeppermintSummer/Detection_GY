import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils.callback import LossHistory
from models.frcnn import FasterRCNN
from models.frcnn_training import FasterRCNNTrainer, weights_init
from utils.dataloader import FRCNNDataset, frcnn_dataset_collate
from utils.utils import get_classes
from utils.utils_fit import fit_one_epoch


if __name__ == '__main__':
    Cuda=True
    classes_path = 'model_data/voc_classes.txt'
    model_path = 'model_data/voc_weights_resnet.pth'
    input_shape = [600, 600]
    backbone = 'resnet50'
    pretained=False
    #   anchors_size用于设定先验框的大小，每个特征点均存在9个先验框。
    #   anchors_size每个数对应3个先验框。
    #   当anchors_size = [8, 16, 32]的时候，生成的先验框宽高约为：
    #   [90, 180] ; [180, 360]; [360, 720]; [128, 128];
    #   [256, 256]; [512, 512]; [180, 90] ; [360, 180];
    #   [720, 360]; 详情查看anchors.py
    #   如果想要检测小物体，可以减小anchors_size靠前的数。
    #   比如设置anchors_size = [4, 16, 32]
    anchors_size = [8, 16, 32]
    #   冻结阶段训练参数
    #   此时模型的主干被冻结了，特征提取网络不发生改变
    #   占用的显存较小，仅对网络进行微调
    Init_Epoch = 0
    Freeze_Epoch = 50
    Freeze_batch_size = 4
    Freeze_lr = 1e-4
    #   此时模型的主干不被冻结了，特征提取网络会发生改变
    #   占用的显存较大，网络所有的参数都会发生改变
    UnFreeze_Epoch = 100
    Unfreeze_batch_size = 2
    Unfreeze_lr = 1e-5

    Freeze_Train = True # #   是否进行冻结训练，默认先冻结主干训练后解冻训练。
    num_workers = 4

    train_annotation_path = '2007_train.txt'
    val_annotation_path = '2007_val.txt'

    class_names,num_classes = get_classes(classes_path)

    model = FasterRCNN(num_classes,anchors_size,backbone,pretained)
    if pretained:
        weights_init(model)
    if model_path != '':
        print('load model:{}'.format(model_path))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_dict = model.state_dict()
        pretained_dict = torch.load(model_path,map_location=device)
        pretrained_dict = {k: v for k, v in pretained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
        model_dict.update(pretrained_dict)
        model.load_state_dict()

    model_train = model.train()
    if Cuda:
        model_train = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        model_train=model_train.cuda()

    loss_history = LossHistory('logs/')

    with open(train_annotation_path) as f:
        train_lines = f.readlines()
    with open(val_annotation_path) as f:
        val_lines = f.readlines()
    num_train = len(train_lines)
    num_val = len(val_lines)

    batch_size = Freeze_batch_size
    lr = Freeze_lr
    start_epoch = Init_Epoch
    end_epoch = Freeze_Epoch

    optimizer = optim.Adam(model_train.parameters(), lr, weight_decay=5e-4)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.96)

    train_dataset = FRCNNDataset(train_lines, input_shape, train=True)
    val_dataset = FRCNNDataset(val_lines, input_shape, train=False)
    gen = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                     drop_last=True, collate_fn=frcnn_dataset_collate)
    gen_val = DataLoader(val_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                         drop_last=True, collate_fn=frcnn_dataset_collate)

    epoch_step = num_train // batch_size
    epoch_step_val = num_val // batch_size

    if epoch_step == 0 or epoch_step_val == 0:
        raise ValueError("数据集过小，无法进行训练，请扩充数据集。")

    # ------------------------------------#
    #   冻结一定部分训练
    # ------------------------------------#
    if Freeze_Train:
        for param in model.extractor.parameters():
            param.requires_grad = False

    # ------------------------------------#
    #   冻结bn层
    # ------------------------------------#
    model.freeze_bn()

    train_util = FasterRCNNTrainer(model, optimizer)

    for epoch in range(start_epoch, end_epoch):
        fit_one_epoch(model, train_util, loss_history, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val,
                      end_epoch, Cuda)
        lr_scheduler.step()

    batch_size = Unfreeze_batch_size
    lr = Unfreeze_lr
    start_epoch = Freeze_Epoch
    end_epoch = UnFreeze_Epoch

    optimizer = optim.Adam(model_train.parameters(), lr, weight_decay=5e-4)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.96)

    train_dataset = FRCNNDataset(train_lines, input_shape, train=True)
    val_dataset = FRCNNDataset(val_lines, input_shape, train=False)
    gen = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                     drop_last=True, collate_fn=frcnn_dataset_collate)
    gen_val = DataLoader(val_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                         drop_last=True, collate_fn=frcnn_dataset_collate)

    epoch_step = num_train // batch_size
    epoch_step_val = num_val // batch_size

    if epoch_step == 0 or epoch_step_val == 0:
        raise ValueError("数据集过小，无法进行训练，请扩充数据集。")

    # ------------------------------------#
    #   冻结一定部分训练
    # ------------------------------------#
    if Freeze_Train:
        for param in model.extractor.parameters():
            param.requires_grad = True

    # ------------------------------------#
    #   冻结bn层
    # ------------------------------------#
    model.freeze_bn()

    train_util = FasterRCNNTrainer(model, optimizer)

    for epoch in range(start_epoch, end_epoch):
        fit_one_epoch(model, train_util, loss_history, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val,
                      end_epoch, Cuda)
        lr_scheduler.step()
