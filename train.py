"""
Main file for training Yolo model on Pascal VOC and COCO dataset
"""

import os
from pathlib import Path
import shutil
from albumentations.pytorch import transforms
from matplotlib.pyplot import box
from torch._C import dtype
from torch.utils.data.dataloader import DataLoader
from dataset import  YOLODataset, saveImgPredLabel
import config
import torch
import torchvision
import torch.optim as optim
import signal
import sys
from torch.utils.tensorboard import SummaryWriter, writer

from model import YOLOv3
from tqdm import tqdm
from utils import (
    load_checkpoint_transfer,
    mean_average_precision,
    cells_to_bboxes,
    get_evaluation_bboxes,
    save_checkpoint,
    load_checkpoint,
    check_class_accuracy,
    get_loaders,
    plot_couple_examples
)
from loss import YoloLoss
import warnings
warnings.filterwarnings("ignore")

torch.backends.cudnn.benchmark = True

tb = SummaryWriter(config.LOG_DIR)

def sigInterrupt_handler(sig, frame):
    print('You pressed Ctrl+C! closing tb writer')
    tb.close()
    sys.exit(0)


def train_fn(train_loader, model, optimizer, loss_fn, scaler, scaled_anchors):
    loop = tqdm(train_loader, leave=True)
    for batch_idx, (x, y, depth_target) in enumerate(loop):
        losses = []

        x = x.to(config.DEVICE)
        y0, y1, y2 = (
            y[0].to(config.DEVICE),
            y[1].to(config.DEVICE),
            y[2].to(config.DEVICE),
        )

        with torch.cuda.amp.autocast():
            out = model(x)
            loss = (
                loss_fn(out[0], y0, scaled_anchors[0], depth_target, config.S[0])
                + loss_fn(out[1], y1, scaled_anchors[1], depth_target, config.S[1])
                + loss_fn(out[2], y2, scaled_anchors[2], depth_target, config.S[2])
            )

        losses.append(loss.item())
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update progress bar
        mean_loss = sum(losses) / len(losses)
        loop.set_postfix(loss=loss.item())


def train_depth_fn(train_loader, model, optimizer, loss_fn, scaler,epoch):
    loop = tqdm(train_loader, leave=True)
    scaled_anchors = (
        torch.tensor(config.ANCHORS)
        * torch.tensor(config.S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    ).to(config.DEVICE)
    losses = []
    for batch_idx, (x, y, depth_target) in enumerate(loop):
        global_step = epoch * len(loop) + batch_idx
        x = x.to(config.DEVICE, dtype=torch.float)
        depth_target = depth_target.to(config.DEVICE)
        y0, y1, y2 = (
            y[0].to(config.DEVICE),
            y[1].to(config.DEVICE),
            y[2].to(config.DEVICE),
        )

        with torch.cuda.amp.autocast():
            out = model(x)
            loss = []
            (box_loss0, obj_loss0, noObj_loss0, cls_loss0, depth_loss0,noobj_depth_loss0,depth_stats0) = loss_fn(out[0], y0, depth_target, scaled_anchors[0], config.S[0])
            (box_loss1, obj_loss1, noObj_loss1, cls_loss1, depth_loss1,noobj_depth_loss1,depth_stats1) = loss_fn(out[1], y1, depth_target, scaled_anchors[1], config.S[1])
            (box_loss2, obj_loss2, noObj_loss2, cls_loss2, depth_loss2,noobj_depth_loss2,depth_stats2) = loss_fn(out[2], y2, depth_target, scaled_anchors[2], config.S[2])
            # loss = (
            #         loss_fn(out[0],y0, depth_target,scaled_anchors[0],config.S[0])+
            #         loss_fn(out[1],y1, depth_target,scaled_anchors[1],config.S[1])+
            #         loss_fn(out[2],y2, depth_target,scaled_anchors[2],config.S[2])
            # )
        box_loss = box_loss0+box_loss1+box_loss2
        obj_loss = obj_loss0+obj_loss1+obj_loss2
        noObj_loss = noObj_loss0+noObj_loss1+noObj_loss2
        cls_loss = cls_loss0+cls_loss1+cls_loss2
        depth_loss = depth_loss0+depth_loss1+depth_loss2
        noobj_depth_loss = noobj_depth_loss0 + noobj_depth_loss1 + noobj_depth_loss2
        total_loss = box_loss+obj_loss+noObj_loss+cls_loss+depth_loss + noobj_depth_loss
        loss = total_loss

        tb.add_scalars('Loss_Train/partial',{'box':box_loss,'obj':obj_loss,'noobj':noObj_loss,'cls':cls_loss,'obj_depth':depth_loss,'noobj_depth':noobj_depth_loss,'total':loss},global_step)
        # Loss / batch size , normalized!
        loop_description = f't:{loss:0f}box:{box_loss:1f},obj:{obj_loss:1f},noObj:{noObj_loss:1f},cls:{cls_loss:1f},obj_depth:{depth_loss:1f},noObjDepth:{noobj_depth_loss:1f}'
        losses.append(loss.item())
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        # loss.backward()
        # optimizer.step()
        scaler.step(optimizer)
        scaler.update()

        # update progress bar
        mean_loss = sum(losses) / len(losses)
        loop.set_postfix_str(loop_description)
        tb.add_scalar('Loss_Train/mean',mean_loss,global_step)

def eval_Val_Loss(test_loader, model, loss_fn,epoch):
    model.eval()
    print('Evaluate Test Set Loss')
    loop = tqdm(test_loader, leave=True)
    scaled_anchors = (
        torch.tensor(config.ANCHORS)
        * torch.tensor(config.S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    ).to(config.DEVICE)
    losses = []
    for batch_idx, (x, y, depth_target) in enumerate(loop):
        global_step = epoch * len(loop) + batch_idx
        x = x.to(config.DEVICE, dtype=torch.float)
        depth_target = depth_target.to(config.DEVICE)
        y0, y1, y2 = (
            y[0].to(config.DEVICE),
            y[1].to(config.DEVICE),
            y[2].to(config.DEVICE),
        )

        with torch.no_grad():
            out = model(x)
            loss = []
            (box_loss0, obj_loss0, noObj_loss0, cls_loss0, depth_loss0,noobj_depth_loss0,depth_stats0) = loss_fn(out[0], y0, depth_target, scaled_anchors[0], config.S[0])
            (box_loss1, obj_loss1, noObj_loss1, cls_loss1, depth_loss1,noobj_depth_loss1,depth_stats1) = loss_fn(out[1], y1, depth_target, scaled_anchors[1], config.S[1])
            (box_loss2, obj_loss2, noObj_loss2, cls_loss2, depth_loss2,noobj_depth_loss2,depth_stats2) = loss_fn(out[2], y2, depth_target, scaled_anchors[2], config.S[2])
 # loss = (
            #         loss_fn(out[0],y0, depth_target,scaled_anchors[0],config.S[0])+
            #         loss_fn(out[1],y1, depth_target,scaled_anchors[1],config.S[1])+
            #         loss_fn(out[2],y2, depth_target,scaled_anchors[2],config.S[2])
            # )
        box_loss = box_loss0+box_loss1+box_loss2
        obj_loss = obj_loss0+obj_loss1+obj_loss2
        noObj_loss = noObj_loss0+noObj_loss1+noObj_loss2
        cls_loss = cls_loss0+cls_loss1+cls_loss2
        depth_loss = depth_loss0+depth_loss1+depth_loss2
        noobj_depth_loss = noobj_depth_loss0 + noobj_depth_loss1 + noobj_depth_loss2
        total_loss = box_loss+obj_loss+noObj_loss+cls_loss+depth_loss + noobj_depth_loss
        loss = total_loss

        tb.add_scalars('Loss_Test/partial',{'box':box_loss,'obj':obj_loss,'noobj':noObj_loss,'cls':cls_loss,'obj_depth':depth_loss,'noobj_depth':noobj_depth_loss,'total':loss},global_step)
        tb.add_scalars('DepthErrorStats/scale0',depth_stats0,global_step)
        tb.add_scalars('DepthErrorStats/scale1',depth_stats1,global_step)
        tb.add_scalars('DepthErrorStats/scale2',depth_stats2,global_step)
        # Loss / batch size , normalized!
        loop_description = f'Test t:{loss:0f}box:{box_loss:1f},obj:{obj_loss:1f},noObj:{noObj_loss:1f},cls:{cls_loss:1f},obj_depth:{depth_loss:1f},noObjDepth:{noobj_depth_loss:1f}'
        losses.append(loss.item())

        # update progress bar
        mean_loss = sum(losses) / len(losses)
        tb.add_scalar('Loss_Test/mean',mean_loss,global_step)
        loop.set_postfix_str(loop_description)
    model.train()


def depthMain():
    Path(config.TRAINING_EXAMPLES_PLOT_DIR_DEPTH).mkdir(parents=True,exist_ok=True)
    Path(config.LOG_DIR).mkdir(parents=True,exist_ok=True)
    shutil.copyfile('config.py',config.LOG_DIR+'/0_backup_config.py', follow_symlinks=True)
    signal.signal(signal.SIGINT, sigInterrupt_handler)
    dataset = config.DATASET
    model = YOLOv3(num_classes=config.NUM_CLASSES).to(config.DEVICE)
    optimizer = optim.Adam(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )
    loss_fn = YoloLoss()
    scaler = torch.cuda.amp.GradScaler()

    scaled_anchors = (
        torch.tensor(config.ANCHORS)
        * torch.tensor(config.S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    ).to(config.DEVICE)

    if dataset == 'DepthAndObjectDetection':
        train_dataset = YOLODataset(
            config.DATASET_TRAIN_CSV,
            config.IMAGE_DIR,
            config.BBOX_LABEL_DIR,
            config.DEPTH_NN_MAP_LABEL,
            S=config.S,
            anchors=config.ANCHORS,
            transform=None,
            cache_images=True
        )
        # TODO make test dataset csv etc
        test_dataset = YOLODataset(
            config.DATASET_TEST_CSV,
            config.IMAGE_DIR,
            config.BBOX_LABEL_DIR,
            config.DEPTH_NN_MAP_LABEL,
            S=config.S,
            anchors=config.ANCHORS,
            transform=None,
            cache_images=True

        )

    if dataset =='ObjectDetection':
        # TODO Dataset handling missing depth maps.
        train_dataset = YOLODataset(
            config.DATASET_TRAIN_CSV,
            config.IMAGE_DIR,
            config.BBOX_LABEL_DIR,
            None,
            S=config.S,
            anchors=config.ANCHORS,
            transform=None,
            cache_images=True

        )
        test_dataset = YOLODataset(
            config.DATASET_TEST_CSV,
            config.IMAGE_DIR,
            config.BBOX_LABEL_DIR,
            None,
            S=config.S,
            anchors=config.ANCHORS,
            transform=None,
            cache_images=True

        )

    # if dataset == 'KITTI':
    #     train_dataset = DepthDataset("KITTI/kitti_eigen_train_files_with_gt.txt",
    #                                  "KITTI/",
    #                                  "KITTI/trainval_combined/"
    #                                  )
    #     test_dataset = DepthDataset("KITTI/kitti_eigen_test_files_with_gt.txt",
    #                                 "KITTI/",
    #                                 "KITTI/trainval_combined/")

    train_loader = DataLoader(dataset=train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS)
    test_loader = DataLoader(dataset=test_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS)
    if config.LOAD_MODEL:
        if config.TRANSFER_MODEL:
            load_checkpoint_transfer(
                config.LOAD_CHECKPOINT_FILE, model, optimizer, config.LEARNING_RATE
            )
        else:
            load_checkpoint(
                config.LOAD_CHECKPOINT_FILE, model, optimizer, config.LEARNING_RATE
            )


    dir = 'testOutYoloDepth'
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.mkdir(dir)

    dir = 'valOutYoloDepth'
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.mkdir(dir)

    cache_loader = DataLoader(dataset=train_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS)
    for batch_indx, (x,y,depthtarget) in tqdm(enumerate(cache_loader)):
        for i in range(0,x.shape[0]):
            idx=batch_indx*config.BATCH_SIZE+i
            train_dataset.imgs[idx] = x[i]
            train_dataset.targets[idx] = (y[0][i],y[1][i],y[2][i])
            train_dataset.depth_targets[idx] = depthtarget[i]
    train_dataset.isCached=True

    cache_loader = DataLoader(dataset=test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS)
    for batch_indx, (x,y,depthtarget) in tqdm(enumerate(cache_loader)):
        for i in range(0,x.shape[0]):
            idx=batch_indx*config.BATCH_SIZE+i
            test_dataset.imgs[idx] = x[i]
            test_dataset.targets[idx] = (y[0][i],y[1][i],y[2][i])
            test_dataset.depth_targets[idx] = depthtarget[i]
    test_dataset.isCached = True

    del cache_loader
    for epoch in range(config.NUM_EPOCHS):

        plot_couple_examples(model, test_loader, config.CONF_THRESHOLD, config.NMS_IOU_THRESH, scaled_anchors,noOfExamples=2, epochNo=epoch)
        
        if config.SAVE_MODEL:
            if epoch%2==0:
                save_checkpoint(model, optimizer, filename=config.SAVE_CHECKPOINT_FILE)
            else:
                save_checkpoint(model, optimizer, filename='backup_'+config.SAVE_CHECKPOINT_FILE)

        print(f"Currently epoch {epoch}")
        print("On Train Eval loader:")
        # TODO implement testing of yoloDepth

        # if epoch == 0 or (epoch > 1 and epoch % 3 == 0) and config.DATASET=='ObjectDetection':
        if (epoch >= 0 and epoch % 3 == 0) : 
        # if False :
            clsAccuracy,noobjAccuracy,objAccuracy = check_class_accuracy(model, test_loader, threshold=config.CONF_THRESHOLD)
            # TODO fix nms performance!
            pred_boxes, true_boxes = get_evaluation_bboxes(
                test_loader,
                model,
                iou_threshold=config.NMS_IOU_THRESH,
                anchors=config.ANCHORS,
                threshold=config.CONF_THRESHOLD,
                device=config.DEVICE
            )
            mapval_per_class = mean_average_precision(
                pred_boxes,
                true_boxes,
                iou_threshold=config.MAP_IOU_THRESH,
                box_format="midpoint",
                num_classes=config.NUM_CLASSES,
                reduction=False
            )
            print("=============== MAPs: ==============")
            for idx, mapc in enumerate(mapval_per_class):
                print(f"Class: {config.MVD_CLASSES[idx]}: {mapc}")
                tb.add_scalar(f'mAP_Val/{idx}',mapc,epoch)
            f = open("logTraining.txt", "a")
            tb.add_scalar(f'mAP_Val/sum',sum(mapval_per_class)/len(mapval_per_class),epoch)
            tb.add_scalar(f'Accuracy_Val/class',clsAccuracy,epoch)
            tb.add_scalar(f'Accuracy_Val/noObj',noobjAccuracy,epoch)
            tb.add_scalar(f'Accuracy_Val/Obj',objAccuracy,epoch)
            f.write(f"MAP: {sum(mapval_per_class) / len(mapval_per_class)}")



        # check_class_accuracy(model, test_loader, threshold=config.CONF_THRESHOLD)
        # if epoch == 0 or epoch % 1 == 0:
        #     images, labels, depth_target =next(iter(train_loader))
        #     images = images.to(config.DEVICE,dtype=torch.float)
        # model.eval()
        # with torch.no_grad():
        #     testOut=model(images)
        # model.train()
        # # testout is complete 3 scaled prediction tensor
        # testOut=[out/torch.max(out).item() for out in testOut]
        # labels =[l/torch.max(l).item() for l in labels]
        # depth_target =[dt/torch.max(dt).item() for dt in depth_target]

        # for i in range(3):
        #     for j in range(8):
        #         saveImgPredLabel(testOut[i][j],labels[i][j],images[j],depth_target[j],f'testOut/{i}_{j}_{epoch}_test.png')

        # meanLoss=evaluateDepth(test_loader,model,loss_fn)
        # f = open("lossoutput.txt","a")
        # f.write(f'{meanLoss}\n')
        # f.close
        
        if True:
            print("On Test loader:")
            eval_Val_Loss(test_loader,model,loss_fn,epoch)
        
        print("On Train loader:")
        train_depth_fn(train_loader, model, optimizer, loss_fn, scaler,epoch=epoch)


def evaluateDepth(test_loader, model, loss_fn):
    loop = tqdm(test_loader, leave=True)
    for x, y, depthfullres in loop:
        model.eval()
        losses = []
        x = x.to(config.DEVICE, dtype=torch.float)
        y0, y1, y2 = (
            y[0].to(config.DEVICE),
            y[1].to(config.DEVICE),
            y[2].to(config.DEVICE),
        )

        with torch.cuda.amp.autocast():
            with torch.no_grad():
                out = model(x)
                loss = (
                    loss_fn(out[0], y0)
                    + loss_fn(out[1], y1)
                    + loss_fn(out[2], y2)
                )
        losses.append(loss.item())

        mean_loss = sum(losses) / len(losses)
        loop.set_postfix(valLoss=mean_loss)

        model.train()
    return mean_loss

def main():
    model = YOLOv3(num_classes=config.NUM_CLASSES).to(config.DEVICE)
    optimizer = optim.Adam(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )
    loss_fn = YoloLoss()
    scaler = torch.cuda.amp.GradScaler()

    train_loader, test_loader, train_eval_loader = get_loaders(
        train_csv_path=config.DATASET + "/train.csv", test_csv_path=config.DATASET + "/test.csv"
    )

    if config.LOAD_MODEL:
        load_checkpoint(
            config.LOAD_CHECKPOINT_FILE, model, optimizer, config.LEARNING_RATE
        )

    scaled_anchors = (
        torch.tensor(config.ANCHORS)
        * torch.tensor(config.S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    ).to(config.DEVICE)

    images, labels = next(iter(train_loader))
    images = images.to(config.DEVICE)
    writer.add_graph(torch.jit.trace(model, images, strict=False), [])
    writer.close
    for epoch in range(config.NUM_EPOCHS):
        plot_couple_examples(model, test_loader, 0.3, 0.3, scaled_anchors, epochNo=epoch)
        train_fn(train_loader, model, optimizer, loss_fn, scaler, scaled_anchors)

        # if config.SAVE_MODEL:
        save_checkpoint(model, optimizer, filename=config.SAVE_CHECKPOINT_FILE)

        print(f"Currently epoch {epoch}")
        #print("On Train Eval loader:")
        #print("On Train loader:")
        #check_class_accuracy(model, train_loader, threshold=config.CONF_THRESHOLD)

        if epoch > 0 and epoch % 3 == 0:
            check_class_accuracy(model, test_loader, threshold=config.CONF_THRESHOLD)
            # TODO fix nms performance!
            pred_boxes, true_boxes = get_evaluation_bboxes(
                test_loader,
                model,
                iou_threshold=config.NMS_IOU_THRESH,
                anchors=config.ANCHORS,
                threshold=config.CONF_THRESHOLD,
                device=config.DEVICE
            )
            mapval = mean_average_precision(
                pred_boxes,
                true_boxes,
                iou_threshold=config.MAP_IOU_THRESH,
                box_format="midpoint",
                num_classes=config.NUM_CLASSES,
            )
            print(f"MAP: {mapval.item()}")
            f = open("logTraining.txt", "a")
            f.write(f"MAP: {mapval.item()}")
            model.train()


if __name__ == "__main__":
    depthMain()
