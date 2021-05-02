"""
Main file for training Yolo model on Pascal VOC and COCO dataset
"""

import os
import shutil
from torch._C import dtype
from torch.utils.data.dataloader import DataLoader
from dataset import DepthDataset, saveImgPredLabel
import config
import torch
import torchvision
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

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
from loss import DepthLoss, SILogLoss, YoloLoss
import warnings
warnings.filterwarnings("ignore")

torch.backends.cudnn.benchmark = True


def train_fn(train_loader, model, optimizer, loss_fn, scaler, scaled_anchors):
    loop = tqdm(train_loader, leave=True)
    losses = []
    for batch_idx, (x, y) in enumerate(loop):
        x = x.to(config.DEVICE)
        y0, y1, y2 = (
            y[0].to(config.DEVICE),
            y[1].to(config.DEVICE),
            y[2].to(config.DEVICE),
        )

        with torch.cuda.amp.autocast():
            out = model(x)
            loss = (
                loss_fn(out[0], y0, scaled_anchors[0])
                + loss_fn(out[1], y1, scaled_anchors[1])
                + loss_fn(out[2], y2, scaled_anchors[2])
            )

        losses.append(loss.item())
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update progress bar
        mean_loss = sum(losses) / len(losses)
        loop.set_postfix(loss=mean_loss)

def train_depth_fn(train_loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(train_loader, leave=True)
    losses = []
    for batch_idx, (x, y, depth_target) in enumerate(loop):
        x = x.to(config.DEVICE,dtype=torch.float)
        depth_target=depth_target.to(config.DEVICE)
        y0, y1, y2 = (
            y[0].to(config.DEVICE),
            y[1].to(config.DEVICE),
            y[2].to(config.DEVICE),
        )
        mask = depth_target > 1.0


        with torch.cuda.amp.autocast():
            out = model(x)
            out[0] = torch.clamp(out[0],1.0,255.0)
            out[1] = torch.clamp(out[1],1.0,255.0)
            out[2] = torch.clamp(out[2],1.0,255.0)
            loss = (
                loss_fn(out[0].permute(0,3,1,2), depth_target, mask=mask.to(torch.bool), interpolate=True)
                + loss_fn(out[1].permute(0,3,1,2), depth_target, mask=mask.to(torch.bool), interpolate=True)
                + loss_fn(out[2].permute(0,3,1,2), depth_target, mask=mask.to(torch.bool), interpolate=True)
            )

        losses.append(loss.item())
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update progress bar
        mean_loss = sum(losses) / len(losses)
        loop.set_postfix(loss=mean_loss)


def depthMain():
    dataset='Ostring'
    model = YOLOv3(num_classes=config.NUM_CLASSES).to(config.DEVICE)
    optimizer = optim.Adam(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )
    loss_fn=SILogLoss()
    scaler = torch.cuda.amp.GradScaler()
    if dataset=='DepthDataset':
        train_dataset = DepthDataset("DepthDataset/dataset.csv",
        "DepthDataset/imgs/",
        "DepthDataset/depth_imgs/"
        )
    if dataset=='Ostring':
        train_dataset = DepthDataset("OstringDepthDataset/OstringDataSet.csv",
        "OstringDepthDataset/imgs/",
        "OstringDepthDataset/depth_labels/"
        )
        test_dataset=train_dataset

    if dataset=='KITTI':
        train_dataset = DepthDataset("KITTI/kitti_eigen_train_files_with_gt.txt",
        "KITTI/",
        "KITTI/trainval_combined/"
        )
        test_dataset = DepthDataset("KITTI/kitti_eigen_test_files_with_gt.txt",
        "KITTI/",
        "KITTI/trainval_combined/")
    train_loader = DataLoader(dataset=train_dataset,batch_size=config.BATCH_SIZE,shuffle=False,num_workers=128)
    test_loader = DataLoader(dataset=test_dataset,batch_size=config.BATCH_SIZE,shuffle=True,num_workers=64)
    if config.LOAD_MODEL:
        load_checkpoint_transfer(
            config.CHECKPOINT_FILE, model, optimizer, config.LEARNING_RATE
        )
    
    dir = 'testOut'
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.mkdir(dir)

    dir = 'valOut'
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.mkdir(dir)
    for epoch in range(config.NUM_EPOCHS):
        # plot_couple_examples(model, test_loader, 0.3, 0.3, scaled_anchors,epochNo=epoch)

        if config.SAVE_MODEL:
            save_checkpoint(model, optimizer, filename=f"checkpoint.pth.tar")

        print(f"Currently epoch {epoch}")
        #print("On Train Eval loader:")
        #print("On Train loader:")
        #check_class_accuracy(model, train_loader, threshold=config.CONF_THRESHOLD)
        if epoch == 0 or epoch % 1 == 0:
            images, labels, depth_target =next(iter(train_loader))
            images = images.to(config.DEVICE,dtype=torch.float)
            model.eval()
            with torch.no_grad():
                testOut=model(images)
            model.train()
            testOut=[out/torch.max(out).item() for out in testOut]
            labels =[l/torch.max(l).item() for l in labels]
            depth_target =[dt/torch.max(dt).item() for dt in depth_target]
            
            for i in range(3):
                for j in range(8):
                    saveImgPredLabel(testOut[i][j],labels[i][j],images[j],depth_target[j],f'testOut/{i}_{j}_{epoch}_test.png')


            # meanLoss=evaluateDepth(test_loader,model,loss_fn)
            # f = open("lossoutput.txt","a")
            # f.write(f'{meanLoss}\n')
            # f.close
            

            images, labels, depth_target =next(iter(test_loader))
            images = images.to(config.DEVICE,dtype=torch.float)
            model.eval()
            with torch.no_grad():
                testOut=model(images)
            model.train()
            testOut=[out/torch.max(out).item() for out in testOut]
            labels =[l/torch.max(l).item() for l in labels]
            depth_target =[dt/torch.max(dt).item() for dt in depth_target]

            
            for i in range(3):
                for j in range(8):
                    saveImgPredLabel(testOut[i][j],labels[i][j],images[j],depth_target[j],f'valOut/{i}_{j}_{epoch}_test.png') 

        train_depth_fn(train_loader, model, optimizer, loss_fn, scaler)

def evaluateDepth(test_loader, model, loss_fn):
    loop = tqdm(test_loader,leave=True)
    for x,y,depthfullres in loop:
        model.eval()
        losses=[]
        x = x.to(config.DEVICE,dtype=torch.float)
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
            config.CHECKPOINT_FILE, model, optimizer, config.LEARNING_RATE
        )

    scaled_anchors = (
        torch.tensor(config.ANCHORS)
        * torch.tensor(config.S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    ).to(config.DEVICE)

    writer = SummaryWriter("myRun")
    images, labels = next(iter(train_loader))
    images = images.to(config.DEVICE)
    writer.add_scalar('test/scalar',20,10)
    writer.add_graph(torch.jit.trace(model, images, strict=False), [])
    writer.close
    for epoch in range(config.NUM_EPOCHS):
        plot_couple_examples(model, test_loader, 0.3, 0.3, scaled_anchors,epochNo=epoch)
        train_fn(train_loader, model, optimizer, loss_fn, scaler, scaled_anchors)

        #if config.SAVE_MODEL:
        save_checkpoint(model, optimizer, filename=f"checkpoint.pth.tar")

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
            f = open("logTraining.txt","a")
            f.write(f"MAP: {mapval.item()}")
            model.train()


if __name__ == "__main__":
    depthMain()