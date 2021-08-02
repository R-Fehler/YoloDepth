"""
Main file for training Yolo model on Pascal VOC and COCO dataset
"""

import os
from pathlib import Path
import shutil
from matplotlib.pyplot import box
from torch.utils.data.dataloader import DataLoader
from dataset import  YOLODataset, saveImgPredLabel, test
import config
import torch
import torch.optim as optim
import signal
import sys
import glob
from torch.utils.tensorboard import SummaryWriter, writer
from torchsummary import summary

from model import YOLOv3
from tqdm import tqdm
from utils import (
    load_checkpoint_transfer,
    mean_average_precision,
    get_evaluation_bboxes,
    save_checkpoint,
    load_checkpoint,
    check_class_accuracy,
    plot_couple_examples
)
from loss import YoloLoss
import warnings
warnings.filterwarnings("ignore")

torch.backends.cudnn.benchmark = True

tb = SummaryWriter(config.TB_SUB_DIR)

def sigInterrupt_handler(sig, frame):
    print('You pressed Ctrl+C! closing tb writer')
    tb.close()
    sys.exit(0)


def train_fn(detect_train_loader,depth_only_loader, model, optimizer, loss_fn:YoloLoss, scaler,epoch):
    detect_iterator = tqdm(detect_train_loader, leave=True)
    scaled_anchors = (
        torch.tensor(config.ANCHORS)
        * torch.tensor(config.S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    ).to(config.DEVICE)

    depth_anchors = (
        torch.tensor(config.DEPTH_ANCHORS)
    ).to(config.DEVICE)
    # idea for second dataset:
    depth_only_loader_iterator=iter(depth_only_loader)

    for batch_idx, (x_detect, y_detect, depth_target_detect, _) in enumerate(detect_iterator):
        try:
            # Samples the batch
            x_depthDS, y_depthDS, depthMap_depthDS, _ = next(depth_only_loader_iterator)
        except StopIteration:
            # restart the dataloader if the previous is exhausted. (reshuffling dataset etc.)
            depth_only_loader_iterator = iter(depth_only_loader)
            x_depthDS, y_depthDS, depthMap_depthDS, _ = next(depth_only_loader_iterator)
            print('repeating depth dataset: reshuffling dataloader')

        x=torch.cat((x_detect,x_depthDS),0) # concat the two dataset items

        # permute the items in batch randomly to make sure no weird stuff happens, esoteric, not analyzed
        if x.shape[0]>1:
            perm_idx = torch.randperm(x.shape[0]) 
        y=tuple( [torch.cat((y_i,y_depthDS_i),0) for y_i,y_depthDS_i in zip(y_detect,y_depthDS)] ) # tuple with 3 scale targets, right format
        depth_target=torch.cat((depth_target_detect,depthMap_depthDS),0)
        # execute the permutation
        if x.shape[0]>1:
            x = x[perm_idx]
            y = (y[0][perm_idx], y[1][perm_idx], y[2][perm_idx])
            depth_target[perm_idx]

        global_step = epoch * len(detect_iterator) + batch_idx
        x = x.to(config.DEVICE, dtype=torch.float)
        depth_target = depth_target.to(config.DEVICE)
        y0, y1, y2 = (
            y[0].to(config.DEVICE),
            y[1].to(config.DEVICE),
            y[2].to(config.DEVICE),
        )

        with torch.cuda.amp.autocast():
            # run forward pass
            out = model(x)
            loss = []
            # calc losses
            (box_loss0, obj_loss0, noObj_loss0, cls_loss0, depth_loss_from_map0,noobj_depth_loss0,obj_depth_loss_from_lidar0,distribution_stats0,depth_stats0) = loss_fn(out[0], y0, depth_target, scaled_anchors[0], config.S[0],depth_anchors[0],tb)
            (box_loss1, obj_loss1, noObj_loss1, cls_loss1, depth_loss_from_map1,noobj_depth_loss1,obj_depth_loss_from_lidar1,distribution_stats1,depth_stats1) = loss_fn(out[1], y1, depth_target, scaled_anchors[1], config.S[1],depth_anchors[1],tb)
            (box_loss2, obj_loss2, noObj_loss2, cls_loss2, depth_loss_from_map2,noobj_depth_loss2,obj_depth_loss_from_lidar2,distribution_stats2,depth_stats2) = loss_fn(out[2], y2, depth_target, scaled_anchors[2], config.S[2],depth_anchors[2],tb)
        # add losses from different scales to specific term
        box_loss = box_loss0+box_loss1+box_loss2
        obj_loss = obj_loss0+obj_loss1+obj_loss2
        noObj_loss = noObj_loss0+noObj_loss1+noObj_loss2
        cls_loss = cls_loss0+cls_loss1+cls_loss2
        depth_loss_from_map = depth_loss_from_map0+depth_loss_from_map1+depth_loss_from_map2
        noobj_depth_loss = noobj_depth_loss0 + noobj_depth_loss1 + noobj_depth_loss2
        obj_depth_loss_from_lidar = obj_depth_loss_from_lidar0 + obj_depth_loss_from_lidar1 + obj_depth_loss_from_lidar2
        total_loss = box_loss+obj_loss+noObj_loss+cls_loss+depth_loss_from_map + noobj_depth_loss + obj_depth_loss_from_lidar
        loss = total_loss

        tb.add_scalars('Loss_Train/partial',{'box':box_loss,'obj':obj_loss,'noobj':noObj_loss,'cls':cls_loss,'obj_depth':depth_loss_from_map,'noobj_depth':noobj_depth_loss,'obj_depth_loss_from_lidar':obj_depth_loss_from_lidar,'total':loss},global_step)
        tb.add_scalars('DepthErrorStats/scale0',depth_stats0,global_step)
        tb.add_scalars('DepthErrorStats/scale1',depth_stats1,global_step)
        tb.add_scalars('DepthErrorStats/scale2',depth_stats2,global_step)
        for key,value in distribution_stats0.items():
            if value.shape[0]!=0:
                tb.add_histogram('Train_Distributions/scale0'+key,value,global_step) 
        for key,value in distribution_stats1.items():
            if value.shape[0]!=0:
                tb.add_histogram('Train_Distributions/scale1'+key,value,global_step)       
        for key,value in distribution_stats2.items():
            if value.shape[0]!=0:
                tb.add_histogram('Train_Distributions/scale2'+key,value,global_step)
        # Loss / batch size , normalized!
        loop_description = f't:{loss:0f}box:{box_loss:1f},obj:{obj_loss:1f},noObj:{noObj_loss:1f},cls:{cls_loss:1f},obj_depth:{depth_loss_from_map:1f},noObjDepth:{noobj_depth_loss:1f}'
        
        # do backward pass
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        # when not using grad scaler and automatic precision fp 16 etc
        # loss.backward()
        # optimizer.step()
        
        # optimize model parameters
        scaler.step(optimizer)
        scaler.update()

        # update progress bar
        # running mean loss
        detect_iterator.set_postfix_str(loop_description)
        tb.add_scalar('Loss_Train/Loss',loss,global_step)

        
def eval_Val_Loss(test_loader, model, loss_fn:YoloLoss,epoch,dataset_name=''):
    model.eval()
    print('Evaluate Test Set Loss')
    loop = tqdm(test_loader, leave=True)
    scaled_anchors = (
        torch.tensor(config.ANCHORS)
        * torch.tensor(config.S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    ).to(config.DEVICE)

    depth_anchors = (
        torch.tensor(config.DEPTH_ANCHORS)
    ).to(config.DEVICE)
    losses = []
    for batch_idx, (x, y, depth_target,_) in enumerate(loop):
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
            (box_loss0, obj_loss0, noObj_loss0, cls_loss0, depth_loss_from_map0,noobj_depth_loss0,obj_depth_loss_from_lidar0,distribution_stats0,depth_stats0) = loss_fn(out[0], y0, depth_target, scaled_anchors[0], config.S[0],depth_anchors[0],tb)
            (box_loss1, obj_loss1, noObj_loss1, cls_loss1, depth_loss_from_map1,noobj_depth_loss1,obj_depth_loss_from_lidar1,distribution_stats1,depth_stats1) = loss_fn(out[1], y1, depth_target, scaled_anchors[1], config.S[1],depth_anchors[1],tb)
            (box_loss2, obj_loss2, noObj_loss2, cls_loss2, depth_loss_from_map2,noobj_depth_loss2,obj_depth_loss_from_lidar2,distribution_stats2,depth_stats2) = loss_fn(out[2], y2, depth_target, scaled_anchors[2], config.S[2],depth_anchors[2],tb)

        box_loss = box_loss0+box_loss1+box_loss2
        obj_loss = obj_loss0+obj_loss1+obj_loss2
        noObj_loss = noObj_loss0+noObj_loss1+noObj_loss2
        cls_loss = cls_loss0+cls_loss1+cls_loss2
        depth_loss_from_map = depth_loss_from_map0+depth_loss_from_map1+depth_loss_from_map2
        noobj_depth_loss = noobj_depth_loss0 + noobj_depth_loss1 + noobj_depth_loss2
        obj_depth_loss_from_lidar = obj_depth_loss_from_lidar0 + obj_depth_loss_from_lidar1 + obj_depth_loss_from_lidar2
        total_loss = box_loss+obj_loss+noObj_loss+cls_loss+depth_loss_from_map + noobj_depth_loss + obj_depth_loss_from_lidar
        loss = total_loss

        tb.add_scalars(dataset_name+'Loss_Test/partial',{'box':box_loss,'obj':obj_loss,'noobj':noObj_loss,'cls':cls_loss,'obj_depth':depth_loss_from_map,'noobj_depth':noobj_depth_loss,'obj_depth_loss_from_lidar':obj_depth_loss_from_lidar,'total':loss},global_step)
        tb.add_scalars(dataset_name+'DepthErrorStats/scale0',depth_stats0,global_step)
        tb.add_scalars(dataset_name+'DepthErrorStats/scale1',depth_stats1,global_step)
        tb.add_scalars(dataset_name+'DepthErrorStats/scale2',depth_stats2,global_step)
        for key,value in distribution_stats0.items():
            if value.shape[0]!=0:
                tb.add_histogram(dataset_name+'Test_Distributions/scale0'+key,value,global_step) 
        for key,value in distribution_stats1.items():
            if value.shape[0]!=0:
                tb.add_histogram(dataset_name+'Test_Distributions/scale1'+key,value,global_step)       
        for key,value in distribution_stats2.items():
            if value.shape[0]!=0:
                tb.add_histogram(dataset_name+'Test_Distributions/scale2'+key,value,global_step)
        # Loss / batch size , normalized!
        loop_description = dataset_name+f'Test t:{loss:0f}box:{box_loss:1f},obj:{obj_loss:1f},noObj:{noObj_loss:1f},cls:{cls_loss:1f},obj_depth:{depth_loss_from_map:1f},noObjDepth:{noobj_depth_loss:1f}'
        losses.append(loss.item())

        # update progress bar
        mean_loss = sum(losses) / len(losses)
        tb.add_scalar(dataset_name+'Loss_Test/mean',mean_loss,global_step)
        loop.set_postfix_str(loop_description)

    model.train()
    return mean_loss


def main():
    # Backups and making directories 
    Path(config.TRAINING_EXAMPLES_PLOT_DIR_DEPTH).mkdir(parents=True,exist_ok=True)
    Path(config.TB_SUB_DIR).mkdir(parents=True,exist_ok=True)
    Path(config.SRC_CODE_BACKUP).mkdir(parents=True,exist_ok=True)
    if config.SAVE_SRC_AND_CONFIG:
        for src_code_file in glob.glob(r'*.py'):
            shutil.copyfile(src_code_file,config.SRC_CODE_BACKUP+src_code_file, follow_symlinks=True)
            print(src_code_file,'is backed up')
    signal.signal(signal.SIGINT, sigInterrupt_handler)
    
    # lower upper limits for saving best models
    bestMAP=0
    minValidationLoss_MVD=1.0e+9
    minValidationLoss_OST=1.0e+9
    lastBestModelName=''

    model = YOLOv3(num_classes=config.NUM_CLASSES)
    summary(model,(3,config.IMAGE_SIZE,config.IMAGE_SIZE),device='cpu')
    model = model.to(config.DEVICE)
    optimizer = optim.Adam(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )
    loss_fn = YoloLoss()
    scaler = torch.cuda.amp.GradScaler()

    scaled_anchors = (
        torch.tensor(config.ANCHORS)
        * torch.tensor(config.S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    ).to(config.DEVICE)

    depth_anchors = (
        torch.tensor(config.DEPTH_ANCHORS)
    ).to(config.DEVICE)
    
    #  Ostring Datasets 
    # =======================
    train_dataset_ostring = YOLODataset(
        config.OSTRING_DS.DATASET_TRAIN_CSV,
        config.OSTRING_DS.IMAGE_DIR,
        config.OSTRING_DS.BBOX_LABEL_DIR,
        config.OSTRING_DS.DEPTH_NN_MAP_LABEL,
        S=config.S,
        anchors=config.ANCHORS,
        transform=config.train_transforms_img if config.USE_DATA_AUGMENTATION_TRAIN else None,
        bboxLabelColumnIndex=config.OSTRING_DS.BOX_LABEL_COL,
    )

    test_dataset_ostring = YOLODataset(
        config.OSTRING_DS.DATASET_TEST_CSV,
        config.OSTRING_DS.IMAGE_DIR,
        config.OSTRING_DS.BBOX_LABEL_DIR,
        config.OSTRING_DS.DEPTH_NN_MAP_LABEL,
        S=config.S,
        anchors=config.ANCHORS,
        transform=None,
        bboxLabelColumnIndex=config.OSTRING_DS.BOX_LABEL_COL,
    )


    # MVD Datasets
    # =========================
    train_dataset_mvd = YOLODataset(
        config.MAPILLARY_DS.DATASET_TRAIN_CSV,
        config.MAPILLARY_DS.IMAGE_DIR,
        config.MAPILLARY_DS.BBOX_LABEL_DIR,
        config.MAPILLARY_DS.DEPTH_NN_MAP_LABEL,
        S=config.S,
        anchors=config.ANCHORS,
        transform=config.train_transforms_img if config.USE_DATA_AUGMENTATION_TRAIN else None,
        bboxLabelColumnIndex=config.MAPILLARY_DS.BOX_LABEL_COL,
    )

    test_dataset_mvd = YOLODataset(
        config.MAPILLARY_DS.DATASET_TEST_CSV,
        config.MAPILLARY_DS.IMAGE_DIR,
        config.MAPILLARY_DS.BBOX_LABEL_DIR,
        config.MAPILLARY_DS.DEPTH_NN_MAP_LABEL,
        S=config.S,
        anchors=config.ANCHORS,
        transform=None,
        bboxLabelColumnIndex=config.MAPILLARY_DS.BOX_LABEL_COL,
    )
    # Choose right Dataloaders with fitting batch size
    if not config.JUST_VIZUALIZE_NO_TRAINING:
        train_loader_ostring = DataLoader(dataset=train_dataset_ostring, batch_size=config.BATCH_SIZE_OST, shuffle=config.SHUFFLE_DATA_LOADER, num_workers=config.NUM_WORKERS//2,drop_last=True)
        train_loader_mvd = DataLoader(dataset=train_dataset_mvd, batch_size=config.BATCH_SIZE_MVD, shuffle=config.SHUFFLE_DATA_LOADER, num_workers=config.NUM_WORKERS,drop_last=True)
    else: # for visualization datasets arent mixed in batch, so we can max out batch size
        train_loader_ostring = DataLoader(dataset=train_dataset_ostring, batch_size=config.BATCH_SIZE, shuffle=config.SHUFFLE_DATA_LOADER, num_workers=config.NUM_WORKERS//2,drop_last=True)
        train_loader_mvd = DataLoader(dataset=train_dataset_mvd, batch_size=config.BATCH_SIZE, shuffle=config.SHUFFLE_DATA_LOADER, num_workers=config.NUM_WORKERS,drop_last=True)

    test_loader_ostring = DataLoader(dataset=test_dataset_ostring, batch_size=config.BATCH_SIZE, shuffle=config.SHUFFLE_DATA_LOADER, num_workers=config.NUM_WORKERS//2,drop_last=True)
    test_loader_mvd = DataLoader(dataset=test_dataset_mvd, batch_size=config.BATCH_SIZE, shuffle=config.SHUFFLE_DATA_LOADER, num_workers=config.NUM_WORKERS//2,drop_last=True)
    
    
    if config.LOAD_MODEL:
        if config.TRANSFER_MODEL: # deletes last layer
            load_checkpoint_transfer(
                config.LOAD_CHECKPOINT_FILE, model, optimizer, config.LEARNING_RATE
            )
        else:
            load_checkpoint(
                config.LOAD_CHECKPOINT_FILE, model, optimizer, config.LEARNING_RATE
            )


    # caching like this is a problem when i have data augmentation transformations or large ds
    # cache_ds(train_dataset_ostring)
    # cache_ds(test_dataset_ostring)
    # # cache_ds(train_dataset_mvd,num_workers=config.NUM_WORKERS//2)
    # cache_ds(test_dataset_mvd,num_workers=config.NUM_WORKERS//2)


    print('Log results under: '+config.TRAINING_LOG_DIR)


    # =============================================
    #           Main Loop over Epochs
    # =============================================

    for epoch in range(config.NUM_EPOCHS):
        if config.PLOT_COUPLE_EXAMPLES and (epoch >= 1 or config.PLOT_BEFORE_TRAINING):
            print('plotting examples:')

            plot_couple_examples(model, test_loader_ostring, config.INF_CONF_THRESHOLD, config.NMS_IOU_THRESH, scaled_anchors,depth_anchors,noOfExamples=config.NO_OF_EXAMPLES_TEST, epochNo=epoch,datasetName='OstringTest')
            plot_couple_examples(model, test_loader_mvd, config.INF_CONF_THRESHOLD, config.NMS_IOU_THRESH, scaled_anchors,depth_anchors,noOfExamples=config.NO_OF_EXAMPLES_TEST, epochNo=epoch,datasetName='MVDTest')
            plot_couple_examples(model, train_loader_ostring, config.INF_CONF_THRESHOLD, config.NMS_IOU_THRESH, scaled_anchors,depth_anchors,noOfExamples=config.NO_OF_EXAMPLES_TRAIN, epochNo=epoch,datasetName='OstringTrain')
            plot_couple_examples(model, train_loader_mvd, config.INF_CONF_THRESHOLD, config.NMS_IOU_THRESH, scaled_anchors,depth_anchors,noOfExamples=config.NO_OF_EXAMPLES_TRAIN, epochNo=epoch,datasetName='MVDTrain')
            


        print(f"========== Epoch No: {epoch} ==============")

        def evaluateMAP(testloader,bestMAP=bestMAP,lastBestModelName=lastBestModelName,datasetName=''):
            clsAccuracy,noobjAccuracy,objAccuracy = check_class_accuracy(model, testloader, threshold=config.TEST_CONF_THRESHOLD)
            # TODO fix nms performance!
            pred_boxes, true_boxes,train_idx_filename_dict = get_evaluation_bboxes(
                testloader,
                model,
                iou_threshold=config.NMS_IOU_THRESH,
                anchors=config.ANCHORS,
                depth_anchors=depth_anchors,
                threshold=config.TEST_CONF_THRESHOLD,
                device=config.DEVICE
            )
            mapval, mapval_per_class = mean_average_precision(
                pred_boxes,
                true_boxes,
                iou_threshold=config.MAP_IOU_THRESH,
                box_format="midpoint",
                num_classes=config.NUM_CLASSES,
                reduction=False,
                filename_dict=train_idx_filename_dict
            )
            # TODO evaluate depth metrics for pred_boxes vs true boxes, includes the Non Max Suppression Computation, compared to only Test Loss
            # depth_var, depth_mean = evaluateDepth(
            #     pred_boxes,
            #     true_boxes,
            #     iou_threshold = config.MAP_IOU_THRESH,
            #     num_classes = config.NUM_CLASSES
            # )

            if mapval > bestMAP:
                bestMAP = mapval
                if config.SAVE_MODEL:
                    filename = config.SAVE_CHECKPOINT_FILE +datasetName+f'bestMAP_{mapval}'
                    save_checkpoint(model,optimizer,filename)
                    # TODO somehow this doesnt work. saves new modelweights each time
                    if os.path.exists(lastBestModelName):
                        os.remove(lastBestModelName)
                    lastBestModelName=filename
                
            print(f"=============== {datasetName} MAPs: Best/now: {bestMAP}/{mapval}  ==============")
            for idx, mapc in enumerate(mapval_per_class):
                print(f"Class: {config.MVD_CLASSES[idx]}: {mapc}")
                tb.add_scalar(datasetName+f'mAP_Val/{idx}',mapc,epoch)
            f = open("logTraining.txt", "a")
            tb.add_scalar(datasetName+f'mAP_Val/sum',sum(mapval_per_class)/len(mapval_per_class),epoch)
            tb.add_scalar(datasetName+f'Accuracy_Val/class',clsAccuracy,epoch)
            tb.add_scalar(datasetName+f'Accuracy_Val/noObj',noobjAccuracy,epoch)
            tb.add_scalar(datasetName+f'Accuracy_Val/Obj',objAccuracy,epoch)
            f.write(f"MAP: {mapval}")

        if (epoch > 0 and epoch % config.EVAL_MAP_FREQ == 0) : 
            evaluateMAP(test_loader_mvd,datasetName='MVD')
            evaluateMAP(test_loader_ostring,datasetName='OST')
        elif(epoch == 0 and config.TEST_MAP_AT_START):
#for depth only eval
###################################
            # evaluateMAP(test_loader_mvd,datasetName='MVD')
###################################            
            
            
            
            evaluateMAP(test_loader_ostring,datasetName='OST')
            # evaluateMAP(train_loader_ostring,datasetName='OST_Train')
        # Entering Training Function
        if not config.JUST_VIZUALIZE_NO_TRAINING:
            if config.EVALUATE_TEST_LOSS:
                print("On Test loader:")
                valLoss_OST=eval_Val_Loss(test_loader_ostring,model,loss_fn,epoch,'ostring')
                valLoss_MVD=eval_Val_Loss(test_loader_mvd,model,loss_fn,epoch,'mvd')
                if valLoss_MVD < minValidationLoss_MVD:
                        minValidationLoss_MVD=valLoss_MVD
                        if config.SAVE_MODEL:
                            print('saving Model with Best Detection Performance on MVD')
                            filename = config.SAVE_CHECKPOINT_FILE + 'bestMVDLoss'
                            save_checkpoint(model,optimizer,filename)
                if valLoss_OST < minValidationLoss_OST:
                        minValidationLoss_OST=valLoss_OST
                        if config.SAVE_MODEL:
                            print('saving Model with Best Detection Performance on OST')
                            filename = config.SAVE_CHECKPOINT_FILE + 'bestOSTLoss'
                            save_checkpoint(model,optimizer,filename)

            print('logging detection head weights and biases')
            for key,val in model.state_dict().items():
                if ('layers.15.pred.' in key and ('conv.weight' in key or 'conv.bias' in key)):
                    tb.add_histogram(key,val,epoch)
                if ('layers.22.pred.' in key and ('conv.weight' in key or 'conv.bias' in key)):
                    tb.add_histogram(key,val,epoch)
                if ('layers.29.pred.' in key and ('conv.weight' in key or 'conv.bias' in key)):
                    tb.add_histogram(key,val,epoch)


            # Training function
            # ==================
            print("Training Loop:")
            train_fn(train_loader_mvd,train_loader_ostring, model, optimizer, loss_fn, scaler,epoch=epoch)
            if config.SAVE_MODEL:
                save_checkpoint(model, optimizer, filename=config.SAVE_CHECKPOINT_FILE)

                
# not used with large datasets, or when augmenting data
def cache_ds(dataset,num_workers=config.NUM_WORKERS):
    cache_loader = DataLoader(dataset=dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=num_workers)
    iter=tqdm(cache_loader) # len(iter) is no of batches
    
    for batch_indx, (x,y,depthtarget) in enumerate(iter):
        for i in range(0,x.shape[0]):
            # only one epoch: i=bi*BS + i
            idx=batch_indx*config.BATCH_SIZE+i
            # TODO List assignment index out of range
            
            dataset.imgs[idx] = x[i]
            dataset.targets[idx] = (y[0][i],y[1][i],y[2][i])
            dataset.depth_targets[idx] = depthtarget[i]
    dataset.isCached=True

if __name__ == "__main__":
    main()
