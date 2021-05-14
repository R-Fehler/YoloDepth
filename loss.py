"""
Implementation of Yolo Loss Function similar to the one in Yolov3 paper,
the difference from what I can tell is I use CrossEntropy for the classes
instead of BinaryCrossEntropy.
"""
from Timer import Timer
from torch._C import dtype
from model import YOLOv3
from dataset import DepthDataset, YOLODataset
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import config

from utils import cells_to_bboxes, intersection_over_union


class YoloLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.entropy = nn.CrossEntropyLoss()
        self.sigmoid = nn.Sigmoid()
        self.SILogLoss = SILogLoss()

        # Constants signifying how much to pay for each respective part of the loss
        self.lambda_class = 1
        self.lambda_noobj = 10
        self.lambda_obj = 1
        self.lambda_box = 10
        self.lambda_depth = 1

    def forward(self, predictions, target , anchors, depth_target,S, mask=None, interpolate=True):
        anchors_const=anchors
        # Check where obj and noobj (we ignore if target == -1)
        obj = target[..., 0] == 1  # in paper this is Iobj_i
        noobj = target[..., 0] == 0  # in paper this is Inoobj_i

        # ======================= #
        #   FOR NO OBJECT LOSS    #
        # ======================= #

        no_object_loss = self.bce(
            (predictions[..., 0:1][noobj]), (target[..., 0:1][noobj]),
        )

        # ==================== #
        #   FOR OBJECT LOSS    #
        # ==================== #

        anchors = anchors.reshape(1, 3, 1, 1, 2)
        box_preds = torch.cat([self.sigmoid(predictions[..., 1:3]), torch.exp(predictions[..., 3:5]) * anchors], dim=-1)
        ious = intersection_over_union(box_preds[obj], target[..., 1:5][obj]).detach()
        object_loss = self.mse(self.sigmoid(predictions[..., 0:1][obj]), ious * target[..., 0:1][obj])

        # ======================== #
        #   FOR BOX COORDINATES    #
        # ======================== #

        predictions[..., 1:3] = self.sigmoid(predictions[..., 1:3])  # x,y coordinates
        target[..., 3:5] = torch.log(
            (1e-16 + target[..., 3:5] / anchors)
        )  # width, height coordinates
        box_loss = self.mse(predictions[..., 1:5][obj], target[..., 1:5][obj])

        # ================== #
        #   FOR CLASS LOSS   #
        # ================== #

        class_loss = self.entropy(
            (predictions[..., 6:][obj]), (target[..., 5][obj].long()),
        )


        # ================== #
        #   FOR DEPTH LOSS   #
        # ================== #
        #     assert out[0].shape == (batch,B Bbox per cell, IMAGE_SIZE//32, IMAGE_SIZE//32, 5 +1+num_classes)
        depth_loss=torch.tensor(0.0,device=config.DEVICE)

        depth_pred=torch.zeros(depth_target.shape).to(config.DEVICE) # needs a layer for each bbox?
        depth_pred_small_patch=torch.zeros(depth_target.shape).to(config.DEVICE) # needs a layer for each bbox?
        # TODO check with debugger the shape of coords_depth_input
        coords_depth_input=predictions[...,1:3] # problem: relative to cell origin...
        noOfPartialLosses=0
        for k in range(predictions.shape[1]): # iterate over no of bboxes ( eg. 3 per grid cell pro scale)
            bboxes = cells_to_bboxes(predictions,anchors_const,S,is_preds=True,reshape=False)
            for i in range(predictions.shape[2]): # iterate over with eg 13
                for j in range(predictions.shape[3]): # iterate over height eg 13 @Scale 13

                    for b in range(predictions.shape[0]): # iterate over batch 
                        coords_x = bboxes[b,k,i,j,1]*config.IMAGE_SIZE
                        coords_y = bboxes[b,k,i,j,2]*config.IMAGE_SIZE
                        w = bboxes[b,k,i,j,3]*config.IMAGE_SIZE
                        h = bboxes[b,k,i,j,4]*config.IMAGE_SIZE
                        
                        if(w > 0 and h > 0 and not torch.isinf(h) and not torch.isinf(w) and not torch.isnan(w) and not torch.isnan(h)):
                            depth_pred[ b, int((coords_x-(w/2)).item()) : int((coords_x+(w/2)).item()), int((coords_y-(h/2)).item()) : int((coords_y+(h/2)).item())] = torch.maximum(torch.tensor(1.0,device=config.DEVICE),predictions[b,k,i,j,5])
                            depth_pred_small_patch[ b, int((coords_x-(w/4)).item()) : int((coords_x+(w/4)).item()), int((coords_y-(h/4)).item()) : int((coords_y+(h/4)).item())] = torch.maximum(torch.tensor(1.0,device=config.DEVICE),predictions[b,k,i,j,5]) # use the whole bbox as depth prediction area
                            mask_full_bbox = torch.logical_and(depth_target >= 1.0 , depth_pred >= 1.0)
                            mask_small = torch.logical_and(depth_target >= 1.0 , depth_pred_small_patch >= 1.0)
                            if (mask_full_bbox.max()==True):
                                partial_loss = self.SILogLoss(depth_pred, depth_target, mask=mask_full_bbox, interpolate=False)
                                if(not torch.isnan(partial_loss)):
                                    depth_loss += partial_loss
                                    noOfPartialLosses=noOfPartialLosses + 1
                            if  (mask_small.max()==True):
                                partial_loss = self.SILogLoss(depth_pred_small_patch, depth_target, mask=mask_small, interpolate=False)
                                if(not torch.isnan(partial_loss)):
                                    depth_loss += partial_loss
                                    noOfPartialLosses=noOfPartialLosses + 1

        depth_loss=depth_loss/noOfPartialLosses

        
#         for i in predictions.shape[2]: # iterate over with ? eg 13
#             for j in predictions.shape[3]: # iterate over height ? or witdh? eg 13 @Scale 13
#                 for k in predictions.shape[1]: # iterate over no of bboxes ( eg. 3 per grid cell pro scale)
#                     coords = predictions[:,k,i,j,1:3]
#                     w = predictions[:,k,i,j,3]
#                     h = predictions[:,k,i,j,4]
#                     # anchors are scaled to image pixel space
#                     #    scaled_anchors = (
#                     #         torch.tensor(config.ANCHORS)
#                     #         * torch.tensor(config.S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
#                     #     ).to(config.DEVICE)
#                     # TODO do i need anchor distances. eg take dataset, do kmeans for bbox w,h, and z. they are most likely correlated. 
#                     # i could also check the dataset for statistical analysis of w, h, z and class label. 
#                     w = anchors[0,k,0]*torch.exp(w)
#                     h = anchors[0,k,1]*torch.exp(h)
#                     # min_side=min(w,h) für die Gewichtung der Lidar Punkte
#                     # TODO use x,y (need to be tensors (each grid celll etc. )) from function parameters or this 
#                     coords_x = config.IMAGE_SIZE / predictions.shape[2] * (i + coords[i]) # coords in total image pixel 
#                     coords_y = config.IMAGE_SIZE / predictions.shape[3] * (j + coords[j])
#                     depth_input[ :, k, coords_x-(w/2) : coords_x+(w/2), coords_y-(h/2) : coords_y+(h/2)] = predictions[:,k,i,j,5] # use the whole bbox as depth prediction area
#                     # TODO convert x,y for bbox w h, in circle coords? or 2 for loops calc euklid length with threshhold == radius
#                     depth_input_small_patch[ :, k, coords_x-(w/4) : coords_x+(w/4), coords_y-(h/4) : coords_y+(h/4)] = predictions[:,k,i,j,5] # use the whole bbox as depth prediction area
#                 ######################
#                 #                   #
#                 #    ##########     #     
#                 #    #        #     #
#                 #    #  z=10m #     #   416        vs lidar projection map (416x416)
#                 #    #        #     #
#                 #    ##########     #
# #               #####################
#                 #       
#                 #       416

#                 # mask is intersection of depthinput>0 AND depth_target > 0
#                 # mask is intersection of depthinput_smallpatch>0 AND depth_target > 0

#                     #TODO depth input würde projiziert werden upscaled. dh to total image size. 
#                     #TODO we want to calculate the loss only for the part of the image that contains the bbox. and calc. loss only for valid lidar points

#                 #        missing is the weighting of the projection values compared to the cp of the bbox.
#         # do a second one with a small patch around cp with higher weighing coefficient?
#         depth_loss = self.SILogLoss(depth_input, depth_target, mask=mask_full_bbox, interpolate=interpolate)
#         depth_loss += self.SILogLoss(depth_input_small_patch, depth_target, mask=mask_small, interpolate=interpolate)
        
        print("__________________________________")
        print(self.lambda_box * box_loss)
        print(self.lambda_obj * object_loss)
        print(self.lambda_noobj * no_object_loss)
        print(self.lambda_class * class_loss)
        print(self.lambda_depth * depth_loss)
        print("\n")

        return (
            self.lambda_box * box_loss
            + self.lambda_obj * object_loss
            + self.lambda_noobj * no_object_loss
            + self.lambda_class * class_loss
            + self.lambda_depth * depth_loss
        )


class DepthLoss(nn.Module):
    def __init__(self):
        super().__init__()


    def forward(self,input, target):
        assert input.shape==target.shape ,"loss shapes not fitting"
        input = input.clamp(1.,255.)
        target = target.clamp(1.,255.)
        g = torch.log(input) - torch.log(target)
        Dg = torch.var(g) + 0.15 * torch.pow(torch.mean(g),2)
        return 10 * torch.sqrt(Dg)




class SILogLoss(nn.Module):  # Main loss function used in AdaBins paper
    def __init__(self):
        super(SILogLoss, self).__init__()
        self.name = 'SILog'

    def forward(self, input, target, mask=None, interpolate=True):
        if interpolate:
            input = nn.functional.interpolate(input, target.shape[-2:], mode='bilinear', align_corners=True)
        input = input.squeeze(dim=1)
        if mask is not None:
            input = input[mask]
            target = target[mask]
        g = torch.log(input) - torch.log(target)
        # n, c, h, w = g.shape
        # norm = 1/(h*w)
        # Dg = norm * torch.sum(g**2) - (0.85/(norm**2)) * (torch.sum(g))**2

        Dg = torch.var(g) + 0.15 * torch.pow(torch.mean(g), 2)
        return 10 * torch.sqrt(Dg)

if __name__=='__main__':
    scaled_anchors = (
        torch.tensor(config.ANCHORS)
        * torch.tensor(config.S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    ).to(config.DEVICE)
    dataset = YOLODataset(
        "OstringDepthDataset/OstringDataSet.csv",
        "OstringDepthDataset/imgs/",
        "OstringDepthDataset/bbox_labels/",
        "OstringDepthDataset/depth_labels/",
        S=[13, 26, 52],
        anchors=config.ANCHORS,
        transform=None,
    )
    loader = DataLoader(dataset=dataset,batch_size=8)
    model = YOLOv3(num_classes=config.NUM_CLASSES).to(config.DEVICE)
    for idx,(x,y,depth_target) in enumerate(loader):
        depth_target=depth_target.to(config.DEVICE,dtype=torch.float)
        x = x.to(config.DEVICE,dtype=torch.float)
        y0, y1, y2 = (
            y[0].to(config.DEVICE),
            y[1].to(config.DEVICE),
            y[2].to(config.DEVICE),
        )
        with torch.no_grad():
            out=model(x)
        loss_fn=YoloLoss()
        #    def forward(self, predictions, target , anchors, depth_target,S, mask=None, interpolate=True):
        # out[2] = torch.clamp(out[2],1.0,255.0)
        with Timer('loss calc'):
            l_dense = loss_fn(out[2],y2,scaled_anchors[2], depth_target,config.S[2], interpolate=True)
        print(l_dense)
