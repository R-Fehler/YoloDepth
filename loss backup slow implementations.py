"""
Implementation of Yolo Loss Function similar to the one in Yolov3 paper,
the difference from what I can tell is I use CrossEntropy for the classes
instead of BinaryCrossEntropy.
"""
from Timer import Timer
from torch._C import device, dtype
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

        # Constants signifying how much to pay for each respective part of the loss
        self.lambda_class = 1
        self.lambda_noobj = 10
        self.lambda_obj = 1
        self.lambda_box = 10
        self.lambda_depth = 1

    def forward(self, predictions, target, depth_target, anchors,scale_S, mask=None, interpolate=True):
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
        # time 0.013 to 0.25

        #     assert out[0].shape == (batch,B Bbox per cell, IMAGE_SIZE//32, IMAGE_SIZE//32, 5 +1+num_classes)
        depth_loss=torch.tensor(0.0,device=config.DEVICE)

        # depth_pred=torch.zeros(depth_target.shape).to(config.DEVICE) # needs a layer for each bbox?
        # depth_pred_small_patch=torch.zeros(depth_target.shape).to(config.DEVICE) # needs a layer for each bbox?
        # TODO check with debugger the shape of coords_depth_input
        noOfPartialLosses=0

        # --------------------------------- #
        # Vectorized img per bbox SILOGLoss |
        # --------------------------------- #


        # for k in range(predictions.shape[1]): # iterate over no of bboxes ( eg. 3 per grid cell pro scale)
        #     bboxes = cells_to_bboxes(predictions,anchors_const,S,is_preds=True,reshape=False)
        #     for i in range(predictions.shape[2]): # iterate over with eg 13
        #         for j in range(predictions.shape[3]): # iterate over height eg 13 @Scale 13

        #             for b in range(predictions.shape[0]): # iterate over batch 
        #                 coords_x = bboxes[b,k,i,j,1]*config.IMAGE_SIZE
        #                 coords_y = bboxes[b,k,i,j,2]*config.IMAGE_SIZE
        #                 w = bboxes[b,k,i,j,3]*config.IMAGE_SIZE
        #                 h = bboxes[b,k,i,j,4]*config.IMAGE_SIZE
                        
        #                 if(w > 0 and h > 0 and not torch.isinf(h) and not torch.isinf(w) and not torch.isnan(w) and not torch.isnan(h)):
        #                     depth_pred[ b, int((coords_x-(w/2)).item()) : int((coords_x+(w/2)).item()), int((coords_y-(h/2)).item()) : int((coords_y+(h/2)).item())] = torch.maximum(torch.tensor(1.0,device=config.DEVICE),predictions[b,k,i,j,5])
        #                     depth_pred_small_patch[ b, int((coords_x-(w/4)).item()) : int((coords_x+(w/4)).item()), int((coords_y-(h/4)).item()) : int((coords_y+(h/4)).item())] = torch.maximum(torch.tensor(1.0,device=config.DEVICE),predictions[b,k,i,j,5]) # use the whole bbox as depth prediction area
        #                     mask_full_bbox = torch.logical_and(depth_target >= 1.0 , depth_pred >= 1.0)
        #                     mask_small = torch.logical_and(depth_target >= 1.0 , depth_pred_small_patch >= 1.0)
        #                     if (mask_full_bbox.max()==True):
        #                         partial_loss = self.SILogLoss(depth_pred, depth_target, mask=mask_full_bbox, interpolate=False)
        #                         if(not torch.isnan(partial_loss)):
        #                             depth_loss += partial_loss
        #                             noOfPartialLosses=noOfPartialLosses + 1
        #                     if  (mask_small.max()==True):
        #                         partial_loss = self.SILogLoss(depth_pred_small_patch, depth_target, mask=mask_small, interpolate=False)
        #                         if(not torch.isnan(partial_loss)):
        #                             depth_loss += partial_loss
        #                             noOfPartialLosses=noOfPartialLosses + 1

        # depth_loss=depth_loss/noOfPartialLosses
        
        # --------------------------------------------------------------- #
        # Working Loss, sampling the nearest neighbour griddata depth map |
        # ----------------------------------------------------------------#

        bboxes = cells_to_bboxes(predictions,anchors_const,scale_S,is_preds=True,reshape=False).to(config.DEVICE)
        pred_centers = bboxes[:,:,:,:,1:3] * config.IMAGE_SIZE
        pred_centers = pred_centers.clamp(0,config.IMAGE_SIZE-1)
        pred_depths = bboxes[:,:,:,:,6]
        # clamp durch differenzierbare funktion auf den bereich 1-255 mappen
        pred_depths = pred_depths.clamp(0.1,255)
        depth_target = depth_target.clamp(0.1,255)
        # depth target is bs x 416 x 416
        log_depth_target = torch.log(depth_target)
        # depth_pred is bs x 3 x 13 x 13 
        log_depth_pred = torch.log(pred_depths)
        # shape is bs x 3 x 13 x 13 x 2
        indices = pred_centers.type(torch.int64).detach() # the x,y indices for selecting the depth value in the dense depth map
        
        target_depths_selected_with_pred_centers = torch.zeros_like(
            log_depth_pred,device=config.DEVICE)
            # with this implementation the calc time is 0.04 instead of 0.014 which is acceptable
        for b in range(pred_centers.shape[0]):
            # TODO assign the target depth selected only the depth pixel values at the coordinates corresponding to the centers of the bboxes
            # cant access because shape is not 1D for this type of acces: []
            #==========================================
            #  target_depths_selected_with_pred_centers.shape
            # torch.Size([1, 3, 52, 52])
            # log_depth_target.shape
            # torch.Size([1, 416, 416])
            # indices.shape
            # torch.Size([1, 3, 52, 52, 2])
            # ========================================
            # Generates CUDA Error

            target_depths_selected_with_pred_centers[b,:,:,:] = log_depth_target[b][indices[b,:,:,:,0],indices[b,:,:,:,1]]
            
        g = target_depths_selected_with_pred_centers - log_depth_pred
        g = g.view((g.shape[0],-1))
        Dg = torch.var(g, dim=1) + 0.15 * torch.pow(torch.mean(g, dim=1), 2)
        depth_loss = 10 * torch.sqrt(Dg)
        depth_loss = depth_loss.mean() #SCALAR
        
        # print("__________________________________")
        # print(self.lambda_box * box_loss)
        # print(self.lambda_obj * object_loss)
        # print(self.lambda_noobj * no_object_loss)
        # print(self.lambda_class * class_loss)
        # print(self.lambda_depth * depth_loss)
        # print("\n")

        return (
            self.lambda_box * box_loss
            , self.lambda_obj * object_loss
            , self.lambda_noobj * no_object_loss
            , self.lambda_class * class_loss
            , self.lambda_depth * depth_loss
        )


        # -------------------------------------------- #
        # Loss first attempt, img per bbox SILOGLoss   #
        # -------------------------------------------- #     
        ts=depth_target.shape
        ps=predictions.shape
                                        # batch,416,416,k(bbox per cell),i,j(cell coord)
        depth_pred_tensor = torch.zeros(ts[0],ts[1],ts[2],ps[1],ps[2],ps[3],device=config.DEVICE) # needs 5GB of memory for 1 img!!
        print("n of elements: \n")
        print(depth_pred_tensor.nelement())
        print("size of all elements in bytes:\n")
        print(depth_pred_tensor.element_size()*depth_pred_tensor.nelement())

        depth_pred_small_tensor = torch.zeros(ts[0],ts[1],ts[2],ps[1],ps[2],ps[3],device=config.DEVICE)
                                        # batch,416,416,k(bbox per cell),i,j(cell coord),416,416(bolean mask for selecting only valid pixels in SILogLoss)
        depth_target_tensor = torch.zeros(ts[0],ts[1],ts[2],ps[1],ps[2],ps[3],device=config.DEVICE)
        with Timer('loop, building the huge tensor'): # takes about 10 sec per img
            for k in range(predictions.shape[1]): # iterate over no of bboxes ( eg. 3 per grid cell pro scale)
                for i in range(predictions.shape[2]): # iterate over with eg 13
                    for j in range(predictions.shape[3]): # iterate over height eg 13 @Scale 13
                        for b in range(predictions.shape[0]): # iterate over batch 

                        
                            coords_x = bboxes[b,k,i,j,1]*config.IMAGE_SIZE
                            coords_y = bboxes[b,k,i,j,2]*config.IMAGE_SIZE
                            w = bboxes[b,k,i,j,3]*config.IMAGE_SIZE
                            h = bboxes[b,k,i,j,4]*config.IMAGE_SIZE
                            depth_target_tensor[ :, :, :, k, i, j] = depth_target

                            if(w > 0 and h > 0 and not torch.isinf(h) and not torch.isinf(w) and not torch.isnan(w) and not torch.isnan(h)):
                                depth_pred_tensor[ b, int((coords_x-(w/2)).item()) : int((coords_x+(w/2)).item()), int((coords_y-(h/2)).item()) : int((coords_y+(h/2)).item()), k, i, j] = torch.maximum(torch.tensor(1.0,device=config.DEVICE),predictions[b,k,i,j,5])
                                depth_pred_small_tensor[ b, int((coords_x-(w/4)).item()) : int((coords_x+(w/4)).item()), int((coords_y-(h/4)).item()) : int((coords_y+(h/4)).item()),k,i,j] = torch.maximum(torch.tensor(1.0,device=config.DEVICE),predictions[b,k,i,j,5]) # use the whole bbox as depth prediction area
        # mask has shape of pred and target, is bool tensor
        mask_full_bbox = torch.logical_and(depth_target_tensor >= 1.0 , depth_pred_tensor >= 1.0)
        mask_small = torch.logical_and(depth_target_tensor >= 1.0 , depth_pred_small_tensor >= 1.0)
        if (mask_full_bbox.max()==True):
            with Timer('calc loss given all tensors'): # takes about 6.5 -10 sec per imgs
                    # TODO  torch.log(target)
                partial_loss = self.SILogLoss(depth_pred_tensor, depth_target_tensor, mask=mask_full_bbox, interpolate=False)
                if(not torch.isnan(partial_loss)):
                    depth_loss += partial_loss
                    noOfPartialLosses=noOfPartialLosses + 1
        # if  (mask_small.max()==True):
        #     partial_loss = self.SILogLoss(depth_pred_small_patch, depth_target, mask=mask_small, interpolate=False)
        #     if(not torch.isnan(partial_loss)):
        #         depth_loss += partial_loss
        #         noOfPartialLosses=noOfPartialLosses + 1

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
    # docstring mit tensor shapes [in out etc]

        # input as float scalar 
        if interpolate:
            input = nn.functional.interpolate(input, target.shape[-2:], mode='bilinear', align_corners=True) # TODO check if this still works with [b,I,I,k,i,j] tensor shape
        input = input.squeeze(dim=1)
        if mask is not None:
            # input = input[mask]
            target = target[mask] # flatted
        # ab hier nur noch 1D Vector?!
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
    loader = DataLoader(dataset=dataset,batch_size=10)
    model = YOLOv3(num_classes=config.NUM_CLASSES).to(config.DEVICE)
    loss_fn=YoloLoss()

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
        #    def forward(self, predictions, target , anchors, depth_target,S, mask=None, interpolate=True):
        # out[2] = torch.clamp(out[2],1.0,255.0)
        with Timer('loss calc'):
            l_dense = loss_fn(out[2],y2,depth_target,scaled_anchors[2] ,config.S[2], interpolate=True)
        print(l_dense)
