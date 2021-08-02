import random

import torch
import torch.nn as nn
from torch._C import device, dtype
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard.writer import SummaryWriter

import config
from dataset import YOLODataset
from model import YOLOv3
from Timer import Timer
from utils import cells_to_bboxes, intersection_over_union


class YoloLoss(nn.Module):
    '''
        def forward(self, predictions, target, dense_depth_target, anchors, scale_S,depth_anchors, mask=None, interpolate=True):
    ''' 
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.entropy = nn.CrossEntropyLoss()
        self.sigmoid = nn.Sigmoid()

        # Constants signifying how much to pay for each respective part of the loss
        # self.lambda_class = 0
        self.lambda_class = 1
        self.lambda_noobj = 10
        self.lambda_obj =  1
        self.lambda_box = 10
        # self.lambda_box = 0
        self.lambda_depth_bbox = config.USE_OBJ_DEPTH_LOSS * 1
        self.lambda_depth_lidar = 0 # weighing factor for autodetecting objects, no pseudo labels in depth dataset
        self.lambda_noobj_depth = config.USE_NO_OBJ_DEPTH_LOSS * 0.5
        # self.lambda_class = 0
        # self.lambda_noobj = 0
        # self.lambda_obj   = 0
        # self.lambda_box   = 0
        # self.lambda_depth =  1

    def forward(self, predictions, target, dense_depth_target, anchors, scale_S,depth_anchors,tensorboard:SummaryWriter, mask=None, interpolate=True):
        '''
        target.shape [BS,numAnchorsAtScale,S,S,(obj,x,y,w,h,z,cls)]
        dense_depth_target.shape [BS,imgsize,imgsize]
        predictions.shape [BS,numAnchorsAtScale,S,S,(obj,x,y,w,h,z,cls*numclassses)]

        '''
        anchors_const = anchors
        # Check where obj and noobj (we ignore if target == -1)
        obj = target[..., 0] == 1  # in paper this is Iobj_i
        noobj = target[..., 0] == 0  # in paper this is Inoobj_i

        # idea:
        # noObjLabel = target[...,0] == -2 then skip detection loss
        # and if 2D Depth Map: select obj depth if objectness > threshold.
        noValidTargetDepth = target[...,5] <= 0
        validTargetDepth = target[...,5] > 0
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
        # after reshaping anchors look like this
        # tensor([[[[[1.3000, 1.3000]]],
        #  [[[1.3000, 3.9000]]],
        #  [[[9.1000, 2.6000]]]]],
        box_preds = torch.cat([self.sigmoid(predictions[..., 1:3]), torch.exp(predictions[..., 3:5]) * anchors], dim=-1)
        ious = intersection_over_union(box_preds[obj], target[..., 1:5][obj]).detach()
        object_loss = self.mse(self.sigmoid(predictions[..., 0:1][obj]), ious * target[..., 0:1][obj])

        # ======================== #
        #   FOR BOX COORDINATES    #
        # ======================== #

        predictions[..., 1:3] = self.sigmoid(
            predictions[..., 1:3])  # x,y coordinates
        target[..., 3:5] = torch.log(
            (1e-16 + target[..., 3:5] / anchors)
        )  # width, height coordinates
        box_loss = self.mse(predictions[..., 1:5][obj], target[..., 1:5][obj])

        # ================== #
        #   FOR CLASS LOSS   #
        # ================== #

        class_loss = self.entropy(
            (predictions[..., 6:][obj]), (target[..., 6][obj].long()),
        )

        # ======================================================================================================================== #
        #  THATS THE ONE!  FOR DEPTH LOSS Obj Specific from Map/ generated pseudo bbox labels with depth from lidar imgs   #
        # ======================================================================================================================== #
        depth_anchors_const = depth_anchors
        depth_anchors = depth_anchors.reshape(1,len(depth_anchors),1,1,1)
        # make 2 depth bins one conditional depth p(d|obj) other p(d)
        noObj_depth_loss = torch.tensor(0.0, device=config.DEVICE)
        obj_depth_loss_from_bbox = torch.tensor(0.0, device=config.DEVICE)
        obj_depth_loss_picked_from_lidar = torch.tensor(0.0, device=config.DEVICE)
        # TODO check that it dynamically evaluates when there is a depth label value in bbox txt file
        # Problem , batch part has target[...,5] == 0 everywhere
        obj_depth_target = target[...,5][torch.logical_and(obj,validTargetDepth)]
        use_obj_depth_loss = len(obj_depth_target) > 1
        if  use_obj_depth_loss:
            # returns converted_bboxes = torch.cat((best_class, scores, x, y, w_h,z), dim=-1).reshape(BATCH_SIZE, num_anchors, S , S, 7)
            obj_pred_depths = torch.exp(predictions[...,5:6]) * depth_anchors
            obj_pred_depths = torch.squeeze(obj_pred_depths)
            obj_pred_depths = obj_pred_depths[torch.logical_and(obj,validTargetDepth)]
            obj_pred_depths = obj_pred_depths.clamp(1, 255)
            obj_depth_target = obj_depth_target.clamp(1, 255)
            log_obj_depth_target = torch.log(obj_depth_target)
            log_obj_depth_pred = torch.log(obj_pred_depths)
            depth_difference = obj_pred_depths - obj_depth_target
            log_obj_depth_difference = log_obj_depth_pred - log_obj_depth_target
            assert log_obj_depth_difference.dim() == 1, 'depth loss g not dimension == 1'
            # g = g.view((g.shape[0], -1))
            log_obj_depth_variance = torch.var(log_obj_depth_difference)
            obj_depth_variance = torch.var(depth_difference)
            log_obj_depth_mean = torch.mean(log_obj_depth_difference)
            obj_depth_mean = torch.mean(depth_difference)
            Dg = log_obj_depth_variance +  0.15 * torch.pow(log_obj_depth_mean, 2)
            obj_depth_loss_from_bbox = torch.sqrt(Dg) # was 10 * 
            # depth_loss = self.mse(log_obj_depth_pred,log_depth_target) # was 10 * 
            # depth_loss = depth_loss.mean()  # SCALAR
        else: # only for distribution plotting 
            obj_pred_depths = torch.exp(predictions[...,5:6]) * depth_anchors
            obj_pred_depths = torch.squeeze(obj_pred_depths)
            obj_pred_depths = obj_pred_depths[torch.logical_and(obj,validTargetDepth)]
            obj_pred_depths = obj_pred_depths.clamp(1, 255)
            log_obj_depth_pred = torch.log(obj_pred_depths)

            # ======================================== #
            #   FOR DEPTH LOSS non obj Specific /Dense #
            # ======================================== #
        if dense_depth_target.dim() > 1 and config.USE_2D_DEPTH_MAP:
        # if depth_target.dim() > 1:
            
                    # ================== =======================================================================#
                    #   ACHTUNG !!! func cells_to_bboxes ändert reihenfolge von prediction tensor schichten!    #
                    # ================== =======================================================================#


            # TODO MVD no Obj Depth Loss NAN, bias 

            # returns converted_bboxes = torch.cat((best_class, scores, x, y, w_h,z), dim=-1).reshape(BATCH_SIZE, num_anchors, S , S, 7)
            bboxes = cells_to_bboxes(
                predictions, anchors_const, scale_S, depth_anchors=depth_anchors_const, is_preds=True, reshape=False).to(config.DEVICE)
            pred_centers = bboxes[:, :, :, :, 2:4] * config.IMAGE_SIZE
            pred_centers = pred_centers.clamp(0, config.IMAGE_SIZE-1)
            pred_depths = bboxes[:, :, :, :, 6]

            pred_depths = pred_depths.clamp(1, 255)
            # validDepthIndices = depth_target > 0.1 # doesnt work because depth target has img size dimensions not grid size dimension

            dense_depth_target = dense_depth_target.clamp(1, 255)
            # TODO find out why dense depth target has to be transposed everywhere
            dense_depth_target = torch.transpose(dense_depth_target,1,2)
            # depth target is bs x 416 x 416
            log_depth_target = torch.log(dense_depth_target)
            # depth_pred is bs x 3 x 13 x 13
            log_depth_pred = torch.log(pred_depths)
            # shape is bs x 3 x 13 x 13 x 2
            # the x,y indices for selecting the depth value in the dense depth map
            indices = pred_centers.type(torch.int64).detach()

            target_depths_selected_with_pred_centers = torch.zeros_like(
                log_depth_pred, device=config.DEVICE)

            for b in range(pred_centers.shape[0]):
                # ==========================================
                #  target_depths_selected_with_pred_centers.shape
                # torch.Size([1, 3, 52, 52])
                # log_depth_target.shape
                # torch.Size([1, 416, 416])
                # indices.shape
                # torch.Size([1, 3, 52, 52, 2])
                # ========================================
                target_depths_selected_with_pred_centers[b, :, :, :] = log_depth_target[b][indices[b, :, :, :, 0], indices[b, :, :, :, 1]]
            validDepthIndices = target_depths_selected_with_pred_centers > 0.01 # in log space: log(1) = 0
            validDepthAndNoObjIndices = torch.logical_and(noobj, validDepthIndices)
            
            log_depth_difference = target_depths_selected_with_pred_centers[validDepthAndNoObjIndices] - log_depth_pred[validDepthAndNoObjIndices]
            if len(log_depth_difference)!=0:
                assert log_depth_difference.dim() == 1, 'depth loss g not dimension == 1'

                Dg = torch.var(log_depth_difference) + 0.15 * torch.pow(torch.mean(log_depth_difference), 2)
                # g = target_depths_selected_with_pred_centers - log_depth_pred
                # g = g.view((g.shape[0], -1))
                # Dg = torch.var(g, dim=1) + 0.15 * torch.pow(torch.mean(g, dim=1), 2)
                noObj_depth_loss = torch.sqrt(Dg) # was 10 * 
                noObj_depth_loss = noObj_depth_loss.mean()  # SCALAR
            

            # =============================================== #
            #   FOR DEPTH LOSS obj Specific from 2D Depth Map #
            # =============================================== #
        if dense_depth_target.dim() > 1  and config.NO_PSEUDO_LABELS_IN_DEPTH_DS:
            x=bboxes[:, :, :, :, 2:3]
            y=bboxes[:, :, :, :, 3:4]
            width=bboxes[:, :, :, :, 4:5]
            height=bboxes[:, :, :, :, 5:6]

            # PROBLEM: How to vectorize the roi depth clustering
            x1,x2 = x - width/2, x + width/2
            y1,y2 = y - height/2 ,y + height/2
            x1.clamp(0,0.99999)
            x2.clamp(0,0.99999)
            y1.clamp(0,0.99999)
            y2.clamp(0,0.99999)
            x1 =int(x1*config.IMAGE_SIZE)
            x2 =int(x2*config.IMAGE_SIZE)
            y1 =int(y1*config.IMAGE_SIZE)
            y2 =int(y2*config.IMAGE_SIZE)
            # TODO Why is the Depth Coordinate System transposed compared to the labels?
            # TODO How to select the right view/ slice in vectorized form?
            roi=dense_depth_target[y1:y2,x1:x2]
            roi=roi[roi>1]
            if len(roi)>0:
                z = torch.quantile(roi,config.QUANTILE_DEPTH_CLUSTER) # when there is no valid depth value aka 0 in the dense depth target location it will be ignored in loss eval
                log_z = torch.log(z)
            else:
                z = 0
            unsupervised_obj_indices = self.sigmoid(predictions[..., 0:1]) > config.UNSUPERVISED_OBJ_THRESHOLD
            validDepthAndObjIndices = torch.logical_and(unsupervised_obj_indices,validDepthIndices) # when there is a obj label in the dataset
            
            log_depth_difference = log_z[validDepthAndObjIndices] - log_depth_pred[validDepthAndObjIndices]
            if len(log_depth_difference)!=0:
                assert log_depth_difference.dim() == 1, 'depth loss g not dimension == 1'
                Dg = torch.var(log_depth_difference) + 0.15 * torch.pow(torch.mean(log_depth_difference), 2)
                obj_depth_loss_picked_from_lidar = torch.sqrt(Dg)




        ################## BINNED DEPTH LOSS ##############################

        #     depth_loss = self.entropy(
        #     (predictions[..., 6:][obj]), (target[..., 6][obj].long()),
        # )



        # print("__________________________________")
        # print(self.lambda_box * box_loss)
        # print(self.lambda_obj * object_loss)
        # print(self.lambda_noobj * no_object_loss)
        # print(self.lambda_class * class_loss)
        # print(self.lambda_depth * depth_loss)
        # print("\n")
        if use_obj_depth_loss:
            return (
                self.lambda_box * box_loss,
                self.lambda_obj * object_loss,
                self.lambda_noobj * no_object_loss,
                self.lambda_class * class_loss,
                self.lambda_depth_bbox * obj_depth_loss_from_bbox,
                self.lambda_noobj_depth * noObj_depth_loss,
                self.lambda_depth_lidar * obj_depth_loss_picked_from_lidar,
                {'objDepthTarget':obj_depth_target,
                'objDepthPredicted':obj_pred_depths,
                'log_obj_depth_target' :log_obj_depth_target,
                'log_obj_depth_pred': log_obj_depth_pred,
                'objDepthDifference':depth_difference,
                'logObjDepthDifference':log_obj_depth_difference,
                'class_distribution':target[..., 6][obj].long(),
                },
                {'logvar':log_obj_depth_variance,
                'var':obj_depth_variance,
                'logmean':log_obj_depth_mean,
                'mean':obj_depth_mean} ,
            )
        else:
                        return (
                self.lambda_box * box_loss,
                self.lambda_obj * object_loss,
                self.lambda_noobj * no_object_loss,
                self.lambda_class * class_loss,
                self.lambda_depth_bbox * obj_depth_loss_from_bbox,
                self.lambda_noobj_depth * noObj_depth_loss,
                self.lambda_depth_lidar * obj_depth_loss_picked_from_lidar,
                {
                'objDepthPredicted':obj_pred_depths,
                'log_obj_depth_pred': log_obj_depth_pred,
                'class_distribution':target[..., 6][obj].long(),
                },
                {'logvar':0,
                'var':0,
                'logmean':0,
                'mean':0} ,
            )


class SILogLoss(nn.Module):  # Main loss function used in AdaBins paper
    def __init__(self):
        super(SILogLoss, self).__init__()
        self.name = 'SILog'

    def forward(self, input, target, mask=None, interpolate=True):
        # docstring mit tensor shapes [in out etc]

        # input as float scalar
        if interpolate:
            # TODO check if this still works with [b,I,I,k,i,j] tensor shape
            input = nn.functional.interpolate(
                input, target.shape[-2:], mode='bilinear', align_corners=True)
        input = input.squeeze(dim=1)
        if mask is not None:
            # input = input[mask]
            target = target[mask]  # flatted
        # ab hier nur noch 1D Vector?!
        g = torch.log(input) - torch.log(target)
        # n, c, h, w = g.shape
        # norm = 1/(h*w)
        # Dg = norm * torch.sum(g**2) - (0.85/(norm**2)) * (torch.sum(g))**2

        Dg = torch.var(g) + 0.15 * torch.pow(torch.mean(g), 2)
        return 10 * torch.sqrt(Dg)


if __name__ == '__main__':
    scaled_anchors = (
        torch.tensor(config.ANCHORS)
        * torch.tensor(config.S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    ).to(config.DEVICE)
    dataset = YOLODataset(
            config.DATASET_TRAIN_CSV,
            config.IMAGE_DIR,
            config.BBOX_LABEL_DIR,
            config.DEPTH_NN_MAP_LABEL,
            S=config.S,
            anchors=config.ANCHORS,
            transform=None,
            cache_images=True
        )
    loader = DataLoader(dataset=dataset, batch_size=10)
    model = YOLOv3(num_classes=config.NUM_CLASSES).to(config.DEVICE)
    loss_fn = YoloLoss()

    for idx, (x, y, depth_target) in enumerate(loader):
        depth_target = depth_target.to(config.DEVICE, dtype=torch.float)
        x = x.to(config.DEVICE, dtype=torch.float)
        y0, y1, y2 = (
            y[0].to(config.DEVICE),
            y[1].to(config.DEVICE),
            y[2].to(config.DEVICE),
        )
        with torch.no_grad():
            out = model(x)
        with Timer('loss calc'):
            l_dense = loss_fn(out[2], y2, depth_target, scaled_anchors[2], config.S[2])# TODO fix missing args
        print(l_dense)
