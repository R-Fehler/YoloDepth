import random

import torch
import torch.nn as nn
from torch._C import device, dtype
from torch.utils.data import DataLoader, Dataset

import config
from dataset import DepthDataset, YOLODataset
from model import YOLOv3
from Timer import Timer
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

    def forward(self, predictions, target, depth_target, anchors, scale_S, mask=None, interpolate=True):
        anchors_const = anchors
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

        # ================== #
        #   FOR DEPTH LOSS   #
        # ================== #

        depth_loss = torch.tensor(0.0, device=config.DEVICE)
        if depth_target.dim() > 1:
            bboxes = cells_to_bboxes(
                predictions, anchors_const, scale_S, is_preds=True, reshape=False).to(config.DEVICE)
            pred_centers = bboxes[:, :, :, :, 1:3] * config.IMAGE_SIZE
            pred_centers = pred_centers.clamp(0, config.IMAGE_SIZE-1)
            pred_depths = bboxes[:, :, :, :, 6]
            pred_depths = pred_depths.clamp(0.1, 255)
            depth_target = depth_target.clamp(0.1, 255)
            # depth target is bs x 416 x 416
            log_depth_target = torch.log(depth_target)
            # depth_pred is bs x 3 x 13 x 13
            log_depth_pred = torch.log(pred_depths)
            # shape is bs x 3 x 13 x 13 x 2
            # the x,y indices for selecting the depth value in the dense depth map
            indices = pred_centers.type(torch.int64).detach()

            target_depths_selected_with_pred_centers = torch.zeros_like(
                log_depth_pred, device=config.DEVICE)
            # with this implementation the calc time is 0.04 instead of 0.014 which is acceptable
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

            g = target_depths_selected_with_pred_centers - log_depth_pred
            g = g.view((g.shape[0], -1))
            Dg = torch.var(g, dim=1) + 0.15 * torch.pow(torch.mean(g, dim=1), 2)
            depth_loss = 10 * torch.sqrt(Dg)
            depth_loss = depth_loss.mean()  # SCALAR

        # print("__________________________________")
        # print(self.lambda_box * box_loss)
        # print(self.lambda_obj * object_loss)
        # print(self.lambda_noobj * no_object_loss)
        # print(self.lambda_class * class_loss)
        # print(self.lambda_depth * depth_loss)
        # print("\n")

        return (
            self.lambda_box * box_loss,
            self.lambda_obj * object_loss,
            self.lambda_noobj * no_object_loss,
            self.lambda_class * class_loss,
            self.lambda_depth * depth_loss
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
        "OstringDepthDataset/OstringDataSet.csv",
        "OstringDepthDataset/imgs/",
        "OstringDepthDataset/bbox_labels/",
        "OstringDepthDataset/depth_labels/",
        S=[13, 26, 52],
        anchors=config.ANCHORS,
        transform=None,
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
            l_dense = loss_fn(out[2], y2, depth_target, scaled_anchors[2], config.S[2], interpolate=True)
        print(l_dense)
