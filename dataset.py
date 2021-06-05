"""
Creates a Pytorch dataset to load the Pascal VOC & MS COCO datasets
"""

from itertools import repeat
from albumentations import augmentations
from numpy.lib.type_check import _imag_dispatcher
from torch._C import dtype
import config
import numpy as np
import os
import pandas as pd
import torch
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import scipy.ndimage
from scipy.interpolate import griddata
from threading import Thread

from multiprocessing.pool import ThreadPool

import tqdm

from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader
from utils import (
    cells_to_bboxes,
    iou_width_height as iou,
    non_max_suppression as nms,
    plot_image
)

ImageFile.LOAD_TRUNCATED_IMAGES = True

class YOLODataset(Dataset):
    def __init__(
        self,
        csv_file,
        img_dir,
        label_dir,
        depth_labels_dir,
        anchors,
        image_size=416,
        S=[13, 26, 52],
        C=20,
        transform=None,
        cache_images=False,
    ):
        self.annotations = pd.read_csv(csv_file)
        self.annotations_depth = pd.read_csv(csv_file,delimiter=',')
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.depth_labels_dir = depth_labels_dir
        self.image_size = image_size
        self.transform = transform
        self.S = S
        self.anchors = torch.tensor(anchors[0] + anchors[1] + anchors[2])  # for all 3 scales
        self.num_anchors = self.anchors.shape[0]
        self.num_anchors_per_scale = self.num_anchors // 3
        self.C = C
        self.ignore_iou_thresh = 0.5
        self.n = len(self.annotations)
        self.imgs = [None] * self.n
        self.targets = [None] * self.n
        self.depth_targets = [None] * self.n
        self.isCached=False
        # if cache_images:
            # n = self.n
            # gb = 0  # Gigabytes of cached images
            # results = ThreadPool(8).imap(lambda x: load_targets(*x), zip(repeat(self), range(n)))  # 8 threads
            # pbar = tqdm.tqdm(enumerate(results), total=n)
            # for i, x in pbar:
            #     self.imgs[i], self.targets[i], self.depth_targets[i] = x  # img, hw_original, hw_resized = load_image(self, i)
            #     gb += self.imgs[i].nbytes
            #     # gb += self.targets[i].nbytes
            #     # gb += self.depth_targets[i].nbytes
            #     pbar.desc = f'Caching targets and imgs ca ({gb / 1E9:.1f}GB)'
            # pbar.close()
            # self.cached = True

    def __len__(self):
        return len(self.annotations)
        
    def __getitem__(self,index):
        if self.isCached:
            return self.imgs[index],self.targets[index],self.depth_targets[index]
        return self.get_complete_targets(index)

    # target nicht tiefe ziel generieren.... nicht hier sondern im Loss
    # ausser bei mapping werten.
    #im csv file: img,depthlabel,bboxes

    def get_complete_targets(self, index):
        # image = np.array(Image.open(img_path).convert("RGB"))
        if(self.depth_labels_dir != None):
            label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 2])
            bboxes = np.roll(np.loadtxt(fname=label_path, delimiter=" ", ndmin=2), 4, axis=1).tolist()
            img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
            depth_label_path = os.path.join(self.depth_labels_dir, self.annotations.iloc[index, 1])
            dense_depth_target = np.array(Image.open(depth_label_path).resize((416,416)),dtype=np.dtype('float'))/255.0
        elif config.DS_NAME=='Mapillary':
            label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
            bboxes = np.roll(np.loadtxt(fname=label_path, delimiter=" ", ndmin=2), 4, axis=1).tolist()
            img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
            dense_depth_target = 0 
        else:
            label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 2])
            bboxes = np.roll(np.loadtxt(fname=label_path, delimiter=" ", ndmin=2), 4, axis=1).tolist()
            img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
            dense_depth_target = 0 
        # TODO check resizing and scaling. is native res possible?
        # dense depth prediction 
        image = np.array(Image.open(img_path).convert("RGB").resize((416,416)),dtype=np.dtype('float'))/255.0
        image = np.transpose(image,(2,0,1)) # channels first


        # valid_depth_pixels = depth_target > 0
        # # outputs two vectors (x y) of length n
        # valid_depth_pixels_indices = np.where(valid_depth_pixels == True)
        # # this does output a 1d vector of length n
        # valid_depth_pixels_values = depth_target[valid_depth_pixels_indices]
        # # this might be error
        # grid_x, grid_y = np.mgrid[0:1536, 0:4096]
        # or this might be wrong: 2 xy vector for indices might need to be transposed or something
        # dense_depth_target = griddata(valid_depth_pixels_indices,valid_depth_pixels_values,(grid_x,grid_y),method='nearest')
        # dense_depth_target_lin = griddata(valid_depth_pixels_indices,valid_depth_pixels_values,(grid_x,grid_y),method='linear')
        # dense_depth_target_cub = griddata(valid_depth_pixels_indices,valid_depth_pixels_values,(grid_x,grid_y),method='cubic')
        # im = Image.fromarray(dense...)
        # im.save(fp) 

        if self.transform:
            augmentations = self.transform(image=image, bboxes=bboxes)
            image = augmentations["image"]
            bboxes = augmentations["bboxes"]

        # Below assumes 3 scale predictions (as paper) and same num of anchors per scale
        targets = [torch.zeros((self.num_anchors // 3, S, S, 7)) for S in self.S]
        for box in bboxes:
            iou_anchors = iou(torch.tensor(box[2:4]), self.anchors)
            anchor_indices = iou_anchors.argsort(descending=True, dim=0) # sort anchors by IoU with Target Bbox
            x, y, width, height, class_label = box
            has_anchor = [False] * 3  # each scale should have one anchor
            for anchor_idx in anchor_indices:
                scale_idx = anchor_idx // self.num_anchors_per_scale
                anchor_on_scale = anchor_idx % self.num_anchors_per_scale
                S = self.S[scale_idx]
                i, j = int(S * y), int(S * x)  # which cell
                anchor_taken = targets[scale_idx][anchor_on_scale, i, j, 0] # eg [1,2,3,0] check if previously anchor was taken
                if not anchor_taken and not has_anchor[scale_idx]:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = 1
                    x_cell, y_cell = S * x - j, S * y - i  # both between [0,1] scaled to cell coords
                    width_cell, height_cell = (
                        width * S,
                        height * S,
                    )  # can be greater than 1 since it's relative to cell
                    box_coordinates = torch.tensor(
                        [x_cell, y_cell, width_cell, height_cell]
                    )
                    # TODO this value is only viable when there is a mapped object
                    # Mapping output in den label txt files das mit rausschreibt
                    if (len(box)==6):
                        z = box[5]
                    # config einstellbar machen ob ich parameter target oder depthmap nehme
                    else:
                        z = 0
                    targets[scale_idx][anchor_on_scale, i, j, 1:5] = box_coordinates
                    targets[scale_idx][anchor_on_scale, i, j, 5] = z
                    targets[scale_idx][anchor_on_scale, i, j, 6] = int(class_label)
                    has_anchor[scale_idx] = True

                elif not anchor_taken and iou_anchors[anchor_idx] > self.ignore_iou_thresh:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = -1  # ignore prediction
        
        return image, tuple(targets), dense_depth_target
    



    def prepare_griddata_depth_maps(self, index):
        '''
        this function saves the dense depth map images 
        '''
        depth_label_path = os.path.join(self.depth_labels_dir, self.annotations.iloc[index, 1])
        depth_target = np.array(Image.open(depth_label_path))
        valid_depth_pixels = depth_target > 0
        # outputs two vectors (x y) of length n
        valid_depth_pixels_indices = np.where(valid_depth_pixels == True)
        # this does output a 1d vector of length n
        valid_depth_pixels_values = depth_target[valid_depth_pixels_indices]
        # this might be error
        grid_x, grid_y = np.mgrid[0:1536, 0:4096]
        # or this might be wrong: 2 xy vector for indices might need to be transposed or something
        dense_depth_target = griddata(valid_depth_pixels_indices,valid_depth_pixels_values,(grid_x,grid_y),method='nearest')
        # dense_depth_target = dense_depth_target.astype(np.dtype('int32'))
        savePath=os.path.join(self.depth_labels_dir,'griddata_linear',self.annotations.iloc[index, 1])
        Image.fromarray(dense_depth_target).save(savePath)
        return 0

class DepthDataset(Dataset):
    def __init__(
        self,
        csv_file,
        img_dir,
        depth_labels_dir,
        # anchors,
        image_size=416, # kitti height , width will be compressed
        S=[13,26,52],
        transform=None,
        ):
        self.annotations = pd.read_csv(csv_file,delimiter=',')
        self.img_dir = img_dir
        self.depth_labels_dir = depth_labels_dir
        self.image_size=image_size
        self.transform = transform
        self.S = S
        # self.anchors = torch.tensor(anchors[0] + anchors[1] +anchors[2])
        # self.num_anchors = self.anchors.shape[0]
        # self.num_anchors_per_scale = self.num_anchors // 3
        # self.ignore_iou_thresh = 0.5
    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        label_path = os.path.join(self.depth_labels_dir, self.annotations.iloc[index, 1])
        # bboxes = np.roll(np.loadtxt(fname=label_path, delimiter=" ", ndmin=2), 4, axis=1).tolist() # [class, x,y,w,h] roll to [x,y,w,h,class]
        depth_target = np.array(Image.open(label_path).resize((416,416)))/255.0
        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        image = np.array(Image.open(img_path).convert("RGB").resize((416,416)),dtype=np.dtype('float'))/255.0
        if self.transform:
            image = self.transform(image)
            depth_target = self.transform(depth_target)

        assert image.shape[0] == depth_target.shape[0], "height of depth and img different" 
        assert image.shape[1] == depth_target.shape[1],"witdh of depth and img different"
        targets = [torch.zeros((S,S,1)) for S in self.S] # [prob_obj,x,y,w,h,class] --> erstmal nur depth
        # targets = [torch.zeros((self.num_anchors_per_scale,S,S,6)) for S in self.S]  # [prob_obj,x,y,w,h,class] --> erstmal nur depth
        h = image.shape[0]
        w = image.shape[1]
        ch = image.shape[2]
        # assign the depth value of the center of each grid cell from depth_target img to target tensor
        # nn_indices = np.zeros(((np.ndim(depth_target),) + depth_target.shape), dtype=np.int32)

        # scipy.ndimage.morphology.distance_transform_edt(
        # depth_target==0, return_distances=False, return_indices=True,indices=nn_indices)
        # for scale_idx,S in enumerate(self.S):
        #     for i in range(S):
        #         for j in range(S):
        #             dt_avg = 0.
        #             n=0
        #             k=1
        #             for x in range (-k,k+1):
        #                 for y in range(-k,k+1):
        #                     nn_x=nn_indices[0][int(h*(1+i*2)/2//S)+x][int(w*(1+j*2)/2//S)+y]
        #                     nn_y=nn_indices[1][int(h*(1+i*2)/2//S)+x][int(w*(1+j*2)/2//S)+y]
        #                     # dt = depth_target[int(h*(1+i*2)/2//S)+x, 
        #                     # int(w*(1+j*2)/2//S)+y] 
        #                     dt = depth_target[nn_x,nn_y]
        #                     if dt>0.:
        #                         n += 1
        #                         dt_avg += dt
        #             if n>0:
        #                 dt_avg = dt_avg/n
                        
        #             else: 
        #                 dt_avg = 0.
        #             if(dt_avg>0.):
        #                 targets[scale_idx][i, j, 0] = dt_avg

            #                     for scale_idx,S in enumerate(self.S):
            # for i in range(S):
            #     for j in range(S):
            #         dt_avg = 0.
            #         n=0
            #         for x in range (-1,1):
            #             for y in range(-1,1):
            #                 dt = depth_target[int(h*(1+i*2)/2//S)+x, int(w*(1+j*2)/2//S)+y] 
            #                 if dt>0.:
            #                     n += 1
            #                     dt_avg += dt
            #         if n>0:
            #             dt_avg = dt_avg/n
            #         else: 
            #             dt_avg = 0.
            #         if(dt_avg>0.):
            #             targets[scale_idx][i, j, 0] = dt_avg
        image = np.transpose(image,(2,0,1))
        return image, tuple(targets), depth_target

def clamp(n, smallest, largest): return max(smallest, min(n, largest))
def testDepth():
    # dataset = DepthDataset("KITTI/kitti_eigen_train_files_with_gt.txt",
    # "KITTI/",
    # "KITTI/trainval_combined/"
    # ) 
    
    dataset = YOLODataset(
        "OstringDepthDataset/OstringDataSet.csv",
        "OstringDepthDataset/imgs/",
        "OstringDepthDataset/bbox_labels/",
        "OstringDepthDataset/depth_labels/",
        S=[13, 26, 52],
        anchors=config.ANCHORS,
        transform=None,
    )
    loader = DataLoader(dataset=dataset,batch_size=1,shuffle=False, num_workers=32)
    for idx,(x,y,depth_target) in enumerate(loader):
        y0,y1,y2=y

        # boxes = []

        # for i in range(y[0].shape[1]):
        #     anchor = scaled_anchors[i]
        #     print(anchor.shape)
        #     print(y[i].shape)
        #     boxes += cells_to_bboxes(
        #         y[i], is_preds=False, S=y[i].shape[2], anchors=anchor
        #     ).tolist()[0]
        # boxes = nms(boxes, iou_threshold=1, threshold=0.7, box_format="midpoint")
        # print(boxes)
        # plot_image(x[0].to("cpu"), boxes,filename=f'datasetTest_{idx}.png')

        Thread(target=saveImgAndDenseDepthLabel, args=(x,depth_target,f'datasettestNNDense/{idx}_datasettest.png'), daemon=True).start()
        # saveImgAndDenseDepthLabel(x,depth_target,f'datasettestNNDense/{idx}_datasettest.png')

def saveGridData():
    dataset = YOLODataset(
        "OstringDepthDataset/OstringDataSet.csv",
        "OstringDepthDataset/imgs/",
        "OstringDepthDataset/bbox_labels/",
        "OstringDepthDataset/depth_labels/",
        S=[13, 26, 52],
        anchors=config.ANCHORS,
        transform=None,
    )
    loader = DataLoader(dataset=dataset,batch_size=48,shuffle=False, num_workers=0)
    for _ in tqdm.tqdm(loader):
        pass
        

def saveImg(x,fp):
    arr = x.mul(255).add_(0.5).clamp_(0, 255).squeeze().to('cpu', torch.uint8).numpy()
    im = Image.fromarray(arr)
    im.save(fp)

def saveImgPredLabel(x,y,img,depthImg,fp):
    x = x.mul(255).add_(0.5).clamp_(0, 255).squeeze().to('cpu', torch.uint8).numpy()
    y = y.mul(255).add_(0.5).clamp_(0, 255).squeeze().to('cpu', torch.uint8).numpy()
    f, (ax1,ax2,ax3,ax4) = plt.subplots(1,4)
    ax1.imshow(x)
    ax2.imshow(y)
    ax3.imshow(img.permute(1,2,0).cpu())
    ax4.imshow(depthImg.cpu())
    f.savefig(fp,dpi=150)
    
def saveImgBBoxDepthLabel(x,y,depthImg,fp):
    x = x.mul(255).add_(0.5).clamp_(0, 255).squeeze().to('cpu', torch.uint8).numpy()
    y = y.mul(255).add_(0.5).clamp_(0, 255).squeeze().to('cpu', torch.uint8).numpy()

def saveImgAndLabel(x,y,depthImg,fp):
    # labels =[l/torch.max(l).item() for l in labels]
    # depth_target =[dt/torch.max(dt).item() for dt in depth_target]
    y = y.squeeze().to('cpu', torch.uint8).numpy()
    f, (ax1,ax2,ax3) = plt.subplots(1,3)
    ax1.imshow(x.squeeze().permute(1,2,0).cpu())
    ax2.imshow(y)
    ax3.imshow(depthImg.permute(1,2,0).cpu())
    f.savefig(fp,dpi=150)
def saveImgAndDenseDepthLabel(x,depthImg,fp):
    # labels =[l/torch.max(l).item() for l in labels]
    # depth_target =[dt/torch.max(dt).item() for dt in depth_target]
    f, (ax1,ax3) = plt.subplots(1,2)
    ax1.imshow(x.squeeze().permute(1,2,0).cpu())
    ax3.imshow(depthImg.squeeze().cpu())
    f.savefig(fp,dpi=150)
    plt.clf()
def test():
    anchors = config.ANCHORS

    transform = config.test_transforms

    dataset = YOLODataset(
        "OstringDepthDataset/OstringDataSet.csv",
        "OstringDepthDataset/imgs/",
        "OstringDepthDataset/bbox_labels/",
        "OstringDepthDataset/depth_labels/",
        S=[13, 26, 52],
        anchors=anchors,
        transform=None,
    )
    S = [13, 26, 52]
    scaled_anchors = torch.tensor(anchors) / (
        1 / torch.tensor(S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    )
    loader = DataLoader(dataset=dataset, batch_size=1, shuffle=True)
    for index , (x, y, depthtarget) in enumerate(loader):
        boxes = []

        for i in range(y[0].shape[1]):
            anchor = scaled_anchors[i]
            print(anchor.shape)
            print(y[i].shape)
            boxes += cells_to_bboxes(
                y[i], is_preds=False, S=y[i].shape[2], anchors=anchor
            ).tolist()[0]
        boxes = nms(boxes, iou_threshold=1, threshold=0.7, box_format="midpoint")
        print(boxes)
        plot_image(x[0].to("cpu"), boxes,filename=f'datasetTest_{index}.png')


if __name__ == "__main__":
    saveGridData()


def load_targets(self, index):
    # loads 1 image from dataset, returns img, original hw, resized hw
    img,targets,depth_target = self[index]
    return img,targets,depth_target
