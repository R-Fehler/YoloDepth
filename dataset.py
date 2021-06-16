"""
Creates a Pytorch dataset to load the Pascal VOC & MS COCO datasets
"""

from itertools import repeat
from albumentations import augmentations
from numpy.core.fromnumeric import ndim
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

import cv2 as cv

import tqdm
import fileinput
import glob
from sklearn.cluster import KMeans

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

    def getLabels(self,index):
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 2])

        bboxes = np.loadtxt(fname=label_path,delimiter=' ',ndmin=2, usecols=(0,1,2,3,4,5))

    # target nicht tiefe ziel generieren.... nicht hier sondern im Loss
    # ausser bei mapping werten.
    #im csv file: img,depthlabel,bboxes

    def get_complete_targets(self, index):
        # image = np.array(Image.open(img_path).convert("RGB"))
        # Load both bbox labels and 2D Depth Maps
        if(self.depth_labels_dir != None):
            label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 2])
            bboxes = np.loadtxt(fname=label_path, delimiter=" ", ndmin=2)
            if len(bboxes)==0:
                bboxes=np.array([[0,0,0,0,0,0],])
            
            if bboxes.shape[1]>5:
                bboxes = bboxes[:,0:6]
                bboxes = np.roll(bboxes, 5, axis=1).tolist()
            elif bboxes.shape[1]==5:
                bboxes = np.roll(bboxes, 4, axis=1).tolist()
            else: raise Exception(f'labelfile {label_path} misses entry, less than 5')

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
            if len(box)==5:
                x, y, width, height, class_label = box
            elif len(box)==6:
                x, y, width, height, depth, class_label = box
            x=clamp(x,0,0.999999999)
            y=clamp(y,0,0.999999999)
            width=clamp(width,0,0.999999999)
            height=clamp(height,0,0.999999999)
            

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
                    if (len(box)==6 and depth>0):
                        z = depth
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
        # savePath=os.path.join(self.depth_labels_dir,'griddata_nearest_withSky',self.annotations.iloc[index, 1])
        # Image.fromarray(dense_depth_target).save(savePath)
        return depth_target, dense_depth_target

    def prepare_griddata_depth_maps_without_sky(self,index):
        depth_target, dense_depth_target = self.prepare_griddata_depth_maps(index)
        depth_target_bw_inv=np.ones(depth_target.shape)
        depth_target_bw_inv[np.where(depth_target>0)]=0
        depth_target_bw_inv = depth_target_bw_inv.astype(np.uint8)

        dist = cv.distanceTransform(depth_target_bw_inv,cv.DIST_L2,cv.DIST_MASK_3)

        dense_depth_target[np.where(dist>config.DIST_THRESHOLD)] = 0
        savePath=os.path.join(self.depth_labels_dir,'griddata_nearest_wo_sky',self.annotations.iloc[index, 1])
        Image.fromarray(dense_depth_target).save(savePath)
        return 0



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
    loader = DataLoader(dataset=dataset,batch_size=48,shuffle=False, num_workers=128)
    for _ in tqdm.tqdm(loader):
        pass
        
def loadAllTextFilesIntoArray(filename_glob_pattern):
    return np.loadtxt(fileinput.input(sorted(glob.glob(filename_glob_pattern))),delimiter=' ',ndmin=2, usecols=(0,1,2,3,4,5))

def loadAllTextFilesIntoList(filename_glob_pattern):
    return [np.loadtxt(fn,delimiter=' ',ndmin=2,usecols=range(6)) for fn in sorted(glob.glob(filename_glob_pattern))]

def clusterDataSet():
    bboxes=loadAllTextFilesIntoArray('/home/fehler/PE_TOOL_RUNS_KNECHT5_LOCAL/ostring_seamseg_for_yoloDepth_CameraCoords/yoloLabels/*.txt')
    # TODO distance metric: d(box, centroid) = 1 − IOU(box, centroid)
    kmeans_anchors = KMeans(n_clusters=9).fit(bboxes[:,3:5])
    kmeans_depth = KMeans(n_clusters=9).fit(bboxes[:,5:6])
    return kmeans_anchors,kmeans_depth
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
    scaled_anchors = (
        torch.tensor(config.ANCHORS)
        * torch.tensor(config.S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    ).to(config.DEVICE)
    transform = config.test_transforms

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
    S = [13, 26, 52]
    scaled_anchors = torch.tensor(anchors) / (
        1 / torch.tensor(S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    )
    loader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=False)
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
        # print(boxes)
        plot_image(x[0].permute(1,2,0).detach().cpu(), pred_boxes=boxes,filename=f'datasetTest_{index}.png',target_boxes=boxes)


if __name__ == "__main__":
    # test()
    saveGridData()


def load_targets(self, index):
    # loads 1 image from dataset, returns img, original hw, resized hw
    img,targets,depth_target = self[index]
    return img,targets,depth_target
